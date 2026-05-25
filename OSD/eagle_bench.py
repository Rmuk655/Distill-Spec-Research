"""
EAGLE benchmark for Qwen3-0.6B — three phases:

  python eagle_bench.py gen   --base Qwen/Qwen3-0.6B --data data/gsm8k_train.jsonl --out data/eagle_tmp --n 500
  python eagle_bench.py train --base Qwen/Qwen3-0.6B --tmp data/eagle_tmp --ckpt checkpoints/eagle-qwen3-06b --steps 2000
  python eagle_bench.py eval  --base Qwen/Qwen3-0.6B --ckpt checkpoints/eagle-qwen3-06b --data data/gsm8k_30.jsonl --K 3

Phase gen:   Run base model on prompts, save (hidden_states, input_ids, loss_mask) tensors.
Phase train: Train 1-layer EAGLE head to predict next hidden state + next token.
Phase eval:  Tree speculative decoding — measure block efficiency vs baseline greedy.
"""

import argparse, json, os, math, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# EAGLE head architecture
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq=32768, base=10000):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._build_cache(max_seq)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None], persistent=False)

    def forward(self, x, seq_len):
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q, k, cos, sin, pos_ids):
    cos = cos.squeeze(0).squeeze(0)[pos_ids].unsqueeze(1)
    sin = sin.squeeze(0).squeeze(0)[pos_ids].unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class EagleAttention(nn.Module):
    def __init__(self, hidden, heads, kv_heads):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = hidden // heads
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(q, T)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        q, k = apply_rotary(q, k, cos, sin, pos)

        # GQA: expand k/v if kv_heads < heads
        if self.kv_heads < self.heads:
            rep = self.heads // self.kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        if attn_mask is not None:
            scores = scores + attn_mask
        scores = F.softmax(scores, dim=-1)
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(out)


class EagleMLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class EagleHead(nn.Module):
    """
    1-layer transformer that predicts h_{t+1} from (h_t, token_t).
    lm_head (shared from base) maps h to logits.
    """
    def __init__(self, hidden_size, vocab_size, num_heads=16, kv_heads=8,
                 intermediate_size=3072, embed_dim=None):
        super().__init__()
        embed_dim = embed_dim or hidden_size
        # project token embedding into hidden space
        self.embed_proj = nn.Linear(embed_dim, hidden_size, bias=False)
        # norm before layer
        self.input_norm = RMSNorm(hidden_size)
        # single transformer layer
        self.attn = EagleAttention(hidden_size, num_heads, kv_heads)
        self.mlp = EagleMLP(hidden_size, intermediate_size)
        self.attn_norm = RMSNorm(hidden_size)
        self.mlp_norm = RMSNorm(hidden_size)
        # lm_head — loaded from base model weights, frozen
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states, token_embeds, attn_mask=None):
        """
        hidden_states: (B, T, H)  — base model hidden states at each position
        token_embeds:  (B, T, E)  — base model token embeddings at each position
        Returns: predicted next hidden states (B, T, H)
        """
        x = hidden_states + self.embed_proj(token_embeds)
        x = self.input_norm(x)
        # causal mask
        if attn_mask is None:
            T = x.shape[1]
            causal = torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype)
            causal = torch.triu(causal, diagonal=1)
            attn_mask = causal[None, None]  # (1,1,T,T)
        residual = x
        x = self.attn(x, attn_mask) + residual
        residual = x
        x = self.mlp(self.mlp_norm(x)) + residual
        return x

    def predict_logits(self, hidden_states, token_embeds, attn_mask=None):
        h = self.forward(hidden_states, token_embeds, attn_mask)
        return self.lm_head(h)


# ---------------------------------------------------------------------------
# Phase 1: generate hidden state training data
# ---------------------------------------------------------------------------

def phase_gen(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[gen] Loading base model {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = _load_prompts(args.data, args.n)
    print(f"[gen] Generating hidden states for {len(prompts)} prompts → {out_dir}")

    saved = 0
    for idx, prompt in enumerate(tqdm(prompts)):
        out_path = out_dir / f"{idx:05d}.pt"
        if out_path.exists():
            saved += 1
            continue
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc.input_ids.cuda()
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
        # last hidden layer: (1, T, H)
        hidden = out.hidden_states[-1].cpu().to(torch.float16)
        ids_cpu = input_ids.cpu()
        # loss_mask: 1 for all tokens except last (which has no next target)
        T = ids_cpu.shape[1]
        loss_mask = torch.ones(T, dtype=torch.float32)
        loss_mask[-1] = 0.0
        torch.save({
            "hidden_state": hidden[0],   # (T, H)
            "input_ids": ids_cpu[0],     # (T,)
            "loss_mask": loss_mask,      # (T,)
        }, out_path)
        saved += 1

    print(f"[gen] Saved {saved} files to {out_dir}")


# ---------------------------------------------------------------------------
# Phase 2: train EAGLE head
# ---------------------------------------------------------------------------

def phase_train(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import glob

    print(f"[train] Loading base model config + lm_head from {args.base}")
    base_cfg = AutoConfig.from_pretrained(args.base, trust_remote_code=True)
    hidden_size = base_cfg.hidden_size
    vocab_size = base_cfg.vocab_size
    num_heads = base_cfg.num_attention_heads
    kv_heads = getattr(base_cfg, "num_key_value_heads", num_heads)
    intermediate = base_cfg.intermediate_size

    # load lm_head weights from base
    print(f"[train] Extracting lm_head from base model (fp16)")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    embed_dim = base_model.model.embed_tokens.embedding_dim

    head = EagleHead(hidden_size, vocab_size, num_heads, kv_heads, intermediate, embed_dim)
    head.lm_head.weight.data = base_model.lm_head.weight.data.clone()
    for p in head.lm_head.parameters():
        p.requires_grad = False
    # store embed_tokens for data loading
    embed_tokens = base_model.model.embed_tokens
    embed_tokens.eval()
    for p in embed_tokens.parameters():
        p.requires_grad = False
    del base_model
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = head.to(device).to(torch.float16)
    embed_tokens = embed_tokens.to(device)

    # trainable params only (not lm_head)
    trainable = [p for p in head.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"[train] Trainable params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    data_files = sorted(glob.glob(str(Path(args.tmp) / "*.pt")))
    if not data_files:
        sys.exit(f"[train] No .pt files in {args.tmp}. Run phase gen first.")
    print(f"[train] {len(data_files)} training files, {args.steps} steps")

    ckpt_dir = Path(args.ckpt)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    head.train()
    step = 0
    losses = []

    while step < args.steps:
        import random
        random.shuffle(data_files)
        for fpath in data_files:
            if step >= args.steps:
                break
            data = torch.load(fpath, map_location="cpu")
            hidden = data["hidden_state"].unsqueeze(0).to(device, torch.float16)  # (1,T,H)
            ids = data["input_ids"].unsqueeze(0).to(device)                        # (1,T)
            loss_mask = data["loss_mask"].to(device)                               # (T,)
            T = ids.shape[1]
            if T < 4:
                continue

            with torch.no_grad():
                embeds = embed_tokens(ids).to(torch.float16)  # (1,T,E)

            # targets: next hidden state at each position
            # input:  h[0..T-2], token_embed[0..T-2]
            # target: h[1..T-1]
            h_in = hidden[:, :-1, :]           # (1, T-1, H)
            e_in = embeds[:, :-1, :]           # (1, T-1, E)
            h_tgt = hidden[:, 1:, :].detach()  # (1, T-1, H)
            ids_tgt = ids[:, 1:]               # (1, T-1)
            mask = loss_mask[:-1]              # (T-1,)

            pred_h = head(h_in, e_in)          # (1, T-1, H)
            logits = head.lm_head(pred_h)      # (1, T-1, V)

            # feature regression loss (cosine)
            feat_loss = (1.0 - F.cosine_similarity(pred_h, h_tgt, dim=-1))  # (1, T-1)
            feat_loss = (feat_loss[0] * mask).sum() / (mask.sum() + 1e-8)

            # cross-entropy token prediction loss
            ce_loss = F.cross_entropy(
                logits[0], ids_tgt[0], reduction="none"
            )  # (T-1,)
            ce_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)

            loss = 0.1 * feat_loss + ce_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            losses.append(loss.item())
            step += 1

            if step % 100 == 0:
                avg = sum(losses[-100:]) / len(losses[-100:])
                print(f"  step {step}/{args.steps}  loss={avg:.4f}")

            if step % 500 == 0:
                _save_ckpt(head, ckpt_dir, step, hidden_size, vocab_size, num_heads, kv_heads, intermediate, embed_dim)

    _save_ckpt(head, ckpt_dir, args.steps, hidden_size, vocab_size, num_heads, kv_heads, intermediate, embed_dim)
    print(f"[train] Done. Checkpoint: {ckpt_dir}")


def _save_ckpt(head, ckpt_dir, step, hidden_size, vocab_size, num_heads, kv_heads, intermediate, embed_dim):
    path = ckpt_dir / f"step{step}.pt"
    torch.save({
        "state_dict": head.state_dict(),
        "meta": {
            "hidden_size": hidden_size, "vocab_size": vocab_size,
            "num_heads": num_heads, "kv_heads": kv_heads,
            "intermediate_size": intermediate, "embed_dim": embed_dim,
        }
    }, path)
    # also save as latest
    torch.save({"state_dict": head.state_dict(), "meta": {
        "hidden_size": hidden_size, "vocab_size": vocab_size,
        "num_heads": num_heads, "kv_heads": kv_heads,
        "intermediate_size": intermediate, "embed_dim": embed_dim,
    }}, ckpt_dir / "latest.pt")
    print(f"  [ckpt] saved {path}")


# ---------------------------------------------------------------------------
# Phase 3: evaluate block efficiency
# ---------------------------------------------------------------------------

def phase_eval(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    print(f"[eval] Loading base model {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    base_model.eval()

    # load EAGLE head
    ckpt_path = Path(args.ckpt) / "latest.pt"
    if not ckpt_path.exists():
        ckpt_path = sorted(Path(args.ckpt).glob("step*.pt"))[-1]
    print(f"[eval] Loading EAGLE head from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt["meta"]
    head = EagleHead(**meta)
    head.load_state_dict(ckpt["state_dict"])
    device = next(base_model.parameters()).device
    head = head.to(device).to(torch.float16)
    head.eval()

    embed_tokens = base_model.model.embed_tokens
    embed_tokens.eval()

    prompts_data = _load_prompts_full(args.data, args.n_eval)
    print(f"[eval] Evaluating on {len(prompts_data)} prompts, K={args.K}")

    total_accepted = 0
    total_calls = 0
    total_tokens = 0

    for item in tqdm(prompts_data):
        prompt = item if isinstance(item, str) else item.get("prompt", item.get("question", ""))
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = enc.input_ids.to(device)

        acc, calls, toks = _eagle_speculative_decode(
            base_model, head, embed_tokens, input_ids,
            K=args.K, max_new=args.max_new, temperature=args.temperature, tok=tok
        )
        total_accepted += acc
        total_calls += calls
        total_tokens += toks

    be = total_accepted / total_calls if total_calls > 0 else 0.0
    print(f"\n[eval] Results:")
    print(f"  EAGLE block efficiency (K={args.K}): {be:.6f}")
    print(f"  Total target calls: {total_calls}")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Mean tokens/call: {total_tokens/total_calls:.3f}")

    # Save to results DB if available
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from results_db import insert_run
        from datetime import datetime, timezone
        run_tag = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_eagle_K{args.K}_T{args.temperature}"
        insert_run({
            "run_tag": run_tag,
            "draft_label": "eagle-qwen3-0.6b",
            "draft_path": str(Path(args.ckpt) / "latest.pt"),
            "target_path": args.base,
            "loss_name": "eagle",
            "train_steps": 0,
            "dataset": Path(args.data).stem,
            "n_prompts": len(prompts_data),
            "mode": "eagle",
            "K": args.K,
            "L": 0,
            "temperature": args.temperature,
            "block_eff": be,
        })
        print(f"  Saved to results.db as {run_tag}")
    except Exception as e:
        print(f"  (results_db not available: {e})")

    return be


@torch.no_grad()
def _eagle_speculative_decode(base_model, head, embed_tokens, input_ids,
                               K=3, max_new=100, temperature=1.0, tok=None):
    """
    EAGLE speculative decoding (chain, not full tree for simplicity).
    Returns (total_accepted_tokens, total_target_calls, total_generated_tokens).
    """
    device = input_ids.device
    accepted = 0
    calls = 0
    generated = 0

    # prefix KV cache from base model
    out = base_model(input_ids, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    # last hidden state of prefix: (1, T, H)
    h_last = out.hidden_states[-1][:, -1:, :]   # (1, 1, H)
    last_token_id = input_ids[:, -1:]            # (1, 1)
    # Save the target's prediction for the FIRST draft token in every new block.
    # verify_logits[0, i] is the prediction AFTER seeing draft token d_i (i.e. for d_{i+1}),
    # so acceptance of d_0 requires the prefix logit — not anything in verify_logits.
    prefix_logit = out.logits[0, -1]             # (V,) — updated after each block

    eos_id = tok.eos_token_id if tok else 2

    while generated < max_new:
        # --- Draft K tokens using EAGLE head ---
        draft_ids = []
        draft_probs = []
        h_cur = h_last
        tok_cur = last_token_id

        for _ in range(K):
            e_cur = embed_tokens(tok_cur).to(torch.float16)   # (1,1,E)
            pred_h = head(h_cur, e_cur)                       # (1,1,H)
            logits = head.lm_head(pred_h[:, -1, :])           # (1, V)
            if temperature > 1e-5:
                probs = F.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs[0], 1).unsqueeze(0)  # (1,1)
            else:
                next_tok = logits.argmax(-1).unsqueeze(0)
                probs = F.softmax(logits, dim=-1)
            draft_ids.append(next_tok)
            draft_probs.append(probs[0, next_tok[0, 0]])
            h_cur = pred_h
            tok_cur = next_tok

        # --- Verify K draft tokens with base model in one call ---
        # Append all draft tokens to the sequence and do one forward pass
        draft_seq = torch.cat(draft_ids, dim=1)  # (1, K)
        verify_out = base_model(
            draft_seq, past_key_values=past_kv, use_cache=True,
            output_hidden_states=True
        )
        # logits at each of the K positions
        verify_logits = verify_out.logits  # (1, K, V)
        verify_h = verify_out.hidden_states[-1]  # (1, K, H)
        calls += 1

        # --- Acceptance loop ---
        n_accepted = 0
        new_past = past_kv  # tentative; updated below on acceptance
        for i in range(K):
            # Bug fix (off-by-one): verify_logits[0, i] is the target's output AT
            # position i, which predicts token i+1 — NOT the acceptance probability
            # for draft token d_i.  Correct indexing:
            #   i == 0  → use prefix_logit (target's prediction before any draft token)
            #   i  > 0  → use verify_logits[0, i-1] (prediction after d_0..d_{i-1})
            q_logit = prefix_logit if i == 0 else verify_logits[0, i - 1]  # (V,)
            if temperature > 1e-5:
                q_prob = F.softmax(q_logit / temperature, dim=-1)
            else:
                q_prob = F.softmax(q_logit, dim=-1)

            tok_i = draft_ids[i][0, 0]
            p_draft_i = draft_probs[i].item()
            p_target_i = q_prob[tok_i].item()

            accept_prob = min(1.0, p_target_i / (p_draft_i + 1e-9))
            if torch.rand(1).item() < accept_prob:
                n_accepted += 1
                accepted += 1
                generated += 1
                if tok_i.item() == eos_id:
                    # update state and return
                    past_kv = verify_out.past_key_values
                    h_last = verify_h[:, i:i+1, :]
                    last_token_id = tok_i.unsqueeze(0).unsqueeze(0)
                    return accepted, calls, generated
            else:
                # reject: sample replacement token from residual distribution
                residual = F.relu(q_prob - torch.tensor(p_draft_i, device=device) *
                                  F.one_hot(tok_i, q_prob.shape[0]).float())
                residual_sum = residual.sum()
                if residual_sum > 1e-9:
                    replacement = torch.multinomial(residual / residual_sum, 1)
                else:
                    replacement = q_prob.argmax().unsqueeze(0)
                accepted += 1  # replacement counts as 1 accepted token
                generated += 1
                if replacement.item() == eos_id:
                    return accepted, calls, generated
                # update running state to replacement token position
                past_kv = verify_out.past_key_values
                h_last = verify_h[:, i:i+1, :]
                last_token_id = replacement.unsqueeze(0).unsqueeze(0)
                # prefix_logit for next block: target's prediction at position i
                # (best approximation — prediction for d_{i+1} given prefix+d_0..d_i)
                prefix_logit = verify_logits[0, i]
                break
        else:
            # all K accepted — bonus: sample one more from target distribution at last position
            past_kv = verify_out.past_key_values
            bonus_logit = verify_logits[0, -1]
            # prefix_logit for next block: prediction after the last draft token
            # (same tensor as bonus_logit — best approximation without extra fwd pass)
            prefix_logit = verify_logits[0, -1]
            if temperature > 1e-5:
                bonus_prob = F.softmax(bonus_logit / temperature, dim=-1)
                bonus_tok = torch.multinomial(bonus_prob, 1)
            else:
                bonus_tok = bonus_logit.argmax().unsqueeze(0)
            accepted += 1
            generated += 1
            if bonus_tok.item() == eos_id:
                return accepted, calls, generated
            h_last = verify_h[:, -1:, :]
            last_token_id = bonus_tok.unsqueeze(0).unsqueeze(0)

    return accepted, calls, generated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompts(data_path, n=None):
    prompts = []
    with open(data_path) as f:
        for line in f:
            obj = json.loads(line)
            p = obj.get("prompt") or obj.get("question") or obj.get("instruction") or obj.get("text", "")
            prompts.append(p.strip())
    if n:
        prompts = prompts[:n]
    return prompts


def _load_prompts_full(data_path, n=None):
    items = []
    with open(data_path) as f:
        for line in f:
            items.append(json.loads(line))
    if n:
        items = items[:n]
    return items


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EAGLE benchmark for Qwen3-0.6B")
    sub = parser.add_subparsers(dest="phase", required=True)

    # gen
    g = sub.add_parser("gen", help="Generate hidden state training data")
    g.add_argument("--base", default="Qwen/Qwen3-0.6B")
    g.add_argument("--data", default="data/gsm8k_train.jsonl")
    g.add_argument("--out", default="data/eagle_tmp")
    g.add_argument("--n", type=int, default=500, help="Max prompts to process")

    # train
    t = sub.add_parser("train", help="Train EAGLE head")
    t.add_argument("--base", default="Qwen/Qwen3-0.6B")
    t.add_argument("--tmp", default="data/eagle_tmp")
    t.add_argument("--ckpt", default="checkpoints/eagle-qwen3-06b")
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--lr", type=float, default=3e-5)

    # eval
    e = sub.add_parser("eval", help="Evaluate block efficiency")
    e.add_argument("--base", default="Qwen/Qwen3-0.6B")
    e.add_argument("--ckpt", default="checkpoints/eagle-qwen3-06b")
    e.add_argument("--data", default="data/gsm8k_30.jsonl")
    e.add_argument("--K", type=int, default=3)
    e.add_argument("--max_new", type=int, default=100)
    e.add_argument("--temperature", type=float, default=0.6)
    e.add_argument("--n_eval", type=int, default=None)

    args = parser.parse_args()

    if args.phase == "gen":
        phase_gen(args)
    elif args.phase == "train":
        phase_train(args)
    elif args.phase == "eval":
        phase_eval(args)


if __name__ == "__main__":
    main()
