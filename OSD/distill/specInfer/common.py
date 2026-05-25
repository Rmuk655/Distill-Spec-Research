import torch
import time

from dataclasses import dataclass


@dataclass
class InputAndCache:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: torch.Tensor

    # encoder-decoder only fields
    labels: torch.Tensor = None
    decoder_input_ids: torch.Tensor = None


@dataclass
class OutputAndCache:
    generated_len: int
    output_ids: torch.Tensor
    output_logits: torch.Tensor
    output_distribution: torch.Tensor
    past_key_values: torch.Tensor


########################### Sampling ########################
def target_sample_from_distribution(target_distribution, draft_distribution):
    distribution = (target_distribution - draft_distribution)
    distribution = torch.max(distribution,
                             torch.zeros_like(distribution))
    if (distribution.sum(dim=-1, keepdim=True) == 0).any():
        distribution = torch.where(
            distribution == 0, distribution + 1e-10, distribution)
        print("[Warning] Distribution contains zero values")
    distribution = distribution / distribution.sum(dim=-1, keepdim=True)
    return torch.multinomial(distribution, num_samples=1).squeeze(-1)

########################### Utility ########################


def slice_past_key_values(past_key_values, start_idx, slice_len):
    new_past = []
    for idx in range(len(past_key_values)):
        new_past.append(
            (
                past_key_values[idx][0][:, :,
                                        start_idx:start_idx+slice_len, :],
                past_key_values[idx][1][:, :,
                                        start_idx:start_idx+slice_len, :],
            )
        )
    return tuple(new_past)


def slice_past_key_values_seq2seq(past_key_values, start_idx, slice_len):
    new_past = []
    for idx in range(len(past_key_values)):
        new_past.append(
            (
                past_key_values[idx][0][:, :,
                                        start_idx:start_idx+slice_len, :],
                past_key_values[idx][1][:, :,
                                        start_idx:start_idx+slice_len, :],
                past_key_values[idx][2][:, :,
                                        start_idx:start_idx+slice_len, :],
                past_key_values[idx][3][:, :,
                                        start_idx:start_idx+slice_len, :],

            )
        )
    return tuple(new_past)


def slice_mqa_past_key_values(past_key_values, start_idx, slice_len):
    new_past = []
    for idx in range(len(past_key_values)):
        new_past.append(
            past_key_values[idx][:, start_idx:start_idx+slice_len, :]
        )
    return tuple(new_past)


def get_kv_seq_len(past_key_values):
    """Get sequence length from either DynamicCache or legacy tuple KV cache."""
    from transformers import DynamicCache
    if isinstance(past_key_values, DynamicCache):
        return past_key_values.layers[0].keys.shape[2]
    return past_key_values[0][0].shape[2]


def crop_past_key_values(past_key_values, max_len):
    from transformers import DynamicCache
    if isinstance(past_key_values, DynamicCache):
        new_cache = DynamicCache()
        for layer_idx in range(len(past_key_values.layers)):
            k = past_key_values.layers[layer_idx].keys[:, :, :max_len, :].contiguous()
            v = past_key_values.layers[layer_idx].values[:, :, :max_len, :].contiguous()
            new_cache.update(k, v, layer_idx)
        return new_cache
    return slice_past_key_values(past_key_values, 0, max_len)


def crop_past_key_values_seq2seq(past_key_values, max_len):
    return slice_past_key_values_seq2seq(past_key_values, 0, max_len)


def crop_mqa_past_key_values(past_key_values, max_len):
    return slice_mqa_past_key_values(past_key_values, 0, max_len)


def sychronize_time():
    torch.cuda.synchronize()
    return time.time()

# convert a list of 1d tensors to a single 2d tensor
# if those 1d tensors have different shapes, pad them to the longest length


def pad_to_2d(tensor_list, pad_token_id, max_len=None):
    if not isinstance(tensor_list[0], torch.Tensor):
        tensor_list = [torch.tensor(t).reshape(1, -1) for t in tensor_list]
    if max_len is None:
        max_len = max([t.shape[-1] for t in tensor_list])
    assert max_len > 0

    # Pad each tensor to the max length and stack them to form a 2D tensor
    result = torch.cat(
        [
            torch.nn.functional.pad(
                tensor, (0, max_len - tensor.shape[-1]),
                value=pad_token_id
            )
            for tensor in tensor_list
        ],
        dim=0
    )
    return result
