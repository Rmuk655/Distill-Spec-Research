"""
verifier_safe.py — safe wrapper around TreeVerifier (verifiers/verifier.py).

verifiers/verifier.py is the researcher's source of truth. We do NOT modify it.
Instead, every call to TreeVerifier goes through safe_verify() here so that:

  • exceptions are caught and logged with full context for the researcher to debug
  • the eval / train loop can skip the failing prompt and continue cleanly

Evidence is written to verifier_errors.log so the researcher can reproduce and fix
bugs in verifier.py / node.py themselves without us touching those files.

Usage in main.py:
    from verifier_safe import safe_verify, VerifierError
    ver_node, res_token = safe_verify(
        q_paths, q_prefixes, q_probs_dict, p_probs_dict, verification_algo,
        prompt=prompt, K=K, L=L, p_temp=p_temp, prompt_idx=prompt_idx,
    )

Callers should catch VerifierError one level up to skip the failing prompt:
    try:
        speculative_decoding_loop(...)
    except VerifierError:
        print("[skip] verifier exception — prompt skipped; see verifier_errors.log")
        continue
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import traceback

# ---------------------------------------------------------------------------
# Rotating log — 10 MB per file, 3 backups.  Written next to this file so
# the researcher can find it regardless of the working directory.
# ---------------------------------------------------------------------------
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verifier_errors.log")
_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
)
_stderr_handler = logging.StreamHandler()
_stderr_handler.setLevel(logging.WARNING)
_stderr_handler.setFormatter(logging.Formatter("%(levelname)s [verifier_safe] %(message)s"))

_log = logging.getLogger("verifier_safe")
_log.setLevel(logging.DEBUG)
_log.propagate = False
if not _log.handlers:
    _log.addHandler(_file_handler)
    _log.addHandler(_stderr_handler)


class VerifierError(RuntimeError):
    """
    Raised by safe_verify() when verifier.py throws an exception.

    Callers (eval.py, train.py, inference.py) catch this to skip the prompt.
    The full traceback + context are already written to verifier_errors.log.
    """


def safe_verify(
    q_paths,
    q_prefixes,
    q_probs_dict,
    p_probs_dict,
    mode: str,
    *,
    prompt: str = "<unknown>",
    K=None,
    L=None,
    p_temp=None,
    prompt_idx=None,
):
    """
    Construct a TreeVerifier and call verify(mode), catching any exception.

    On success: returns (ver_node, res_token) — same as tree_verifier.verify().
    On failure: logs the prompt + tree inputs to verifier_errors.log, then
                raises VerifierError so the caller can skip this prompt.

    Parameters logged to verifier_errors.log for researcher debugging:
        prompt_idx, mode, K, L, p_temp, prompt text,
        q_paths (draft tree paths), q_prefixes (node labels), full traceback.
    """
    from verifier import TreeVerifier  # inside function to avoid circular import

    try:
        tv = TreeVerifier(q_paths, q_prefixes, q_probs_dict, p_probs_dict)
        return tv.verify(mode)

    except Exception:
        tb = traceback.format_exc()
        # Truncate very long q_paths / q_prefixes so the log stays readable.
        _q_paths_repr = repr(q_paths[:4]) + (" ..." if len(q_paths) > 4 else "")
        _q_pfx_repr   = repr(q_prefixes[:16]) + (" ..." if len(q_prefixes) > 16 else "")

        _log.error(
            "VERIFIER EXCEPTION\n"
            "  prompt_idx : %s\n"
            "  mode       : %s\n"
            "  K          : %s\n"
            "  L          : %s\n"
            "  p_temp     : %s\n"
            "  prompt     : %r\n"
            "  q_paths    : %s\n"
            "  q_prefixes : %s\n"
            "--- traceback ---\n%s",
            prompt_idx,
            mode,
            K,
            L,
            p_temp,
            prompt,
            _q_paths_repr,
            _q_pfx_repr,
            tb,
        )
        raise VerifierError(
            f"verifier raised for prompt_idx={prompt_idx!r} mode={mode!r} — "
            f"full context written to {_LOG_PATH}"
        ) from None
