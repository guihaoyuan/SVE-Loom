#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SimdBench generator with optional SVE-focused "repair" pipeline:

- Generates completions for SimdBench-style jsonl problems:
    each line: {"task_id": ..., "prompt": ..., "task": "...", "intrinsic": "..."} (task/intrinsic optional)

- Outputs jsonl samples:
    {"task_id": <id>, "completion": <code_completion>}

- Supports torchrun multi-GPU inference via data-parallel sharding.
  Rank 0 merges shard outputs to the requested --output path.

Optional enhancements:
  1) Intrinsics NAME whitelist repair (fix typos / wrong intrinsic names)  [SVE]
  2) Intrinsics CALL-SHAPE repair using signatures from whitelist.json     [SVE]
  3) Optional remote functional-correctness feedback loop via SSH (per-sample):
     - For each generated sample, write a ONE-line jsonl:
         {"task_id": ..., "completion": ...}
     - SCP it to the remote ARM/SVE server
     - Run a remote helper script (simdbench_remote_eval_one.py) that calls
       SimdBench's official evaluate_functional_correctness on that ONE sample
       and prints a JSON summary as the last stdout line
     - If fail, feed remote logs back into the LLM to repair, loop for N rounds.

Notes:
- Repair pipeline is currently SVE-centric (sv* intrinsics). For other intrinsics,
  name/shape repair is skipped unless you provide a suitable whitelist.

NEW ():
- Adds an optional external-API inference backend so you can benchmark:
    - OpenAI GPT-5* via OpenAI Responses API or Chat Completions API
    - DeepSeek R1 via DeepSeek's OpenAI-compatible Chat Completions API

The core generation/repair/remote-feedback logic is unchanged; only the "generate_text"
implementation is extended to support an API backend.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from copy import deepcopy

import hashlib
try:
    import fcntl  # type: ignore
except Exception:
    fcntl = None  # type: ignore
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import datetime as _dt

_PASS1_V22_SHARED_SCRIPT_DIR = Path("/home/user/selective_repo/latest_pairs_DeesSeekR1-32B/scripts")
if _PASS1_V22_SHARED_SCRIPT_DIR.exists():
    _shared_dir_text = str(_PASS1_V22_SHARED_SCRIPT_DIR)
    if _shared_dir_text not in sys.path:
        sys.path.insert(0, _shared_dir_text)
try:
    from compile_diagnostics_pass1_v22 import extract_compile_diagnostics as extract_shared_compile_diagnostics
except Exception:
    extract_shared_compile_diagnostics = None  # type: ignore
try:
    from semantic_plan_v1_lib import (
        parse_signature as parse_shared_signature,
    )
except Exception:
    parse_shared_signature = None  # type: ignore
try:
    from evaluate_nonbenchmark_performance_v22 import (
        build_perf_problem_row as build_nonbenchmark_perf_problem_row,
    )
except Exception:
    build_nonbenchmark_perf_problem_row = None  # type: ignore

# HF backend deps (only used when --llm_backend=hf)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore
    _HAVE_TRANSFORMERS = True
    _TRANSFORMERS_IMPORT_ERR = ""
except Exception as _e:
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    BitsAndBytesConfig = None  # type: ignore
    _HAVE_TRANSFORMERS = False
    _TRANSFORMERS_IMPORT_ERR = repr(_e)

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from peft import PeftModel  # type: ignore
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

# External API backend deps (requests optional; urllib fallback)
import random
import urllib.error
import urllib.request

try:
    import requests  # type: ignore
    _HAVE_REQUESTS = True
except Exception:
    requests = None
    _HAVE_REQUESTS = False


# -----------------------------------------------------------------------------
# Semantic gate global config (set in main)
# -----------------------------------------------------------------------------
SEMANTIC_REDUCTION_GATE: str = "soft"  # off | soft | hard

# -----------------------------------------------------------------------------
# Completion mode
# -----------------------------------------------------------------------------
# SimdBench can be used in two different compilation/evaluation setups:
#
# 1) "snippet"  : the model outputs only the *function body snippet* that will be
#                 appended to a prompt prefix containing headers + the function
#                 signature with an opening brace.
#
# 2) "full"     : the model outputs a *standalone translation unit* (i.e. the
#                 full source file) that will be compiled as-is.
#
# This script implements both modes so the generation/repair logic stays the
# same while the I/O and structural checks adapt.

COMPLETION_MODE_SNIPPET = "snippet"
COMPLETION_MODE_FULL = "full"

# Global mode (overridden by CLI flag in main).
COMPLETION_MODE: str = COMPLETION_MODE_FULL

def render_cpp_case(prompt_prefix: str, completion_text: str) -> str:
    """Render the full C/C++ source file for evaluation.

    In COMPLETION_MODE_FULL, `completion_text` is expected to already contain a full,
    standalone translation unit; we return it as-is.

    In COMPLETION_MODE_SNIPPET, `completion_text` is assumed to be a snippet that should be
    appended to `prompt_prefix` (which contains the surrounding harness / includes / stub).
    """
    if COMPLETION_MODE == COMPLETION_MODE_FULL:
        return completion_text
    return prompt_prefix + completion_text


# For COMPLETION_MODE_FULL we want to guarantee required #include lines are
# present (because the completion is compiled standalone). This list is
# populated per-task in main(), typically by extracting #include lines from the
# task prompt prefix.
CURRENT_REQUIRED_INCLUDES: List[str] = []
# For COMPLETION_MODE_FULL: keep the exact target function declaration extracted from the prompt prefix.
CURRENT_TARGET_FUNC_DECL: str = ""


# =============================================================================
# External API backend (OpenAI / DeepSeek)
# =============================================================================

def _truncate_middle(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    if s is None:
        return ""
    s = str(s)
    if len(s) <= max_chars:
        return s
    k1 = max_chars // 2
    k2 = max_chars - k1
    return s[:k1] + "\n...[TRUNCATED]...\n" + s[-k2:]

def _sleep_backoff(attempt: int, base: float, max_sleep: float) -> None:
    # Exponential backoff with jitter
    if attempt <= 0:
        t = base
    else:
        t = base * (2.0 ** (attempt - 1))
    t = min(max_sleep, max(0.0, t))
    t = t * (0.8 + 0.4 * random.random())
    if t > 0:
        time.sleep(t)

def _coerce_json_obj(s: str, *, what: str) -> Dict:
    if not s:
        return {}
    try:
        obj = json.loads(s)
    except Exception as e:
        raise ValueError(f"{what} must be valid JSON object string: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"{what} must be a JSON object (dict), got {type(obj)}")
    return obj

def _extract_openai_responses_text(resp: Dict) -> str:
    # Some SDKs expose response.output_text; JSON may not, but handle if present.
    ot = resp.get("output_text", None)
    if isinstance(ot, str) and ot.strip():
        return ot.strip()

    out = resp.get("output", None)
    texts: List[str] = []
    if isinstance(out, list):
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content", None)
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    ctype = c.get("type", "")
                    if ctype in ("output_text", "text") and isinstance(c.get("text"), str):
                        texts.append(c["text"])
    return "".join(texts).strip()

def _extract_chat_completions_text(resp: Dict) -> str:
    choices = resp.get("choices", None)
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            msg = c0.get("message", None)
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content.strip()
            # Some OpenAI-compatible servers may return "text" instead
            if isinstance(c0.get("text"), str):
                return str(c0.get("text")).strip()
    return ""

class ApiBackend:
    """
    Minimal API backend that supports:
      - OpenAI Responses API: POST {base_url}/responses
      - OpenAI-compatible Chat Completions: POST {base_url}/chat/completions
        (DeepSeek uses this format)
    """
    _is_api_backend = True

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        endpoint: str,
        timeout_s: int = 60,
        max_retries: int = 8,
        retry_backoff_s: float = 1.6,
        retry_max_sleep_s: float = 20.0,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict] = None,
        prompt_max_chars: int = 0,
        print_requests: bool = False,
    ) -> None:
        self.provider = str(provider or "api")
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = str(api_key or "")
        self.model = str(model or "")
        self.endpoint = str(endpoint or "")
        self.timeout_s = int(timeout_s) if timeout_s and timeout_s > 0 else 60
        self.max_retries = int(max_retries) if max_retries and max_retries >= 0 else 0
        self.retry_backoff_s = float(retry_backoff_s) if retry_backoff_s and retry_backoff_s > 0 else 1.0
        self.retry_max_sleep_s = float(retry_max_sleep_s) if retry_max_sleep_s and retry_max_sleep_s > 0 else 20.0
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self.prompt_max_chars = int(prompt_max_chars) if prompt_max_chars and prompt_max_chars > 0 else 0
        self.print_requests = bool(print_requests)

        self._session = requests.Session() if _HAVE_REQUESTS else None

        if not self.base_url:
            raise ValueError("ApiBackend: base_url is empty")
        if not self.model:
            raise ValueError("ApiBackend: model is empty")
        if not self.endpoint:
            raise ValueError("ApiBackend: endpoint is empty")
        if not self.api_key:
            raise ValueError("ApiBackend: api_key is empty (set --api_key or env var)")

    def eval(self):
        # HF code calls model.eval(); keep compatibility
        return self

    def _temperature_must_be_one(self) -> bool:
        backend_text = " ".join(
            [
                str(self.provider or ""),
                str(self.base_url or ""),
                str(self.model or ""),
            ]
        ).lower()
        return "kimi" in backend_text or "moonshot" in backend_text

    def _normalize_top_p(self, top_p: Optional[float]) -> float:
        # Moonshot/Kimi rejects anything except top_p=0.95. Keep this at the
        # API boundary so every path (main, repair, pre-explain, serial) is safe.
        if self._temperature_must_be_one():
            return 0.95
        if top_p is None:
            return 1.0
        return float(top_p)

    def _apply_extra_body_and_backend_limits(self, payload: Dict, *, top_p: Optional[float]) -> None:
        payload.update(self.extra_body)
        if self._temperature_must_be_one():
            # Moonshot/Kimi has a narrow sampling schema: temperature must be 1,
            # top_p must be 0.95, and top_k-style fields are rejected.
            for key in ("top_k", "topK", "topk"):
                payload.pop(key, None)
            if "temperature" in payload:
                payload["temperature"] = 1.0
            payload["top_p"] = self._normalize_top_p(top_p)

    def _headers(self) -> Dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        for k, v in self.extra_headers.items():
            if k and v is not None:
                h[str(k)] = str(v)
        return h

    def _post_json(self, url: str, payload: Dict) -> Dict:
        headers = self._headers()

        last_err = None
        for attempt in range(self.max_retries + 1):
            if self.print_requests:
                print("=" * 120)
                print(f"[API_REQUEST] provider={self.provider} endpoint={self.endpoint} attempt={attempt}")
                print("[URL]", url)
                print("[HEADERS_KEYS]", sorted(list(headers.keys())))
                # avoid printing api_key
                safe_payload = dict(payload)
                print("[PAYLOAD_KEYS]", sorted(list(safe_payload.keys())))
                print("=" * 120)

            try:
                if _HAVE_REQUESTS and self._session is not None:
                    resp = self._session.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                    status = int(getattr(resp, "status_code", 0) or 0)
                    text = getattr(resp, "text", "") or ""

                    if status >= 200 and status < 300:
                        try:
                            return resp.json()
                        except Exception:
                            # Some servers may return non-json on success (rare)
                            return json.loads(text)

                    # Retry on typical transient statuses
                    if status in (408, 409, 425, 429, 500, 502, 503, 504):
                        ra = resp.headers.get("Retry-After", "") if hasattr(resp, "headers") else ""
                        try:
                            ra_s = float(ra)
                        except Exception:
                            ra_s = 0.0
                        if ra_s > 0:
                            time.sleep(min(self.retry_max_sleep_s, ra_s))
                        else:
                            _sleep_backoff(attempt + 1, self.retry_backoff_s, self.retry_max_sleep_s)
                        last_err = RuntimeError(f"HTTP {status}: {text[:500]}")
                        continue

                    raise RuntimeError(f"HTTP {status}: {text[:2000]}")

                # urllib fallback
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as f:
                    body = f.read().decode("utf-8", errors="replace")
                    try:
                        return json.loads(body)
                    except Exception:
                        # if not json, raise
                        raise RuntimeError(f"Non-JSON response: {body[:2000]}")

            except (urllib.error.HTTPError,) as e:
                status = int(getattr(e, "code", 0) or 0)
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    body = ""
                if status in (408, 409, 425, 429, 500, 502, 503, 504):
                    _sleep_backoff(attempt + 1, self.retry_backoff_s, self.retry_max_sleep_s)
                    last_err = RuntimeError(f"HTTP {status}: {body[:500]}")
                    continue
                raise RuntimeError(f"HTTP {status}: {body[:2000]}") from e

            except (urllib.error.URLError, TimeoutError, OSError) as e:
                _sleep_backoff(attempt + 1, self.retry_backoff_s, self.retry_max_sleep_s)
                last_err = e
                continue

            except Exception as e:
                last_err = e
                break

        raise RuntimeError(f"API request failed after retries: {repr(last_err)}") from last_err

    def generate_text(
        self,
        *,
        user_text: str,
        system_text: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> str:
        # Optionally truncate prompts to avoid API context errors
        if self.prompt_max_chars and self.prompt_max_chars > 0:
            user_text = _truncate_middle(user_text, self.prompt_max_chars)
            system_text = _truncate_middle(system_text, self.prompt_max_chars)

        if self.endpoint == "responses":
            url = self.base_url + "/responses"
            payload: Dict = {
                "model": self.model,
                "input": user_text,
                "max_output_tokens": int(max_new_tokens),
            }
            if system_text:
                payload["instructions"] = system_text
            omit_sampling = self.provider == "openai" and "codex" in self.model.lower()
            if not omit_sampling:
                if do_sample:
                    payload["temperature"] = float(temperature)
                    payload["top_p"] = self._normalize_top_p(top_p)
                else:
                    payload["temperature"] = 0.0
                    payload["top_p"] = self._normalize_top_p(None)
                if self._temperature_must_be_one():
                    payload["temperature"] = 1.0

            # repetition_penalty not supported in Responses; ignore (kept for interface parity)
            self._apply_extra_body_and_backend_limits(payload, top_p=top_p if do_sample else None)

            js = self._post_json(url, payload)
            return _extract_openai_responses_text(js)

        if self.endpoint in ("chat_completions", "chat"):
            url = self.base_url + "/chat/completions"
            messages: List[Dict[str, str]] = []
            if system_text:
                messages.append({"role": "system", "content": system_text})
            messages.append({"role": "user", "content": user_text})

            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "max_tokens": int(max_new_tokens),
            }
            if do_sample:
                payload["temperature"] = float(temperature)
                payload["top_p"] = self._normalize_top_p(top_p)
            else:
                payload["temperature"] = 0.0
                payload["top_p"] = self._normalize_top_p(None)
            if self._temperature_must_be_one():
                payload["temperature"] = 1.0

            self._apply_extra_body_and_backend_limits(payload, top_p=top_p if do_sample else None)

            js = self._post_json(url, payload)
            return _extract_chat_completions_text(js)

        raise ValueError(f"Unknown API endpoint mode: {self.endpoint}")


def _repair_temperature_for_backend(model: Any, temperature: float) -> float:
    backend_text = " ".join(
        str(getattr(model, attr, "") or "")
        for attr in ("provider", "base_url", "model", "model_id", "name")
    ).lower()
    if "kimi" in backend_text or "moonshot" in backend_text:
        return 1.0
    return min(float(temperature), 0.25)


# =============================================================================
# SimdBench prompt helpers (baseline-compatible)
# =============================================================================

def sys_prompt(intrinsic: str, task_text: str) -> str:
    intrin_desc = {
        "SVE": "SVE (Arm ACLE for Scalable Vector Extension)",
        "Neon": "Neon (Advanced SIMD)",
        "AVX": "AVX/AVX2",
        "SSE": "SSE/SSE2",
        "RVV": "RISC-V Vector Extension",
    }
    include_hint = {
        "SVE": "#include <arm_sve.h>",
        "Neon": "#include <arm_neon.h>",
        "AVX": "#include <immintrin.h>",
        "SSE": "#include <immintrin.h>",
        "RVV": "#include <riscv_vector.h>",
    }

    mode_note = ""
    if COMPLETION_MODE == COMPLETION_MODE_FULL:
        mode_note = (
            "- The output will be compiled as-is (no prompt prefix is prepended).\n"
            "- Output a full standalone C/C++ translation unit: required #includes and the full target function definition.\n"
            "- Do NOT output any prompt markers or metadata; output ONLY C/C++ code.\n"
        )
    else:
        mode_note = (
            "- The output will be appended to a provided prompt prefix that already contains headers and the function signature.\n"
            "- Output ONLY the function body snippet (do NOT repeat the prompt prefix, do NOT add #includes, do NOT re-declare the function).\n"
            "- Output ONLY code. No markdown. No explanations.\n"
        )

    # SVE-specific guardrails: reduce hangs + predicate/count width errors.
    sve_guard = ""
    if intrinsic.strip().upper() == "SVE":
        sve_guard = (
            "SVE guardrails (must follow):\n"
            "- If you iterate over an array/string with svwhilelt_bX, the loop index variable MUST advance every iteration by the vector length for that element width:\n"
            "  * b8/u8/s8 -> i += svcntb();  b16/u16/s16 -> i += svcnth();  b32/u32/s32/f32 -> i += svcntw();  b64/u64/s64/f64 -> i += svcntd().\n"
            "- Do NOT use the result counter variable as the loop index. Keep separate variables, e.g. size_t i (index) and int count (answer).\n"
            "- For loads/stores, use the current predicate pg from svwhilelt_* for tail safety.\n"
            "- For comparisons producing predicates and boolean ops, prefer sv* _z/_m forms and svand_b_z / svorr_b_z / svbic_b_z etc. Avoid using C operators (&,|) on svbool_t.\n"
            "- When counting lanes, use svcntp_bX(pg, pred) with X matching the element width (b8/b16/b32/b64). Do NOT use svcntp_b64 on b8 lanes.\n"
            "- IMPORTANT: svcntp_bX counts predicate lanes; it does NOT compute popcount of integer vectors. For per-element bit counts of integer vectors, use svcnt_{u/s}* intrinsics (e.g., svcnt_u32_x(pg, v)).\n"
        )

    return f"""You are an expert SIMD programmer.
Target intrinsic: {intrinsic} - {intrin_desc.get(intrinsic, intrinsic)}.

Requirements:
- Write C/C++ code using {intrinsic} intrinsics.
- Include the correct header: {include_hint.get(intrinsic, "")}.
- Follow the task specification below and produce a compilable implementation.
- The exact target function signature shown in the task prompt / prompt prefix is fixed and must not be changed.
{mode_note}
{sve_guard}
Task:
{task_text}
"""


# =============================================================================
# Pre-explain / prompt expansion (NEW)
# =============================================================================

_SIMDBENCH_SVE_REQ_RE = re.compile(
    r"The requirement is to implement the function with SVE\s*"
    r"\(Arm C Language Extensions\s*\(ACLE\)\s*for the Arm Scalable Vector Extension\s*\(SVE\)\)\s*"
    r"intrinsics for parallelism\.\s*",
    flags=re.IGNORECASE,
)

PRE_EXPLAIN_REPLACEMENT_TEXT = (
    "You will be given a C/C++ function signature and comment that describe a SIMD/SVE task.\n"
    "Write an expanded spec that clarifies the exact semantics (what it must compute), assumptions, and edge cases.\n\n"
    "IMPORTANT constraints (must follow):\n"
    "- Do NOT output any C/C++ code.\n"
    "- Do NOT mention any intrinsic or function names that start with 'sv', 'sve', or 'srv'.\n"
    "- Do NOT invent pseudo syntax like '/pg/' or 'svadd/pg/'.\n"
    "- Keep it implementation-agnostic; focus on correctness, not performance.\n\n"
    "Output format MUST be exactly:\n"
    "[EXPANDED_SPEC]\n"
    "1) Inputs:\n"
    "- List each parameter with its type, meaning, shape (if array/matrix), and any constraints (alignment, aliasing).\n"
    "2) Output:\n"
    "- The return value or output buffer semantics.\n"
    "3) Semantics & edge cases:\n"
    "- Step-by-step computation.\n"
    "- Include corner cases (empty sizes, tails, overflow behavior, NaN handling if floats, etc.).\n"
    "- If memory layout / contiguity is not explicitly specified in the prompt, write 'unspecified' and do NOT guess.\n"
    "- If the prompt says the data is 'flattened into 1D', treat layout/contiguity as specified (contiguous 1D indexing).\n"
    "- Do NOT claim support for non-contiguous layouts unless the prompt explicitly provides strides/leading dimensions.\n"
    "[END_EXPANDED_SPEC]\n"
)

def sys_prompt_explain(intrinsic: str) -> str:
    intrin_desc = {
        "SVE": "SVE (Arm ACLE for Scalable Vector Extension)",
        "Neon": "Neon (Advanced SIMD)",
        "AVX": "AVX/AVX2",
        "SSE": "SSE/SSE2",
        "RVV": "RISC-V Vector Extension",
    }
    return f"""You are an expert SIMD programmer.
Target intrinsic: {intrinsic} - {intrin_desc.get(intrinsic, intrinsic)}.

You are in a PRE-EXPLAIN stage.
- Do NOT write any C/C++ code.
- Do NOT output pseudocode.
- Do NOT use markdown or code fences.
- Do NOT repeat, quote, or restate the prompt verbatim.
- Do NOT copy any existing code from the prompt (no '#include', no function signatures, no braces-only code blocks).
- Output ONLY the requested English explanation text in the required marker format.

The user prompt contains the original task description and possibly code context.
Focus on semantics and an SVE vectorization plan, not on reproducing code.
"""

def build_pre_explain_prompt(raw_prompt: str, replacement_text: str = PRE_EXPLAIN_REPLACEMENT_TEXT) -> Tuple[str, bool]:
    if _SIMDBENCH_SVE_REQ_RE.search(raw_prompt):
        return _SIMDBENCH_SVE_REQ_RE.sub(replacement_text, raw_prompt, count=1), True

    m = re.search(r"\*/", raw_prompt)
    if m:
        newp = raw_prompt[:m.start()] + "\n" + replacement_text + "\n" + raw_prompt[m.start():]
        return newp, False

    return raw_prompt + "\n\n" + replacement_text + "\n", False




# =============================================================================
# Pre-explain output validation / extraction (NEW)
# =============================================================================

_EXPANDED_SPEC_BEGIN = "[EXPANDED_SPEC]"
_EXPANDED_SPEC_END = "[END_EXPANDED_SPEC]"

_FUNC_DEF_LIKE_RE = re.compile(
    r"(?m)^\s*(?:static\s+|inline\s+|constexpr\s+)?"
    r"(?:extern\s+\"C\"\s+)?"
    r"(?:[\w:<>]+\s+)+"
    r"[A-Za-z_]\w*\s*\([^;]*\)\s*\{"
)

_RE_DEFINES_FUNCTION = _FUNC_DEF_LIKE_RE

def extract_marked_block(text: str, start_marker: str, end_marker: str) -> str:
    if not text:
        return ""
    s = str(text)
    i = s.find(start_marker)
    j = s.find(end_marker)
    if i < 0 or j < 0 or j <= i:
        return ""
    j2 = j + len(end_marker)
    return s[i:j2].strip()

def extract_comment_blocks_for_explain(raw_prompt: str) -> str:
    """
    Best-effort: keep only C/C++ comment blocks from the original prompt, so the model
    focuses on semantics instead of echoing code skeletons.
    """
    if not raw_prompt:
        return ""
    s = str(raw_prompt)

    blocks: List[str] = []
    for m in re.finditer(r"/\*[\s\S]*?\*/", s):
        blk = m.group(0).strip()
        if blk:
            blocks.append(blk)

    if blocks:
        return "\n\n".join(blocks).strip()

    # fallback: keep // comment. lines if present
    lines = []
    for ln in s.splitlines():
        if ln.strip().startswith("//"):
            lines.append(ln)
    if lines:
        return "\n".join(lines).strip()

    return s.strip()

def validate_expanded_spec(text: str) -> Tuple[bool, str, str]:
    """
    Returns (ok, cleaned_block, reason).
    cleaned_block is the extracted [EXPANDED_SPEC]..[END_EXPANDED_SPEC] region if possible.
    """
    if not text or not str(text).strip():
        return False, "", "empty"

    s = str(text).strip()
    blk = extract_marked_block(s, _EXPANDED_SPEC_BEGIN, _EXPANDED_SPEC_END)
    if not blk:
        # Sometimes the model omits end marker; treat as invalid.
        return False, "", "missing_markers"

    # Heuristics to reject code-echo / code output.
    low = blk.lower()
    if "#include" in low or re.search(r"(?m)^\s*#", blk):
        return False, "", "contains_preprocessor"
    if "int main" in low:
        return False, "", "contains_main"
    if _FUNC_DEF_LIKE_RE.search(blk):
        return False, "", "contains_function_def"
    if re.search(r"(?m)^\s*typedef\b", blk) or re.search(r"(?m)^\s*struct\b", blk):
        return False, "", "contains_type_def"
    # If it looks like actual code braces block with many semicolons, reject.
    if "{" in blk and "}" in blk and blk.count(";") >= 3:
        return False, "", "looks_like_code"
    if re.search(r"\b(?:sve|srv)_[A-Za-z0-9_]+\b", blk):
        return False, "", "contains_pseudo_intrinsic"

    return True, blk.strip(), "ok"

def _changed_line_count(a: str, b: str) -> int:
    """Approximate number of changed lines between two code strings.

    Used as a *semantic-stability* preference: when multiple candidates eliminate the same invalid intrinsics,
    prefer the one that changes fewer lines (less algorithmic drift).
    """
    if a is None:
        a = ""
    if b is None:
        b = ""
    a_lines = str(a).splitlines()
    b_lines = str(b).splitlines()
    sm = difflib.SequenceMatcher(a=a_lines, b=b_lines)
    changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            changed += max(i2 - i1, j2 - j1)
    return changed






def _is_trivial_line_for_locality(ln: str) -> bool:
    """Return True if a line change is very unlikely to affect semantics.

    We allow these changes outside the local edit window:
    - blank lines / whitespace-only lines
    - pure comment lines (// ... , /* ... */, * ...)
    - preprocessor lines (#include/#define/...)
    """
    if ln is None:
        return True
    s = ln.strip()
    if not s:
        return True
    if s.startswith("#"):
        return True
    if s.startswith("//"):
        return True
    if s.startswith("/*") or s.startswith("*/") or s.startswith("*"):
        return True
    return False


def _find_call_line_indices(code: str, names: List[str]) -> List[int]:
    if not code or not names:
        return []
    idxs: Set[int] = set()
    lines = code.splitlines()
    for i, ln in enumerate(lines):
        for nm in names:
            if re.search(rf"\b{re.escape(nm)}\s*\(", ln):
                idxs.add(i)
    return sorted(idxs)


def _count_nonlocal_edit_segments(
    code_before: str,
    code_after: str,
    anchor_lines: List[int],
    window: int,
) -> int:
    """Count non-trivial diff segments that fall outside a ±window around anchor_lines (in code_before).

    This is used as a semantic-stability gate for *name-fix / invalid-intrinsic cleanup* stages:
    the model should not rewrite unrelated parts of the program when only intrinsic names/signatures
    are invalid.

    We do NOT use this to block genuine algorithmic fixes later (remote feedback), so callers should
    apply it selectively.
    """
    if not code_before or not code_after:
        return 0
    if not anchor_lines:
        return 0

    a = code_before.splitlines()
    b = code_after.splitlines()

    allowed: Set[int] = set()
    for l in anchor_lines:
        lo = max(0, int(l) - int(window))
        hi = min(len(a), int(l) + int(window) + 1)
        for i in range(lo, hi):
            allowed.add(i)

    sm = difflib.SequenceMatcher(a=a, b=b)
    violations = 0

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        # Insert: changes have no 'a' span; attribute to insertion point i1.
        if tag == "insert":
            inserted = b[j1:j2]
            if all(_is_trivial_line_for_locality(x) for x in inserted):
                continue
            if i1 not in allowed:
                violations += 1
            continue

        # Replace/Delete: check changed lines from original 'a'
        changed_idxs = list(range(i1, i2))
        nontrivial = [i for i in changed_idxs if (0 <= i < len(a)) and (not _is_trivial_line_for_locality(a[i]))]
        if not nontrivial:
            # only touched whitespace/comments/preprocessor
            continue

        if any(i not in allowed for i in nontrivial):
            violations += 1

    return violations


def sanitize_expanded_spec_for_codegen(text: str, intrinsic: str) -> str:
    """Sanitize the pre-explain expanded spec before feeding it into prompts.

    Why:
    - Pre-explain often hallucinates pseudo-intrinsic names like 'sve_ld1', 'svecmpeq_f64', etc.
    - It may also contain an implementation plan that is speculative or outright wrong (e.g. misuse of svptest_any).
    - Including such content in codegen/repair prompts measurably increases compile errors and "cargo-cult" bad calls.

    Policy:
    - For non-SVE, keep text unchanged.
    - For SVE:
        * Drop the whole "SVE implementation plan" section (keep only I/O + scalar semantics + assumptions).
        * Remove lines that contain pseudo-intrinsic tokens starting with "sve" / "srv".
        * Remove lines that look like intrinsic lists / bullet points of sv* calls.
        * Remove lines that contain pseudo syntax like "svxxx/pg/".
    This is intentionally conservative: we prefer *less* guidance over misleading guidance.
    """
    intr = intrinsic.strip().upper()
    if intr != "SVE":
        return text

    if not text:
        return text

    # Keep only sections 1-3 if we see a "4) SVE implementation plan" header.
    # (We still keep the [END_EXPANDED_SPEC...] marker so the downstream prompt format is intact.)
    lines = text.splitlines()
    out_lines: List[str] = []
    in_plan = False

    plan_hdr = re.compile(r"^\s*(?:4\)\s*)?SVE\s+implementation\s+plan\s*:\s*$", re.I)

    for ln in lines:
        if plan_hdr.search(ln):
            in_plan = True
            continue
        if in_plan:
            # Skip everything until the end tag.
            if "[END_EXPANDED_SPEC" in ln:
                in_plan = False
                out_lines.append(ln)
            continue

        # Drop obviously hallucinatory pseudo-intrinsic tokens like sve_ld1 / svecmpeq_f64 / svewhile...
        # Keep genuine 'SVE' mentions.
        if re.search(r"\bsve[_a-zA-Z0-9]+\b", ln) and (not re.search(r"\bSVE\b", ln)):
            continue
        if re.search(r"\bsrv[_a-zA-Z0-9]+\b", ln):
            continue

        # Drop pseudo syntax tokens in the plan that later cause forbidden patterns in code:
        if re.search(r"\bsv[a-zA-Z0-9_]+\s*/", ln) or ("/pg/" in ln):
            continue

        # Drop "Key SVE intrinsics" lists and bullet lines that are just sv* names (often incomplete/wrong).
        if "Key SVE intrinsics" in ln:
            continue
        if re.match(r"\s*[-*]\s*sv[a-zA-Z0-9_]+\b", ln):
            continue

        out_lines.append(ln)

    # Extra semantic sanitization:
    # - Pre-explain sometimes contradicts itself (e.g., it says "flattened into 1D" but later claims
    #   contiguity is "unspecified", or it invents "non-contiguous" requirements despite the prompt).
    # Keep this strictly conservative: only drop lines that are clearly speculative/misleading.
    cleaned: List[str] = []
    has_flattened = any(re.search(r"\bflatten", ln, re.IGNORECASE) for ln in out_lines)
    for ln in out_lines:
        # Speculative "non-contiguous"/"any memory layout" claims tend to mislead codegen.
        if re.search(r"\bnon-?contiguous\b", ln, re.IGNORECASE):
            continue
        if re.search(r"\bany\s+memory\s+layout\b", ln, re.IGNORECASE):
            continue
        # If the spec itself already states "flattened", then "contiguity ... unspecified" is contradictory.
        if has_flattened and re.search(r"memory\s+layout/contiguity.*unspecified", ln, re.IGNORECASE):
            continue
        cleaned.append(ln)

    return "\n".join(cleaned).strip()

def clean_code_block(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:c|cc|cpp|c\+\+|h|hpp)?\s*([\s\S]*?)```", t, flags=re.IGNORECASE)
    if m:
        t = m.group(1).strip()
    t = re.sub(r"```(?:cpp|c\+\+|c)?", "", t)
    t = t.replace("```", "")
    t = re.sub(r"\[/?cpp\]", "", t)
    return t.strip()


def load_problems(problem_file: str) -> List[dict]:
    problems: List[dict] = []
    with open(problem_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))
    for p in problems:
        if "task_id" not in p:
            raise ValueError("problem missing task_id")
        if "prompt" not in p:
            raise ValueError(f"problem {p.get('task_id')} missing prompt")
        if "task" not in p:
            p["task"] = ""
    return problems


def load_serial_ast_bootstrap_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    """Load optional AST-derived bootstrap records keyed by task_id.

    The records are used only for the semantic bootstrap text shown to the
    repair model.  Serial correctness and serial-vs-SIMD comparison still use
    the validated serial reference path.
    """
    out: Dict[str, Dict[str, Any]] = {}
    p = str(path or "").strip()
    if not p:
        return out
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            task_id = str(rec.get("task_id") or rec.get("id") or "").strip()
            if task_id:
                out[task_id] = rec
    return out


def lookup_serial_ast_bootstrap_record(
    serial_ast_bootstrap_map: Dict[str, Dict[str, Any]],
    task_id: str,
) -> Dict[str, Any]:
    """Find an AST bootstrap record for a benchmark task id.

    Some problem files use target-suffixed ids such as ``*_SVE`` while the
    AST bootstrap cache is keyed by the target-independent base id.  Keep the
    lookup deterministic and conservative: try exact first, then only known
    target suffix removals.
    """
    tid = str(task_id or "").strip()
    if not tid or not serial_ast_bootstrap_map:
        return {}

    candidates: List[str] = [tid]
    for suffix in ("_SVE", "_Neon", "_NEON", "_AVX", "_AVX2", "_SSE", "_RVV"):
        if tid.endswith(suffix):
            candidates.append(tid[: -len(suffix)])

    seen = set()
    for key in candidates:
        if not key or key in seen:
            continue
        seen.add(key)
        rec = serial_ast_bootstrap_map.get(key)
        if isinstance(rec, dict):
            out = dict(rec)
            out["_lookup_key"] = key
            out["_requested_task_id"] = tid
            return out
    return {}


def normalize_prompt_prefix(user_prompt: str) -> str:
    p = re.sub(r"\{\s*\}\s*$", "{\n", user_prompt)
    return p


# =============================================================================
# torchrun / distributed + sharding helpers
# =============================================================================

def get_dist_info() -> Tuple[int, int, int, bool]:
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws > 1:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return rank, ws, local_rank, True
    return 0, 1, 0, False


def resolve_remote_flock_path_for_rank(path: str, rank: int, local_rank: int, world_size: int) -> str:
    if not path:
        return ""
    s = str(path)
    if not s:
        return ""
    if "{" in s and "}" in s:
        try:
            return s.format(rank=rank, local_rank=local_rank, world_size=world_size)
        except Exception:
            return s
    if s.endswith(".lock"):
        return f"{s[:-5]}.rank{rank}.lock"
    return f"{s}.rank{rank}"


def shard_contiguous(items: List[dict], num_shards: int, shard_id: int) -> List[dict]:
    if num_shards <= 1:
        return items
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard_id={shard_id} for num_shards={num_shards}")
    n = len(items)
    per = (n + num_shards - 1) // num_shards
    start = shard_id * per
    end = min(start + per, n)
    return items[start:end]


def shard_output_path(base_out: Path, shard_id: int, num_shards: int) -> Path:
    if num_shards <= 1:
        return base_out
    return base_out.with_name(f"{base_out.stem}.shard{shard_id}of{num_shards}{base_out.suffix}")

def _done_marker_path(base_out: Path, rank: int, num_shards: int) -> Path:
    # Keep the marker basename below common filesystem limits even when run tags
    # are long; all ranks call this helper, so wait/cleanup stay consistent.
    stem = base_out.stem
    digest = hashlib.sha1(base_out.name.encode("utf-8", errors="replace")).hexdigest()[:12]
    short_stem = stem if len(stem) <= 120 else f"{stem[:120]}.{digest}"
    return base_out.with_name(f"{short_stem}.rank{rank}of{num_shards}.done")

def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp_{os.getpid()}")
    tmp.write_text(text, encoding="utf-8", errors="replace")
    tmp.replace(path)

def write_done_marker(base_out: Path, rank: int, num_shards: int, payload: Dict) -> Path:
    p = _done_marker_path(base_out, rank, num_shards)
    _atomic_write_text(p, json.dumps(payload, ensure_ascii=False) + "\n")
    return p

def wait_for_done_markers(
    base_out: Path,
    num_shards: int,
    *,
    timeout_s: int,
    poll_s: float = 1.0,
    print_every_s: float = 30.0,
) -> Tuple[bool, List[int]]:
    """Return (ok_all_done, missing_ranks). timeout_s<=0 means wait forever."""
    t0 = time.time()
    last_print = 0.0
    while True:
        missing = []
        for r in range(num_shards):
            if not _done_marker_path(base_out, r, num_shards).exists():
                missing.append(r)

        if not missing:
            return True, []

        if timeout_s > 0 and (time.time() - t0) > timeout_s:
            return False, missing

        if print_every_s > 0 and (time.time() - last_print) > print_every_s:
            print(f"[MERGE_WAIT] missing done markers ranks={missing} (waiting...)")
            last_print = time.time()

        time.sleep(max(0.05, float(poll_s)))


def merge_shard_outputs(base_out: Path, num_shards: int) -> None:
    # Atomic merge: write to temp then replace base_out
    tmp_out = base_out.with_name(base_out.name + f".merge_tmp_{os.getpid()}")
    with tmp_out.open("w", encoding="utf-8") as wf:
        for sid in range(num_shards):
            part = shard_output_path(base_out, sid, num_shards)
            if not part.exists():
                continue
            with part.open("r", encoding="utf-8") as rf:
                for line in rf:
                    wf.write(line)
    tmp_out.replace(base_out)


# =============================================================================
# HF prompt helpers
# =============================================================================

def has_chat_template(tokenizer) -> bool:
    tmpl = getattr(tokenizer, "chat_template", None)
    return bool(tmpl and isinstance(tmpl, str) and tmpl.strip())


def apply_chat_prompt(tokenizer, user_text: str, system_text: str = "") -> Tuple[torch.Tensor, int]:
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})

    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        input_ids = torch.tensor([ids], dtype=torch.long)
        return input_ids, len(ids)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt")
        return enc["input_ids"], enc["input_ids"].shape[1]


# =============================================================================
# Dialogue trace capture
# =============================================================================

_DLG_CAPTURE: Optional[List[Dict]] = None
_DLG_STAGE: str = ""
_DLG_MAX_CHARS: int = 0

def dlg_set_capture(buf: Optional[List[Dict]], max_chars: int = 0) -> None:
    global _DLG_CAPTURE, _DLG_MAX_CHARS
    _DLG_CAPTURE = buf
    _DLG_MAX_CHARS = int(max_chars) if max_chars and max_chars > 0 else 0

def dlg_set_stage(stage: str) -> None:
    global _DLG_STAGE
    _DLG_STAGE = str(stage or "")

def _dlg_trunc(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    if _DLG_MAX_CHARS <= 0:
        return s
    if len(s) <= _DLG_MAX_CHARS:
        return s
    k = _DLG_MAX_CHARS
    k1 = k // 2
    k2 = k - k1
    return s[:k1] + "\n...[TRUNCATED]...\n" + s[-k2:]

def dlg_record(system_text: str, user_text: str, assistant_text: str, gen_args: Dict) -> None:
    if _DLG_CAPTURE is None:
        return
    _DLG_CAPTURE.append({
        "call_idx": len(_DLG_CAPTURE),
        "stage": _DLG_STAGE,
        "system": _dlg_trunc(system_text or ""),
        "user": _dlg_trunc(user_text or ""),
        "assistant": _dlg_trunc(assistant_text or ""),
        "gen_args": gen_args,
        "ts": time.time(),
    })


@torch.no_grad()
def generate_text(
    model,
    tokenizer=None,
    *,
    user_text: str,
    system_text: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    backend: Optional[str] = None,
    api_model: str = "",
    is_t0: bool = False,
    **_ignored: Any,
) -> str:
    """
    Unified generation helper (HF + API).

    Notes:
      - HF backend: `model` is a transformers causal LM and `tokenizer` is a transformers tokenizer.
      - API backend: `model` is an ApiBackend instance; `tokenizer` can be None.
      - Extra keyword args are accepted and ignored for backward compatibility with older call sites.
    """
    # Decide backend:
    # - Prefer model's marker if present
    # - Otherwise accept explicit backend hints
    _backend_hint = (backend or "").strip().lower()
    use_api = bool(getattr(model, "_is_api_backend", False)) or (_backend_hint in {"api", "openai", "deepseek", "responses", "chat_completions"})

    if use_api:
        # Per-call override of API model name (optional)
        orig_model = getattr(model, "model", None)
        if api_model:
            try:
                model.model = api_model  # type: ignore[attr-defined]
            except Exception:
                pass

        out_text = model.generate_text(
            user_text=user_text,
            system_text=system_text,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        out_text = (out_text or "").strip()

        # Restore model name if we temporarily overrode it
        if api_model and (orig_model is not None):
            try:
                model.model = orig_model  # type: ignore[attr-defined]
            except Exception:
                pass

        dlg_record(
            system_text=system_text,
            user_text=user_text,
            assistant_text=out_text,
            gen_args={
                "backend": "api",
                "backend_hint": backend,
                "provider": getattr(model, "provider", ""),
                "endpoint": getattr(model, "endpoint", ""),
                "model": getattr(model, "model", "") if not api_model else api_model,
                "api_model_override": bool(api_model),
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "is_t0": bool(is_t0),
            },
        )
        return out_text

    # HF backend
    if not _HAVE_TRANSFORMERS:
        raise RuntimeError(
            "transformers is required for --llm_backend=hf but could not be imported. "
            f"Import error: {_TRANSFORMERS_IMPORT_ERR}"
        )

    if tokenizer is None:
        raise ValueError("tokenizer is required for HF generation (backend=hf).")

    model.eval()

    if has_chat_template(tokenizer):
        input_ids, prompt_len = apply_chat_prompt(tokenizer, user_text, system_text=system_text)
        dev = getattr(model, "device", None)
        if dev is None:
            dev = next(model.parameters()).device
        input_ids = input_ids.to(dev)
        attn = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        out_ids = gen[0, prompt_len:]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        dlg_record(
            system_text=system_text,
            user_text=user_text,
            assistant_text=out_text,
            gen_args={
                "backend": "hf",
                "backend_hint": backend,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "chat_template": True,
                "is_t0": bool(is_t0),
            },
        )
        return out_text

    if system_text:
        prompt = f"[INST] <<SYS>>\n{system_text}\n<</SYS>>\n\n{user_text} [/INST]"
    else:
        prompt = user_text

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    dev = getattr(model, "device", None)
    if dev is None:
        dev = next(model.parameters()).device
    input_ids = enc["input_ids"].to(dev)
    attn = enc["attention_mask"].to(dev)
    prompt_len = input_ids.shape[1]
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    out_ids = gen[0, prompt_len:]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
    dlg_record(
        system_text=system_text,
        user_text=user_text,
        assistant_text=out_text,
        gen_args={
            "backend": "hf",
            "backend_hint": backend,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "chat_template": False,
            "is_t0": bool(is_t0),
        },
    )
    return out_text



# =============================================================================
# SVE whitelist-based repair (name + call-shape)
# =============================================================================

CALL_RE = re.compile(r"\b(sv[a-zA-Z0-9_]+)\s*\(")

# Some models omit the "sv" prefix on SVE intrinsics (e.g. "shr_n_s32_z").
# We detect these call sites (conservatively) so the name-fix stage can repair them.
CALL_ANY_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

# Match (top-level) function declarations/definitions so we do not treat them as call sites.
FUNC_DECL_RE = re.compile(
    r"(?m)^\s*"
    r"(?:static\s+|inline\s+|constexpr\s+)?"
    r"(?:extern\s+\"C\"\s+)?"
    r"(?:[\w:<>*&]+\s+)+"
    r"([A-Za-z_]\w*)\s*\([^;{]*\)\s*(?:\{|;)"
)
# Backward-compat alias (older code paths refer to this name)
# Any top-level function declaration/definition (including non-sv helpers) so we can avoid treating
# calls to helper functions as pseudo intrinsics.
_TOPLEVEL_FUNC_DECL_RE = re.compile(
    r"(?m)^\s*"
    r"(?:static\s+|inline\s+|constexpr\s+)?"
    r"(?:extern\s+\"C\"\s+)?"
    r"(?:[\w:<>*&]+\s+)+"
    r"([A-Za-z_]\w*)\s*\([^;]*\)\s*"
    r"(?:__attribute__\s*\(\([^)]*\)\)\s*)*"
    r"(?:\{|;)"
)

# Heuristic token: most SVE intrinsics include a lane-type suffix somewhere in the name.
_SVE_LIKE_NAME_TOKEN_RE = re.compile(
    r"_(?:b8|b16|b32|b64|u8|u16|u32|u64|s8|s16|s32|s64|f16|f32|f64)(?:_|$)"
)

# Keywords/builtins that look like calls but are not real functions.
_C_CALL_KEYWORDS: Set[str] = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "alignof",
    "catch",
    "throw",
    "new",
    "delete",
}

DEF_RE = re.compile(
    r"(?m)^\s*"
    r"(?:static\s+|inline\s+|constexpr\s+)*"
    r"(?:extern\s+\"C\"\s+)?"
    r"(?:[\w:<>*&]+\s+)+"
    r"(sv[a-zA-Z0-9_]+)\s*\("
)

DECL_RE = re.compile(
    r"(?m)^\s*(?:const\s+)?([a-zA-Z_]\w*(?:\s*<[^;]+?>)?(?:\s*\*+)?)\s+([a-zA-Z_]\w*)\s*(?:=|;|,)"
)

STRICT_NAME_RULES_SNIPPET = """/*
[INTRINSICS_NAME_FIX_RULES - FUNCTIONAL]

You are repairing a *completion snippet* that will be appended to the provided prompt prefix.
Do NOT repeat the prompt prefix. Output ONLY the corrected completion snippet.

Goal:
- Produce CORRECT, compilable C/C++ Arm SVE (ACLE) completion code.
- Prioritize functional correctness. Fix invalid / wrong SVE intrinsic usage as needed.
- This stage focuses on intrinsic NAME-level issues first, but you MAY also make minimal call/type fixes
  if required to make the intrinsic usage valid and functionally correct.

What you MAY change (when necessary for correctness):
- Fix intrinsic names: typos, missing suffixes, wrong element type/width suffix, missing *_z, wrong predicate form.
- Replace an intrinsic with another VALID SVE ACLE intrinsic if the original name is wrong or cannot be made valid.
- Add a missing predicate argument (e.g., 'pg') when switching to *_z intrinsics.
- Insert minimal supporting code if needed for validity/correctness:
  svld1_*/svst1_*, svdup_*, svreinterpret_*, casts, temporary variables.

Hard constraints:
- The bare intrinsic 'svwhilelt' DOES NOT EXIST. Never output "svwhilelt(".
  You MUST use exactly one of: svwhilelt_b8 / svwhilelt_b16 / svwhilelt_b32 / svwhilelt_b64
- FINAL CHECK before output: Your output must contain ZERO occurrences of: "svwhilelt("

[END_INTRINSICS_NAME_FIX_RULES - FUNCTIONAL]
*/"""


STRICT_NAME_RULES_FULL = """/*
[INTRINSICS_NAME_FIX_RULES - FUNCTIONAL]

You are repairing a *standalone C/C++ source file* completion that will be compiled as-is.
Do NOT assume any prompt prefix will be prepended.

Goal:
- Produce CORRECT, compilable C/C++ SIMD intrinsic code.
- Prioritize functional correctness. Fix invalid / wrong intrinsic usage as needed.

What you MAY change (when necessary for correctness):
- Fix intrinsic names: typos, missing suffixes, wrong element type/width suffix, missing *_z, wrong predicate form.
- Replace an intrinsic with another VALID intrinsic if the original name is wrong or cannot be made valid.
- Add missing predicate arguments (e.g., 'pg') when switching to *_z intrinsics.
- Insert minimal supporting code if needed for validity/correctness.

Hard constraints:
- The bare intrinsic 'svwhilelt' DOES NOT EXIST. Never output "svwhilelt(".
  You MUST use exactly one of: svwhilelt_b8 / svwhilelt_b16 / svwhilelt_b32 / svwhilelt_b64
- FINAL CHECK before output: Your output must contain ZERO occurrences of: "svwhilelt("

Output format:
- Output ONLY compilable C/C++ code for a standalone translation unit.
- Do NOT output any of the prompt metadata markers (e.g., [SPEC], [END], ...).

[END_INTRINSICS_NAME_FIX_RULES - FUNCTIONAL]
*/"""

FUNCTIONAL_REPAIR_RULES_SNIPPET = """/*
[INTRINSICS_REPAIR_RULES - FUNCTIONAL]

You are repairing a *completion snippet* that will be appended to the provided prompt prefix.
Do NOT repeat the prompt prefix. Output ONLY the corrected completion snippet.

Goal:
- Produce CORRECT, compilable C/C++ Arm SVE (ACLE) completion code that matches the specification below.
- Fix any wrong intrinsic usage (wrong intrinsic choice, wrong argument shapes, wrong types, wrong predicates),
  as needed to make the code functionally correct.

What you MAY change (when necessary for correctness):
- Replace an intrinsic with another VALID SVE ACLE intrinsic.
- Change intrinsic call arguments: add/remove/reorder args, add missing predicate argument, fix types.
- Insert minimal supporting code: svld1_*/svst1_*, svdup_*, svreinterpret_*, casts, temporary variables.
- Adjust predicate generation / VLA loop stepping ONLY if required for correctness.

[END_INTRINSICS_REPAIR_RULES - FUNCTIONAL]
*/"""


FUNCTIONAL_REPAIR_RULES_FULL = """/*
[INTRINSICS_REPAIR_RULES - FUNCTIONAL]

You are repairing a *standalone C/C++ source file* completion that will be compiled as-is.
Do NOT assume any prompt prefix will be prepended.

Goal:
- Produce CORRECT, compilable C/C++ Arm SVE (ACLE) code that matches the specification below.
- Fix any wrong intrinsic usage (wrong intrinsic choice, wrong argument shapes, wrong types, wrong predicates),
  as needed to make the code functionally correct.

What you MAY change (when necessary for correctness):
- Replace an intrinsic with another VALID SVE ACLE intrinsic.
- Change intrinsic call arguments: add/remove/reorder args, add missing predicate argument, fix types.
- Insert minimal supporting code: svld1_*/svst1_*, svdup_*, svreinterpret_*, casts, temporary variables.
- Adjust predicate generation / VLA loop stepping ONLY if required for correctness.

Hard constraints:
- Do NOT output any prompt metadata markers (e.g., [SPEC], [END], ...).

[END_INTRINSICS_REPAIR_RULES - FUNCTIONAL]
*/"""

def strip_comments(text: str) -> str:
    text = re.sub(r"/\*[\s\S]*?\*/", "", text)
    text = re.sub(r"//.*", "", text)
    return text

def _looks_like_missing_sv_intrinsic_name(name: str, whitelist_set: Optional[Set[str]]) -> bool:
    """Heuristic: decide whether `name` looks like an SVE intrinsic missing the 'sv' prefix.

    We keep this conservative to avoid treating user helper functions as intrinsics.

    Examples that should return True:
      - shr_n_s32_z  (should be svasr_n_s32_z or svlsr_n_s32_z depending on signedness)
      - add_u32_m
      - whilelt_b64

    We prefer whitelist evidence ("sv"+name exists) when available; otherwise we require
    a strong SVE token (e.g., _s32_, _u64_, _b32_) in the name.
    """
    if not name:
        return False
    if name.startswith("sv"):
        return False
    if name in _C_CALL_KEYWORDS:
        return False
    if name.endswith("_t"):
        # Likely a typedef/constructor-like token (e.g. int32_t(...))
        return False
    # If a whitelist is available and the prefixed version exists, it's almost certainly intended.
    if whitelist_set is not None:
        # Missing 'sv' prefix (exact)
        if ("sv" + name) in whitelist_set:
            return True

        # Common model typo: accidental leading 'v' (e.g., vdiv_u64 -> svdiv_u64_x/_z/...)
        if name.startswith("v") and (not name.startswith("sv")) and len(name) > 1:
            cand = "sv" + name[1:]
            if cand in whitelist_set:
                return True
            for suf in ("_x", "_z", "_m"):
                if (cand + suf) in whitelist_set:
                    return True

    # Otherwise, only treat it as a missing-"sv" intrinsic if it looks very much like an
    # ACLE intrinsic variant (typed token) **and** uses common predication/merge suffixes.
    # This keeps the heuristic conservative and reduces accidental "repair" of user helper
    # functions that happen to include type-like substrings.
    if _SVE_LIKE_NAME_TOKEN_RE.search(name):
        if name.endswith(("_x", "_z", "_m")) or ("_z_" in name) or ("_x_" in name) or ("_m_" in name):
            return True

    return False


def extract_calls(code: str, whitelist_set: Optional[Set[str]] = None, *, context_text: str = "") -> Set[str]:
    """Extract function-like call names we want to validate/repair.

    - Always includes sv* calls (excluding sv* types like svint32_t).
    - Optionally includes non-sv names that look like SVE intrinsics missing the 'sv' prefix.
      This is controlled by a conservative heuristic and/or whitelist evidence.

    The returned set excludes any names that are function declarations/definitions
    (both in `code` itself and, in snippet mode, also in `context_text` which usually
    contains the prompt prefix).
    """
    if not code:
        return set()

    code2 = strip_preprocessor_lines(strip_comments(code))

    # Function declarations/definitions to exclude from call-sites.
    defs_any: Set[str] = set(_TOPLEVEL_FUNC_DECL_RE.findall(code2))
    defs_any |= set(DEF_RE.findall(code2))

    # In snippet mode, the prompt prefix may contain the target function signature and
    # helper prototypes; treat those as defs too so we don't misclassify them as calls.
    if context_text and COMPLETION_MODE == COMPLETION_MODE_SNIPPET:
        ctx = strip_preprocessor_lines(strip_comments(context_text))
        defs_any |= set(_TOPLEVEL_FUNC_DECL_RE.findall(ctx))
        defs_any |= set(DEF_RE.findall(ctx))

    calls: Set[str] = set()

    # 1) Explicit sv* calls
    for n in CALL_RE.findall(code2):
        if n in defs_any:
            continue
        # Exclude sv* type tokens like svint32_t / svbool_t
        if n.endswith("_t"):
            continue
        calls.add(n)

    # 1b) Pseudo-intrinsic calls some models hallucinate: sve*/srv*.
    # These are NOT valid SVE ACLE intrinsics (valid ones start with 'sv'). Treat them as invalid
    # so name-fix / hard-rewrite can eliminate them deterministically.
    for n in re.findall(r"\b((?:sve|srv)[a-zA-Z0-9_]+)\s*\(", code2):
        if n in defs_any:
            continue
        if n.endswith("_t"):
            continue
        calls.add(n)


    # 2) Missing 'sv' prefix calls (conservative)
    for m in CALL_ANY_RE.finditer(code2):
        name = m.group(1)
        if name in defs_any:
            continue
        if name.startswith("sv"):
            continue
        if name in _C_CALL_KEYWORDS:
            continue
        if name.endswith("_t"):
            continue
        # Ignore obvious macros/constants
        if name.isupper():
            continue
        if _looks_like_missing_sv_intrinsic_name(name, whitelist_set):
            calls.add(name)

    return calls

def infer_lane_width(code: str) -> Optional[int]:
    if re.search(r"\bsvcntd\s*\(", code) or re.search(r"\bsvfloat64_t\b", code) or "_f64" in code:
        return 64
    if re.search(r"\bsvcntw\s*\(", code) or re.search(r"\bsvfloat32_t\b", code) or "_f32" in code:
        return 32
    if re.search(r"\bsvcnth\s*\(", code) or re.search(r"\bsvuint16_t\b", code) or "_u16" in code:
        return 16
    if re.search(r"\bsvcntb\s*\(", code) or re.search(r"\bsvuint8_t\b", code) or "_u8" in code:
        return 8
    return None
# ---------------------------------------------------------------------
# Safe SVE postprocess fixes (compile-focused, minimal semantic risk)
# ---------------------------------------------------------------------

# Some invented/invalid SVE call names have shown up in model repairs. Even if a whitelist
# accidentally contains them, we treat them as invalid to force another repair pass.
_FORCE_INVALID_SVE_CALLS: Set[str] = {
    "svsel_z",
    "sve_max_vl",
    "srvshrn_z",
    "svpadd_s32",
}

def postprocess_sve_common_fixes(code: str) -> Tuple[str, Dict]:
    """
    Apply a few *safe* textual fixes for common SVE codegen pitfalls that frequently
    appear in LLM output and lead to obvious compile errors.

    Returns: (new_code, info_dict)
    """
    info: Dict[str, int] = {
        "svcnt_fix": 0,
        "svwhilelt_casts": 0,
        "svcntp_fix": 0,
        "svsel_fix": 0,
        "svindex_fix": 0,
        "svcmp_missing_pg": 0,
        "svcmpn_missing_pg": 0,
        "gather_name_fix": 0,
        "ptr_sizeof_fix": 0,
        "vl_inc_fix": 0,
    }
    if not code:
        return code, info

    out = code
    # --- SVE text sanitization (guard against pseudo syntax) ---
    # Some model outputs contain pseudo tokens like:
    #   svcmpeq/pg/(a, b)   or   svsel/pgt_z(pg, a, b)
    # They will not compile. We strip the bogus "/.../" fragments early so later name/shape repair can work.
    if re.search(r"\bsv[a-zA-Z0-9_]+\s*/", out) or ("/pg/" in out):
        # First: drop a single "/token" suffix when it's immediately followed by "(" (or "/(").
        out = re.sub(r"\b(sv[a-zA-Z0-9_]+)\/[a-zA-Z0-9_]+(?=\s*\/?\s*\()", r"\1", out)
        # Second: drop a leftover slash right before '(' :  svcmpeq/(...) -> svcmpeq(...)
        out = re.sub(r"\b(sv[a-zA-Z0-9_]+)\s*/\s*\(", r"\1(", out)

    # Fix occasional tokenization: "sv foo" -> "svfoo"
    if re.search(r"\bsv\s+[A-Za-z_]", out):
        out = re.sub(r"\bsv\s+([A-Za-z_][A-Za-z0-9_]*)", r"sv\1", out)


    # 1) Fix invented 'svsel_z(...)' -> 'svsel(...)' (drop the 2nd argument if 4-arg form).
    #    Common bad form: svsel_z(pg, pg, a, b)  or  svsel_z(pg, pred, a, b)
    def _fix_svsel_z(m: re.Match) -> str:
        info["svsel_fix"] += 1
        pg = m.group(1).strip()
        a = m.group(3).strip()
        b = m.group(4).strip()
        return f"svsel({pg}, {a}, {b})"

    out, n_svsel = re.subn(
        r"\bsvsel_z\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\)",
        _fix_svsel_z,
        out,
    )
    # If any svsel_z survived (weird formatting), at least rename the symbol to trigger later repair.
    if "svsel_z" in out:
        out = out.replace("svsel_z", "svsel")

    # 2) Fix svcntp_bXX() / svcntp_bXX(x) call shapes.
    #    - No-arg svcntp_* doesn't exist -> use lane-count intrinsic as a reasonable default.
    #    - One-arg form -> add svptrue_bXX() as the counting mask: svcntp_b32(svptrue_b32(), p)
    _cnt_map = {"8": "svcntb", "16": "svcnth", "32": "svcntw", "64": "svcntd"}

    def _fix_svcntp_noargs(m: re.Match) -> str:
        w = m.group(1)
        info["svcntp_fix"] += 1
        return f"{_cnt_map[w]}()"

    out, _ = re.subn(r"\bsvcntp_b(8|16|32|64)\s*\(\s*\)", _fix_svcntp_noargs, out)

    def _fix_svcntp_1arg(m: re.Match) -> str:
        w = m.group(1)
        arg = m.group(2).strip()
        info["svcntp_fix"] += 1
        return f"svcntp_b{w}(svptrue_b{w}(), {arg})"

    # Match 1-arg only (no comma inside).
    out, _ = re.subn(r"\bsvcntp_b(8|16|32|64)\s*\(\s*([^,\)]+?)\s*\)", _fix_svcntp_1arg, out)

    # 3) Disambiguate svwhilelt_bXX(i, n) by casting to uint64_t for simple arguments.
    #    This fixes clang "call to 'svwhilelt_b32' is ambiguous" when i/n are size_t.
    def _fix_svwhilelt(m: re.Match) -> str:
        w = m.group(1)
        a = m.group(2).strip()
        b = m.group(3).strip()
        info["svwhilelt_casts"] += 1
        return f"svwhilelt_b{w}((uint64_t)({a}), (uint64_t)({b}))"

    out, _ = re.subn(
        r"\bsvwhilelt_b(8|16|32|64)\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)",
        _fix_svwhilelt,
        out,
    )

    # Fix common hallucination: svindex_* wrongly taking a predicate as the first arg:
    #   svindex_s32(pg, base, step)  ->  svindex_s32((int32_t)base, (int32_t)step)
    def _fix_svindex_pred_arg(m: re.Match) -> str:
        kind = m.group(1).lower()
        base = m.group(3).strip()
        step = m.group(4).strip()
        cast_map = {
            "s32": "int32_t",
            "u32": "uint32_t",
            "s16": "int16_t",
            "u16": "uint16_t",
            "s8": "int8_t",
            "u8": "uint8_t",
        }
        cty = cast_map.get(kind)
        if cty:
            return f"svindex_{kind}(({cty})({base}), ({cty})({step}))"
        return f"svindex_{kind}({base}, {step})"

    out = re.sub(
        r"\bsvindex_(s32|u32|s16|u16|s8|u8)\s*\(\s*(pg|p\d+|pred|predicate|mask)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)",
        _fix_svindex_pred_arg,
        out,
    )



    # 4) Fix hallucinated svindex_* call shapes like svindex_u16(pg) -> svindex_u16((uint16_t)0, (uint16_t)1)
    def _fix_svindex_1arg(m: re.Match) -> str:
        kind = m.group(1).lower()
        info["svindex_fix"] += 1
        cast_map = {
            "s32": "int32_t",
            "u32": "uint32_t",
            "s16": "int16_t",
            "u16": "uint16_t",
            "s8": "int8_t",
            "u8": "uint8_t",
        }
        cty = cast_map.get(kind)
        if cty:
            return f"svindex_{kind}(({cty})0, ({cty})1)"
        return f"svindex_{kind}(0, 1)"

    out, _ = re.subn(
        r"\bsvindex_(s32|u32|s16|u16|s8|u8)\s*\(\s*(pg|p\d+|pred|predicate|mask)\s*\)",
        _fix_svindex_1arg,
        out,
    )

    # 5) Insert missing predicate argument for svcmp* calls that were emitted as 2-arg forms.
    #    E.g. svcmpeq_f64(a,b) -> svcmpeq_f64(pg, a, b)
    #         svcmpeq_n_u16(v, 0) -> svcmpeq_n_u16(pg, v, 0)
    #
    #    NOTE: be careful to avoid self-referential inserts like:
    #      svbool_t pg_vec = svcmpne_f64(pg_vec, a, b);   // pg_vec is uninitialized here
    #    If the LHS variable matches the chosen pg, we fall back to svptrue_bXX().
    def _infer_width_bits_from_cmp_name(name: str) -> str:
        lname = name.lower()
        if any(s in lname for s in ["_f64", "_s64", "_u64"]):
            return "64"
        if any(s in lname for s in ["_f32", "_s32", "_u32"]):
            return "32"
        if any(s in lname for s in ["_f16", "_s16", "_u16"]):
            return "16"
        if any(s in lname for s in ["_s8", "_u8"]):
            return "8"
        return "32"

    def _pick_pg_expr(src: str, width: str) -> str:
        # strongest: explicit `pg` variable
        if re.search(r"\bsvbool_t\s+pg\b", src):
            return "pg"
        # next: a loop predicate of matching width
        m = re.search(rf"\bsvbool_t\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*svwhilelt_b{width}\b", src)
        if m:
            return m.group(1)
        m = re.search(rf"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*svwhilelt_b{width}\b", src)
        if m:
            return m.group(1)
        # next: a full-true predicate variable
        m = re.search(rf"\bsvbool_t\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*svptrue_b{width}\b", src)
        if m:
            return m.group(1)
        # safe fallback
        return f"svptrue_b{width}()"

    def _find_matching_paren(s: str, open_pos: int) -> int:
        depth = 0
        for i in range(open_pos, len(s)):
            ch = s[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
        return -1

    def _split_top_level_args(s: str) -> List[str]:
        args: List[str] = []
        cur: List[str] = []
        depth = 0
        for ch in s:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                args.append("".join(cur).strip())
                cur = []
                continue
            cur.append(ch)
        if cur:
            args.append("".join(cur).strip())
        return args

    i0 = 0
    while True:
        mcmp = re.search(r"\b(svcmp[a-zA-Z0-9_]*)\s*\(", out[i0:])
        if not mcmp:
            break
        fname = mcmp.group(1)
        open_pos = i0 + mcmp.end() - 1  # position of '('
        close_pos = _find_matching_paren(out, open_pos)
        if close_pos < 0:
            break
        inside = out[open_pos + 1 : close_pos]
        args = _split_top_level_args(inside)

        if len(args) == 2:
            width = _infer_width_bits_from_cmp_name(fname)
            pg_expr = _pick_pg_expr(out, width)

            # Avoid self-referential use when the call is initializing/assigning the same predicate variable.
            # Look back to the start of the current line.
            call_abs_pos = i0 + mcmp.start()
            line_start = out.rfind("\n", 0, call_abs_pos) + 1
            prefix = out[line_start:call_abs_pos]
            lhs = None
            m_lhs = re.search(r"\bsvbool_t\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*$", prefix)
            if not m_lhs:
                m_lhs = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*$", prefix)
            if m_lhs:
                lhs = m_lhs.group(1)
            if lhs and pg_expr == lhs:
                pg_expr = f"svptrue_b{width}()"

            new_inside = f"{pg_expr}, {inside.strip()}"
            out = out[: open_pos + 1] + new_inside + out[close_pos:]
            if "_n_" in fname:
                info["svcmpn_missing_pg"] += 1
            else:
                info["svcmp_missing_pg"] += 1
            i0 = close_pos + 1 + (len(new_inside) - len(inside))
        else:
            i0 = close_pos + 1
    # 6) Fix common hallucinated gather names.
    #    - svld1_gather_z(...) -> svld1_gather_u32index_f32(...) (heuristic)
    #    - svld1_gather_index_f32(...) -> svld1_gather_u32index_f32(...)
    if ("svld1_gather_z" in out) or ("svld1_gather_index_" in out):
        # element type
        elem = "f32"
        if ("svfloat64_t" in out) or re.search(r"\bdouble\b", out):
            elem = "f64"
        # index width
        idxw = "u32"
        if ("svuint64_t" in out) or ("svint64_t" in out):
            idxw = "u64"
        repl_name = f"svld1_gather_{idxw}index_{elem}"
        before = out
        out = re.sub(r"\bsvld1_gather_z\b", repl_name, out)
        out = re.sub(r"\bsvld1_gather_index_(f32|f64)\b", lambda m: f"svld1_gather_{idxw}index_{m.group(1)}", out)
        if out != before:
            info["gather_name_fix"] += 1

    # 7) Fix a very common memory-corruption pattern: multiplying by sizeof(T) when indexing a T*.
    #    This shows up in repairs and tends to trigger glibc malloc corruption / SIGSEGV.
    #    We only touch assignments that are later used in "ptr + offset_var" or "&ptr[offset_var]" where ptr is a T*.
    def _parse_pointer_params(src: str) -> Dict[str, str]:
        ptrs: Dict[str, str] = {}
        # very lightweight: look at the first function signature
        msig = re.search(r"\bvoid\s+[A-Za-z_][A-Za-z0-9_]*\s*\(([^)]*)\)", src)
        if not msig:
            return ptrs
        params = msig.group(1)
        for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*_t|float|double|int|unsigned|char)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)", params):
            base = m.group(1)
            name = m.group(2)
            # strip qualifiers like 'const' if present elsewhere (best-effort)
            ptrs[name] = base
        return ptrs

    ptr_params = _parse_pointer_params(out)
    for ptr_name, base_ty in ptr_params.items():
        # Skip byte pointers: sizeof-multiplication is sometimes intentional there.
        if base_ty in ("uint8_t", "int8_t", "char"):
            continue
        # Fix direct inline "ptr + expr*sizeof(base_ty)"
        out2, n_inline = re.subn(
            rf"\b{re.escape(ptr_name)}\s*\+\s*([A-Za-z_][A-Za-z0-9_]*|\([^)]*\)|[0-9]+)\s*\*\s*sizeof\s*\(\s*{re.escape(base_ty)}\s*\)",
            rf"{ptr_name} + \1",
            out,
        )
        if n_inline:
            info["ptr_sizeof_fix"] += n_inline
            out = out2

        # Fix assignment form: <offset> = <expr> * sizeof(base_ty);
        # only if <offset> is later used with this pointer.
        used_offsets = set(re.findall(rf"\b{re.escape(ptr_name)}\s*\+\s*([A-Za-z_][A-Za-z0-9_]*)\b", out))
        used_offsets |= set(re.findall(rf"&\s*{re.escape(ptr_name)}\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]", out))
        for off in sorted(used_offsets):
            pat = re.compile(
                rf"(?m)^(\s*(?:[A-Za-z_][A-Za-z0-9_<>\s\*]*?\s+)?)({re.escape(off)})\s*=\s*([^;]*?)\s*\*\s*sizeof\s*\(\s*{re.escape(base_ty)}\s*\)\s*;",
            )
            def _repl(mo: re.Match) -> str:
                info["ptr_sizeof_fix"] += 1
                return f"{mo.group(1)}{mo.group(2)} = {mo.group(3).strip()};"
            out = pat.sub(_repl, out)

    # 8) Fix common vector-length increment mismatches with svwhilelt_bXX.
    #    E.g. svwhilelt_b16(i, n) but i += svcntw();  -> i += svcnth();
    def _fix_vl_inc(width: str, wrong: str, right: str) -> None:
        nonlocal out
        pat = re.compile(rf"\b([A-Za-z_][A-Za-z0-9_]*)\s*\+=\s*{wrong}\s*\(\s*\)\s*;")
        def _repl(mo: re.Match) -> str:
            var = mo.group(1)
            # only if the same var is used in svwhilelt_b{width}(var, ...)
            if re.search(rf"\bsvwhilelt_b{width}\s*\(\s*(?:\(uint64_t\)\()?(\()?{re.escape(var)}\b", out):
                info["vl_inc_fix"] += 1
                return f"{var} += {right}();"
            return mo.group(0)
        out = pat.sub(_repl, out)

    _fix_vl_inc("16", "svcntw", "svcnth")
    _fix_vl_inc("8", "svcntw", "svcntb")
    _fix_vl_inc("64", "svcntw", "svcntd")

        # 8) Fix svcntb/svcnth/svcntw/svcntd called with any arguments -> no-arg form.
    #    Some models hallucinate svcntb(pg) etc; the correct intrinsics take no arguments.
    def _fix_svcnt_anyargs(m: re.Match) -> str:
        args = (m.group(2) or "").strip()
        if not args:
            return m.group(0)
        info["svcnt_fix"] += 1
        return f"{m.group(1)}()"

    out = re.sub(r"\b(svcnt[bhwd])\s*\(\s*([^\)]*)\)", _fix_svcnt_anyargs, out)

    # 9) Normalize common alias/typo intrinsic names and lower a few recurring pseudo-intrinsics.
    #    This is a root-cause hygiene pass to prevent "svor/svshr/svslt/svrem" style hallucinations
    #    from surviving into later stages.
    out, info2 = normalize_sve_aliases_and_lower_pseudo_intrinsics(out)
    for k, v in info2.items():
        info[k] = info.get(k, 0) + v

    return out, info




def _insert_after_includes(code: str, block: str) -> str:
    lines = code.splitlines(True)
    inc_idxs = [i for i, l in enumerate(lines[:200]) if l.lstrip().startswith("#include")]
    if inc_idxs:
        i = inc_idxs[-1] + 1
        return "".join(lines[:i]) + block + "".join(lines[i:])
    return block + code


def normalize_sve_aliases_and_lower_pseudo_intrinsics(code: str) -> Tuple[str, Dict]:
    """
    Root-cause hygiene pass:
    - Normalize a handful of *very common* SVE intrinsic alias/typo families that appear in LLM output
      (svor->svorr, svxor->sveor, svshl->svlsl, svshr->svlsr/svasr, svslt->svcmplt, svcmplo->svcmplt).
    - Lower a small set of recurring pseudo-intrinsics (svrem_*, svdivu_*, svbtst_*, svgetlane_*, svzero_*)
      into compilable code (typically via scalar per-lane fallback guarded by the predicate).

    This function is intentionally conservative and only triggers when those exact pseudo names appear.
    """
    info: Dict[str, int] = {
        "alias_fix": 0,
        "pseudo_lower": 0,
        "helper_injected": 0,
        "non_ascii_reject": 0,
    }

    out = code

    # ---- A) Basic alias/typo normalizations (name-only) ----
    def _subn(pat: str, repl: str, s: str) -> str:
        new_s, n = re.subn(pat, repl, s)
        if n:
            info["alias_fix"] += n
        return new_s

    # svor_* -> svorr_*  (ORR)
    out = _subn(r"\bsvor_(?=[a-z0-9_]+\b)", "svorr_", out)
    # svxor_* -> sveor_* (EOR)
    out = _subn(r"\bsvxor_(?=[a-z0-9_]+\b)", "sveor_", out)
    # svshl_* -> svlsl_* (logical shift left)
    out = _subn(r"\bsvshl_(?=[a-z0-9_]+\b)", "svlsl_", out)

    # svshr_{...} -> svlsr_{...} for unsigned, svasr_{...} for signed
    def _fix_svshr(m: re.Match) -> str:
        n = m.group(1) or ""
        su = m.group(2)
        w = m.group(3)
        tail = m.group(4) or ""
        info["alias_fix"] += 1
        op = "svasr" if su == "s" else "svlsr"
        return f"{op}{n}_{su}{w}{tail}"

    out = re.sub(r"\bsvshr(_n)?_(s|u)(8|16|32|64)(_[a-z0-9]+)?\b", _fix_svshr, out)

    # Signed-compare aliases
    out = _subn(r"\bsvslt(?=_)", "svcmplt", out)
    out = _subn(r"\bsvsle(?=_)", "svcmple", out)
    out = _subn(r"\bsvsgt(?=_)", "svcmpgt", out)
    out = _subn(r"\bsvsge(?=_)", "svcmpge", out)
    # Common typo: svcmplo_n_* -> svcmplt_n_*
    out = _subn(r"\bsvcmplo_n_(?=[a-z0-9_]+\b)", "svcmplt_n_", out)

    # ---- B) Pseudo "zero vector" helpers: svzero_u32() -> svdup_n_u32(0) ----
    def _fix_svzero(m: re.Match) -> str:
        kind = m.group(1)
        info["pseudo_lower"] += 1
        if kind.startswith("f"):
            lit = "0.0f" if kind == "f32" else "0.0"
            return f"svdup_n_{kind}({lit})"
        return f"svdup_n_{kind}(0)"

    out = re.sub(r"\bsvzero_(u8|u16|u32|u64|s8|s16|s32|s64|f16|f32|f64)\s*\(\s*\)", _fix_svzero, out)

    # ---- C) Lower a few pseudo-intrinsics via injected helpers (only when they appear) ----
    helpers: List[str] = []

    def _helper_block(texts: List[str]) -> str:
        if not texts:
            return ""
        hdr = "\n/* __SB_PSEUDO_INTRINSIC_HELPERS_BEGIN */\n"
        ftr = "/* __SB_PSEUDO_INTRINSIC_HELPERS_END */\n\n"
        body = "\n\n".join(texts).rstrip() + "\n\n"

        inc = ""
        if ("#include <stdint.h>" not in out) and ("#include <cstdint>" not in out):
            inc = "#include <stdint.h>\n"

        common = (
            "#ifndef __SB_SVE_MAX_BYTES\n"
            "#define __SB_SVE_MAX_BYTES 256\n"
            "#endif\n\n"
        )
        return hdr + inc + common + body + ftr

    # svrem_s32_{z,m}
    if re.search(r"\bsvrem_s32_(z|m)\s*\(", out):
        out = re.sub(r"\bsvrem_s32_z\b", "__sb_svrem_s32_z", out)
        out = re.sub(r"\bsvrem_s32_m\b", "__sb_svrem_s32_m", out)
        info["pseudo_lower"] += 1
        helpers.append('''
static inline svint32_t __sb_svrem_s32_z(svbool_t pg, svint32_t a, svint32_t b) {
    alignas(64) int32_t aa[__SB_SVE_MAX_BYTES / 4];
    alignas(64) int32_t bb[__SB_SVE_MAX_BYTES / 4];
    alignas(64) uint32_t mm[__SB_SVE_MAX_BYTES / 4];
    alignas(64) int32_t rr[__SB_SVE_MAX_BYTES / 4];
    svbool_t all = svptrue_b32();
    svst1_s32(all, aa, a);
    svst1_s32(all, bb, b);
    svst1_u32(all, mm, svsel_u32(pg, svdup_n_u32(1), svdup_n_u32(0)));
    int vl = (int)svcntw();
    for (int i = 0; i < vl; ++i) {
        rr[i] = mm[i] ? (aa[i] % bb[i]) : 0;
    }
    return svld1_s32(all, rr);
}

static inline svint32_t __sb_svrem_s32_m(svbool_t pg, svint32_t a, svint32_t b) {
    alignas(64) int32_t aa[__SB_SVE_MAX_BYTES / 4];
    alignas(64) int32_t bb[__SB_SVE_MAX_BYTES / 4];
    alignas(64) uint32_t mm[__SB_SVE_MAX_BYTES / 4];
    alignas(64) int32_t rr[__SB_SVE_MAX_BYTES / 4];
    svbool_t all = svptrue_b32();
    svst1_s32(all, aa, a);
    svst1_s32(all, bb, b);
    svst1_u32(all, mm, svsel_u32(pg, svdup_n_u32(1), svdup_n_u32(0)));
    int vl = (int)svcntw();
    for (int i = 0; i < vl; ++i) {
        rr[i] = mm[i] ? (aa[i] % bb[i]) : aa[i];
    }
    return svld1_s32(all, rr);
}
'''.strip("\n"))

    # svdivu_z (pseudo int32 division)
    if re.search(r"\bsvdivu_z\s*\(", out):
        out = re.sub(r"\bsvdivu_z\b", "__sb_svdiv_s32_z", out)
        info["pseudo_lower"] += 1
        helpers.append('''
static inline svint32_t __sb_svdiv_s32_z(svbool_t pg, svint32_t a, svint32_t b) {
    alignas(64) int32_t aa[__SB_SVE_MAX_BYTES / 4];
    alignas(64) int32_t bb[__SB_SVE_MAX_BYTES / 4];
    alignas(64) uint32_t mm[__SB_SVE_MAX_BYTES / 4];
    alignas(64) int32_t rr[__SB_SVE_MAX_BYTES / 4];
    svbool_t all = svptrue_b32();
    svst1_s32(all, aa, a);
    svst1_s32(all, bb, b);
    svst1_u32(all, mm, svsel_u32(pg, svdup_n_u32(1), svdup_n_u32(0)));
    int vl = (int)svcntw();
    for (int i = 0; i < vl; ++i) {
        rr[i] = mm[i] ? (aa[i] / bb[i]) : 0;
    }
    return svld1_s32(all, rr);
}
'''.strip("\n"))

    # svgetlane_s32(v, idx)
    if re.search(r"\bsvgetlane_s32\s*\(", out):
        out = re.sub(r"\bsvgetlane_s32\b", "__sb_getlane_s32", out)
        info["pseudo_lower"] += 1
        helpers.append('''
static inline int32_t __sb_getlane_s32(svint32_t v, int idx) {
    alignas(64) int32_t aa[__SB_SVE_MAX_BYTES / 4];
    svbool_t all = svptrue_b32();
    svst1_s32(all, aa, v);
    int vl = (int)svcntw();
    if (vl <= 0) return 0;
    if (idx < 0) idx = 0;
    if (idx >= vl) idx = vl - 1;
    return aa[idx];
}
'''.strip("\n"))

    # svbtst_n_s32(pg, v, bit)
    if re.search(r"\bsvbtst_n_s32\s*\(", out):
        out = re.sub(r"\bsvbtst_n_s32\b", "__sb_btst_n_s32", out)
        info["pseudo_lower"] += 1
        helpers.append('''
static inline svbool_t __sb_btst_n_s32(svbool_t pg, svint32_t v, int bit) {
    uint32_t m = 0;
    if (bit >= 0 && bit < 32) m = (1U << (uint32_t)bit);
    svint32_t mask = svdup_n_s32((int32_t)m);
    svint32_t x = svand_s32_x(pg, v, mask);
    return svcmpne_n_s32(pg, x, 0);
}
'''.strip("\n"))

    block = _helper_block(helpers)
    if block and "__SB_PSEUDO_INTRINSIC_HELPERS_BEGIN" not in out:
        out = _insert_after_includes(out, block)
        info["helper_injected"] += 1

        # ---- D) Normalize non-ASCII identifiers into ASCII placeholders ----
    # Some models emit Chinese identifiers (e.g. `sve元素数`) which will not compile.
    # We do a *pure renaming* (string-substitution) so semantics are unchanged.
    try:
        stripped = strip_comments(out)
    except Exception:
        stripped = out
    tokens = sorted(set(re.findall(r"[^\x00-\x7F]+", stripped)), key=len, reverse=True)
    if tokens:
        info["non_ascii_reject"] += len(tokens)
        for i, tok in enumerate(tokens):
            out = out.replace(tok, f"sb_nonascii_{i}")

    return out, info

def guess_type_tokens(code: str) -> List[str]:
    score = {
        "f32": len(re.findall(r"\bsvfloat32_t\b", code)),
        "f64": len(re.findall(r"\bsvfloat64_t\b", code)),
        "u64": len(re.findall(r"\bsvuint64_t\b", code)),
        "u32": len(re.findall(r"\bsvuint32_t\b", code)),
        "s64": len(re.findall(r"\bsvint64_t\b", code)),
        "s32": len(re.findall(r"\bsvint32_t\b", code)),
        "u16": len(re.findall(r"\bsvuint16_t\b", code)),
        "u8":  len(re.findall(r"\bsvuint8_t\b", code)),
    }
    return [k for k, v in sorted(score.items(), key=lambda x: x[1], reverse=True) if v > 0]

def apply_name_rule_fixes(code: str) -> str:
    code = re.sub(r"\bsvcntf32\s*\(\s*\)", "svcntw()", code)
    code = re.sub(r"\bsvcntf64\s*\(\s*\)", "svcntd()", code)
    if "svwhilelt" in code:
        lane = infer_lane_width(code) or 64
        code = re.sub(r"\bsvwhilelt\s*\(", f"svwhilelt_b{lane}(", code)
    return code

# =============================================================================
# Deterministic SVE gather/scatter rewrites (NEW)
# -----------------------------------------------------------------------------
# Some LLMs hallucinate non-existent intrinsics like:
#   - svld1_gather_s64offset_s64(...)
#   - svst1_scatter_s64offset_s64(...)
# even though the ACLE names are, e.g.:
#   - svld1_gather_s64offset(pg, base, offsets)
#   - svst1_scatter_s64offset(pg, base, offsets, data)
#
# This rewrite is purely syntactic and preserves argument order; it only removes
# the redundant trailing element-type suffix ("_s64", "_u32", ...).
# It is intentionally conservative: it only triggers when the intrinsic name
# matches the known gather/scatter patterns and ends in an extra type suffix.
#
# This fix avoids unnecessary LLM "repair" calls and prevents avoidable compile
# failures, without affecting other logic.
# =============================================================================

# Matches: svld1_gather_s64offset_s64 / svst1_scatter_u32offset_u32 etc.
_SVE_GATHER_SCATTER_S64_RE = re.compile(
    r"\bsv(?P<op>ld1|st1)_(?P<kind>gather|scatter)_(?P<off>s64offset|u64offset|s32offset|u32offset)"
    r"_(?P<tail>[a-z0-9]+)\b"
)

def _sve_gs_extract_tail_lane(tail: str, code: str, *, default_lane: int = 64) -> int:
    """Infer lane width (8/16/32/64) from the bogus tail suffix if possible.

    Examples:
      tail="s64" -> 64
      tail="u32" -> 32
      tail="f16" -> 16
      tail="bf16" -> 16
    """
    m = re.search(r"(?:bf)?(?:f|s|u)(8|16|32|64)\b", tail)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return default_lane
    return infer_lane_width(code) or default_lane

_SVWHILELT_LANE_RE = re.compile(r"\bsvwhilelt_b(8|16|32|64)\s*\(")

def apply_predicate_width_consistency_fixes(code: str) -> str:
    """
    Conservative fixes:
    - If code uses svwhilelt_b8/b16/b32, but uses svcntp_b64 or svptest_any(svptrue_b64(), ...),
      rewrite the suffix to match the lane width. This reduces common counting/loop predicate mistakes.
    """
    s = str(code or "")
    m = _SVWHILELT_LANE_RE.search(s)
    if not m:
        return s
    lane = int(m.group(1))

    if lane in (8, 16, 32, 64):
        if lane != 64:
            s = re.sub(r"\bsvcntp_b64\s*\(", f"svcntp_b{lane}(", s)
            s = re.sub(r"\bsvptest_any\s*\(\s*svptrue_b64\s*\(\s*\)\s*,", f"svptest_any(svptrue_b{lane}(),", s)
        # also normalize svptest_any's first arg if someone used mismatched width
        s = re.sub(r"\bsvptest_any\s*\(\s*svptrue_b(8|16|32|64)\s*\(\s*\)\s*,", f"svptest_any(svptrue_b{lane}(),", s)
    return s

def apply_sve_gather_scatter_rewrites(code: str) -> str:
    """Rewrite bogus SVE gather/scatter intrinsic names into valid ACLE forms.

    This is a no-op unless a known hallucination pattern is detected.
    """
    if "svld1_gather_" not in code and "svst1_scatter_" not in code:
        return code

    def repl(m: re.Match) -> str:
        op = m.group("op")
        kind = m.group("kind")
        off = m.group("off")
        tail = m.group("tail") or ""
        lane = _sve_gs_extract_tail_lane(tail, code)
        # Only strip the tail if it matches the lane width we inferred.
        if re.search(rf"(?:bf)?(?:f|s|u){lane}\b", tail):
            return f"sv{op}1_{kind}_{off}"
        # Otherwise, keep original to avoid accidental harm.
        return m.group(0)

    return _SVE_GATHER_SCATTER_S64_RE.sub(repl, code)



# =============================================================================
# Completion snippet structural checks / normalization (NEW)
# =============================================================================

_PROMPT_FUNC_RE = re.compile(
    r"(?ms)^\s*(?:template\s*<[^;{}]*>\s*)?"
    r"(?:static\s+|inline\s+|constexpr\s+)?(?:extern\s+\"C\"\s+)?"
    r"(?:[\w:<>*&\[\]]+\s+|__attribute__\s*\(\([^)]*\)\)\s*)+"
    r"([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{"
)

# Likely a full function definition (illegal inside the completion snippet).
_TOPLEVEL_FUNC_DEF_RE = re.compile(
    r"(?ms)^\s*(?:template\s*<[^;{}]*>\s*)?"
    r"(?:static\s+|inline\s+|constexpr\s+)?"
    r"(?:extern\s+\"C\"\s+)?"
    r"(?:[\w:<>*&\[\]]+\s+|__attribute__\s*\(\([^)]*\)\)\s*)+"
    r"([A-Za-z_]\w*)\s*\([^;]*\)\s*\{"
)

def extract_function_name_from_prompt_prefix(prompt_prefix: str) -> Optional[str]:
    """
    Best-effort extraction of the target function name from the prompt prefix.
    This is used only for structural validation (to prevent the model from re-defining the function).
    """
    if not prompt_prefix:
        return None
    s = strip_comments(str(prompt_prefix))
    ms = list(_PROMPT_FUNC_RE.finditer(s))
    if not ms:
        return None
    return ms[-1].group(1)

def extract_function_decl_from_prompt_prefix(prompt_prefix: str, func_name: Optional[str]) -> Optional[str]:
    """Extract the *exact* target function declaration (up to and including '{') from the prompt.

    This is used as a safety net for COMPLETION_MODE_FULL: if the model accidentally emits only a
    function *body* (a "snippet"), we can wrap it back into a valid translation unit with the
    correct function signature, preventing file-scope SVE types (e.g., svbool_t) that fail to compile.
    """
    if not prompt_prefix or not func_name:
        return None

    # Work on comment-free prompt text.  SimdBench prompts often contain examples such as
    # ">>> foo(3)" inside the task comment; using raw text lets the regex/fallback mistake those
    # examples for the target declaration and then wrap them into the generated source.
    prompt_text = strip_comments(str(prompt_prefix))

    # First try a reasonably strict regex that captures the full declaration line(s).
    # Allow multi-line parameter lists but stop at braces/semicolons to avoid over-matching.
    pat = re.compile(
        rf"(?ms)^\s*(?:template\s*<[^;{{}}]*>\s*)?"
        rf"(?:static\s+|inline\s+|constexpr\s+)?(?:extern\s+\"C\"\s+)?"
        rf"(?:[\w:<>*&\[\]]+\s+|__attribute__\s*\(\([^)]*\)\)\s*)+"
        rf"{re.escape(func_name)}\s*\([^;{{}}]*?\)\s*\{{"
    )
    m = pat.search(prompt_text)
    if m:
        return m.group(0).strip()

    # Fallback: find the last occurrence of 'func_name(' and then take the slice up to the next '{'.
    # This fallback also works on comment-free text and only accepts declaration-like occurrences:
    # the matching ')' must be followed by the function body's opening brace.
    hits = list(re.finditer(rf"\b{re.escape(func_name)}\s*\(", prompt_text))
    if not hits:
        return None

    for mh in reversed(hits):
        open_paren = prompt_text.find("(", mh.start())
        close_paren = _find_matching_paren(prompt_text, open_paren)
        if close_paren < 0:
            continue

        brace_pos = close_paren + 1
        while brace_pos < len(prompt_text) and prompt_text[brace_pos].isspace():
            brace_pos += 1
        if brace_pos >= len(prompt_text) or prompt_text[brace_pos] != "{":
            continue

        line_start = prompt_text.rfind("\n", 0, mh.start())
        line_start = 0 if line_start < 0 else line_start + 1

        # Include declaration prefixes placed on previous lines, e.g. template/attributes.
        start = line_start
        prev_end = line_start - 1
        while prev_end > 0:
            prev_start = prompt_text.rfind("\n", 0, prev_end)
            prev_start = 0 if prev_start < 0 else prev_start + 1
            prev_line = prompt_text[prev_start:prev_end].strip()
            if not prev_line:
                break
            if prev_line.startswith("#") or prev_line.endswith(";") or prev_line.endswith("}"):
                break
            if re.search(
                r"\b(?:template\s*<|__attribute__|static|inline|constexpr|extern|void|bool|char|short|int|long|float|double|size_t|std::|sv\w+_t|[A-Za-z_]\w*_t)\b",
                prev_line,
            ):
                start = prev_start
                prev_end = prev_start - 1
                continue
            break

        decl = prompt_text[start : brace_pos + 1]
        return decl.strip() if decl else None
    return None


def target_signature_line_from_decl(func_decl: str) -> str:
    decl = str(func_decl or "").strip()
    if not decl:
        return ""
    if "*/" in decl:
        decl = decl.rsplit("*/", 1)[-1].strip()
    candidate_lines = [ln.strip() for ln in decl.splitlines() if ln.strip()]
    signature_like = [ln for ln in candidate_lines if "(" in ln and ")" in ln and not ln.startswith((">>>", "Examples:"))]
    if signature_like:
        decl = signature_like[-1]
    if decl.endswith("{"):
        decl = decl[:-1].rstrip()
    return decl.strip()


def _canonical_cpp_signature_type_for_compare(type_text: str) -> str:
    s = " ".join(str(type_text or "").split()).strip()
    if not s:
        return ""
    # Normalize C++ spelling differences that do not change the function type.
    s = re.sub(r"\bstd\s*::\s*size_t\b", "size_t", s)
    s = re.sub(r"\s*::\s*", "::", s)
    s = re.sub(r"\s*<\s*", "<", s)
    s = re.sub(r"\s*>\s*", ">", s)
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\s*&&\s*", "&&", s)
    s = re.sub(r"\s*&\s*", "&", s)
    s = re.sub(r"\s*\*\s*", "*", s)
    s = re.sub(r"\[\s*([^\]]*?)\s*\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", s)
    s = re.sub(r"\b(.+?)\s+const([*&]?)$", r"const \1\2", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_code_signature_line_from_source_by_name(code: str, func_name: str) -> Optional[str]:
    if not code or not func_name:
        return None
    header_pat = re.compile(
        rf'(?m)^\s*(?:static\s+|inline\s+|constexpr\s+)?(?:extern\s+"C"\s+)?(?:[\w:<>*&\[\]\s]+\s+)?{re.escape(func_name)}\s*\('
    )
    text = str(code)
    for m in header_pat.finditer(text):
        start = m.start()
        open_paren = text.find("(", m.start())
        if open_paren < 0:
            continue
        depth = 0
        close_paren = -1
        for idx in range(open_paren, len(text)):
            ch = text[idx]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    close_paren = idx
                    break
        if close_paren < 0:
            continue
        tail_idx = close_paren + 1
        while tail_idx < len(text) and text[tail_idx].isspace():
            tail_idx += 1
        if tail_idx >= len(text) or text[tail_idx] != "{":
            continue
        return " ".join(text[start : close_paren + 1].split()).strip()
    return None


def build_target_signature_closure_check(
    expected_signature_line: str,
    expected_signature: Dict[str, Any],
    code: str,
) -> Dict[str, Any]:
    expected_name = str(expected_signature.get("name") or "").strip()
    if not expected_name:
        return {"match": False, "mismatches": ["missing_expected_signature"], "actual_signature_line": ""}

    actual_line = extract_code_signature_line_from_source_by_name(code, expected_name)
    if not actual_line:
        return {
            "match": False,
            "mismatches": ["target_function_signature_not_found"],
            "actual_signature_line": "",
        }

    actual_signature = parse_shared_signature(actual_line) if parse_shared_signature else {}
    actual_signature = actual_signature if isinstance(actual_signature, dict) else {}
    mismatches: List[str] = []
    if str(expected_signature.get("name") or "") != str(actual_signature.get("name") or ""):
        mismatches.append("function_name")
    exp_ret = _canonical_cpp_signature_type_for_compare(expected_signature.get("return_type") or "")
    act_ret = _canonical_cpp_signature_type_for_compare(actual_signature.get("return_type") or "")
    if exp_ret != act_ret:
        mismatches.append("return_type")

    exp_params = expected_signature.get("params", []) or []
    act_params = actual_signature.get("params", []) or []
    if len(exp_params) != len(act_params):
        mismatches.append("param_count")
    else:
        for idx, (exp, act) in enumerate(zip(exp_params, act_params)):
            exp_type = _canonical_cpp_signature_type_for_compare((exp or {}).get("type") or "")
            act_type = _canonical_cpp_signature_type_for_compare((act or {}).get("type") or "")
            if exp_type != act_type:
                mismatches.append(f"param_type_{idx}")

    return {
        "match": not mismatches,
        "mismatches": mismatches,
        "actual_signature_line": actual_line,
    }

def wrap_snippet_into_full_translation_unit(snippet_code: str, func_decl: str) -> str:
    """Wrap a snippet (function body) into a full translation unit using func_decl.

    - Keeps any preprocessor lines ('#include', '#define', ...) at file scope.
    - Attempts to remove a single stray trailing '}' if the snippet already closed the function.
    """
    if not func_decl:
        return snippet_code

    preproc_lines: List[str] = []
    body_lines: List[str] = []
    for ln in str(snippet_code or "").splitlines():
        if ln.lstrip().startswith("#"):
            preproc_lines.append(ln.rstrip())
        else:
            body_lines.append(ln.rstrip())

    body = "\n".join(body_lines).strip()

    # If the body has exactly one extra closing brace, assume it's a function-closing brace and drop it.
    ok, delta = brace_delta_matches_expected(body, 0)
    if (not ok) and delta == 1:
        pos = body.rfind("}")
        if pos >= 0:
            body = body[:pos].rstrip()

    out_parts: List[str] = []
    if preproc_lines:
        out_parts.append("\n".join(preproc_lines).strip())
    out_parts.append(func_decl.rstrip())
    if body:
        out_parts.append(body)
    out_parts.append("}")
    return "\n".join(out_parts).strip()

def strip_preprocessor_lines(code: str) -> str:
    if not code:
        return ""
    return "\n".join([ln for ln in str(code).splitlines() if not ln.lstrip().startswith("#")])


_INCLUDE_ANGLE_RE = re.compile(r"(?m)^\s*#\s*include\s*<\s*([^>]+?)\s*>\s*$")


def extract_angle_includes(text: str) -> List[str]:
    """Return a de-duplicated list of <...> include headers (in order of appearance)."""
    headers: List[str] = []
    for m in _INCLUDE_ANGLE_RE.finditer(str(text or "")):
        h = m.group(1).strip()
        if h and h not in headers:
            headers.append(h)
    return headers


def ensure_angle_includes_present(code: str, required_headers: Sequence[str]) -> str:
    """Ensure `#include <...>` for each header is present; if missing, prepend them."""
    s = str(code or "")
    if not required_headers:
        return s

    missing: List[str] = []
    for h in required_headers:
        if not h:
            continue
        pat = rf"(?m)^\s*#\s*include\s*<\s*{re.escape(h)}\s*>\s*$"
        if not re.search(pat, s):
            missing.append(h)

    if not missing:
        return s

    insert = "\n".join([f"#include <{h}>" for h in missing]).strip() + "\n"
    if not s.strip():
        return insert

    # Prepend before existing content.
    return insert + "\n" + s.lstrip("\n")

def _find_matching_brace(src: str, brace_pos: int) -> int:
    """
    Returns the index of the matching '}' for the '{' at brace_pos, or -1.
    Ignores braces inside strings/chars/comments (best-effort).
    """
    if brace_pos < 0 or brace_pos >= len(src) or src[brace_pos] != "{":
        return -1

    depth = 0
    in_str = False
    in_chr = False
    str_ch = ""
    in_line_comment = False
    in_block_comment = False

    i = brace_pos
    while i < len(src):
        ch = src[i]

        # end line comment
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        # end block comment
        if in_block_comment:
            if ch == "*" and (i + 1) < len(src) and src[i + 1] == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        # start comments
        if (not in_str) and (not in_chr):
            if ch == "/" and (i + 1) < len(src):
                nxt = src[i + 1]
                if nxt == "/":
                    in_line_comment = True
                    i += 2
                    continue
                if nxt == "*":
                    in_block_comment = True
                    i += 2
                    continue

        # strings / chars
        if in_str:
            if ch == str_ch and (i == 0 or src[i - 1] != "\\"):  # noqa: W605
                in_str = False
            i += 1
            continue
        if in_chr:
            if ch == "'" and (i == 0 or src[i - 1] != "\\"):  # noqa: W605
                in_chr = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            str_ch = '"'
            i += 1
            continue
        if ch == "'":
            in_chr = True
            i += 1
            continue

        # braces
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    return -1

def try_extract_function_body(code: str, func_name: Optional[str]) -> Optional[str]:
    """
    If `code` contains a full function definition (most commonly because the model
    repeated the prompt prefix), extract the function body and return a valid completion
    snippet: <body> + '\\n}'.

    Returns None if not detected or extraction fails.
    """
    if not code:
        return None
    src = str(code)

    if func_name:
        # Find the *last* occurrence to avoid grabbing helper mention in comments.
        hits = list(re.finditer(rf"\b{re.escape(func_name)}\s*\(", src))
        if hits:
            m = hits[-1]
            b = src.find("{", m.end())
            if b >= 0:
                e = _find_matching_brace(src, b)
                if e > b:
                    body = src[b + 1:e].strip()
                    return (body + "\n}").strip()

    # Fallback: extract the first function-looking block.
    m2 = _TOPLEVEL_FUNC_DEF_RE.search(src)
    if m2:
        b = src.find("{", m2.end())
        if b >= 0:
            e = _find_matching_brace(src, b)
            if e > b:
                body = src[b + 1:e].strip()
                return (body + "\n}").strip()

    return None

def _strip_strings_and_comments(src: str) -> str:
    # best-effort: remove strings/chars/comments for brace counting / pattern checks
    if not src:
        return ""
    out: List[str] = []
    in_str = False
    in_chr = False
    str_ch = ""
    in_line_comment = False
    in_block_comment = False

    i = 0
    while i < len(src):
        ch = src[i]

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                out.append("\n")
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and (i + 1) < len(src) and src[i + 1] == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if (not in_str) and (not in_chr) and ch == "/" and (i + 1) < len(src):
            nxt = src[i + 1]
            if nxt == "/":
                in_line_comment = True
                i += 2
                continue
            if nxt == "*":
                in_block_comment = True
                i += 2
                continue

        if in_str:
            if ch == str_ch and src[i - 1] != "\\":  # noqa: W605
                in_str = False
            i += 1
            continue
        if in_chr:
            if ch == "'" and src[i - 1] != "\\":  # noqa: W605
                in_chr = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            str_ch = '"'
            i += 1
            continue
        if ch == "'":
            in_chr = True
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)

def expected_brace_delta_for_mode(mode: str) -> int:
    """Return the expected (closes - opens) brace delta for the active completion mode."""
    # - snippet mode: appended after a prompt prefix that already opened the function body -> needs one extra '}'.
    # - full/standalone mode: the completion is a full translation unit -> braces should balance.
    return 1 if mode == COMPLETION_MODE_SNIPPET else 0


def brace_delta_matches_expected(code: str, expected_delta: int) -> Tuple[bool, int]:
    """Returns (ok, delta) where delta=(count('}')-count('{')) ignoring comments/strings."""
    s = _strip_strings_and_comments(strip_comments(str(code)))
    opens = s.count("{")
    closes = s.count("}")
    delta = closes - opens
    return delta == expected_delta, delta

def detect_forbidden_structures(code: str, func_name: Optional[str]) -> List[str]:
    issues: List[str] = []
    s = str(code or "")

    # Hard constraint across all modes: the bare svwhilelt(...) intrinsic does not exist.
    if "svwhilelt(" in s:
        issues.append("svwhilelt_bare")

    if COMPLETION_MODE == COMPLETION_MODE_SNIPPET:
        # Snippet mode (appended to prompt prefix): preprocessor directives and any
        # top-level function definitions are illegal.
        if re.search(r"(?m)^\s*#", s):
            issues.append("preprocessor")
        if "#include" in s:
            issues.append("include")

        # Re-definition of target function (illegal in snippet).
        if func_name:
            if re.search(rf"\b{re.escape(func_name)}\s*\([^;]*\)\s*\{{", s):
                issues.append("redef_target_function")

        # Any top-level function definition pattern is also illegal in snippet.
        if _TOPLEVEL_FUNC_DEF_RE.search(s):
            issues.append("toplevel_function_def")

    else:
        # Standalone/full-source mode: allow headers and function definitions, but
        # reject prompt-metadata markers that would break compilation.
        if re.search(
            r"(?m)^\s*\[(?:SPEC|TASK|PROMPT_PREFIX|EXPANDED_SPEC|EXPANDED_SPEC_AND_SVE_PLAN|COMPLETION_TO_FIX|INVALID_INTRINSICS|MISMATCHED_CALLS|END)\]\s*$",
            s,
        ):
            issues.append("prompt_markers")

        # Require that the target function is present somewhere in the file.
        if func_name:
            if not re.search(rf"\b{re.escape(func_name)}\s*\(", s):
                issues.append("missing_target_function")

    # Extra hard constraints for SVE/ACLE style outputs:
    # Some models emit pseudo tokens like `svsel/pgt_z(...)` or `svcmpeq/pg/(...)`,
    # or placeholders like `...` inside argument lists. These will never compile.
    try:
        s_chk = strip_comments(s)
        # Strip string/char literals to avoid false positives.
        s_chk = re.sub(r'"(?:\\.|[^"\\])*"', '""', s_chk)
        s_chk = re.sub(r"'(?:\\.|[^'\\])*'", "''", s_chk)

        if (
            ("/pg/" in s_chk)
            or re.search(r"\bsv[a-zA-Z0-9_]+/[a-zA-Z0-9_]+", s_chk)
            or re.search(r"\bsv[a-zA-Z0-9_]+\s*/\s*\(", s_chk)
        ):
            issues.append("sve_slash_token")

        if re.search(r"\bsv\s+[A-Za-z_]", s_chk):
            issues.append("sve_space_token")

        if "..." in s_chk:
            issues.append("ellipsis_placeholder")

        if re.search(r"\bsvcntp_\b", s_chk):
            issues.append("incomplete_sve_symbol")
    except Exception:
        # Never fail hard in the detector.
        pass

    expected_delta = expected_brace_delta_for_mode(COMPLETION_MODE)
    ok_delta, delta = brace_delta_matches_expected(s, expected_delta)
    if not ok_delta:
        issues.append(f"brace_delta_{delta}")

    return issues

def normalize_completion_snippet(code: str, func_name: Optional[str], *, skip_required_includes: bool = False) -> str:
    """
    Normalize model output into the exact form we compile.

    Modes:
      - snippet: output is a function-body snippet that will be appended to the prompt prefix.
      - full: output is a standalone C/C++ translation unit compiled as-is.

    Notes:
      - We always strip markdown fences.
      - We always apply deterministic intrinsic name fixes (svwhilelt->svwhilelt_b*).
    - In full mode, we preserve preprocessor directives and ensure required <...> includes
        from the current prompt are present unless skip_required_includes=True.
    """
    c = clean_code_block(str(code or "")).strip()
    c = apply_name_rule_fixes(c).strip()
    c = apply_sve_gather_scatter_rewrites(c).strip()
    c = apply_predicate_width_consistency_fixes(c).strip()

    if COMPLETION_MODE == COMPLETION_MODE_SNIPPET:
        # Never allow preprocessor directives inside a function-body snippet.
        if re.search(r"(?m)^\s*#", c):
            c = strip_preprocessor_lines(c).strip()

        # If it still looks like a full function, extract body.
        body = try_extract_function_body(c, func_name)
        if body:
            c = body.strip()

        # Snippet must close the prompt prefix's opening brace.
        if c and not c.rstrip().endswith("}"):
            c = c.rstrip() + "\n}"
        return c.strip()

    # Standalone full-source mode.
    # Safety net: if the model accidentally emitted a "snippet" (function body only) in full mode,
    # wrap it into the exact target signature extracted from the prompt prefix. This prevents
    # file-scope SVE types like svbool_t (sizeless) which fail to compile.
    if func_name and CURRENT_TARGET_FUNC_DECL and (not _TOPLEVEL_FUNC_DEF_RE.search(c)):
        c = wrap_snippet_into_full_translation_unit(c, CURRENT_TARGET_FUNC_DECL).strip()

    if (not skip_required_includes) and CURRENT_REQUIRED_INCLUDES:
        c = ensure_angle_includes_present(c, CURRENT_REQUIRED_INCLUDES).strip()
    return c.strip()


# =============================================================================
# Reduction inference + semantic gate (NEW)
# =============================================================================

# Detect SVE vector reductions (these return scalar)
_SVE_REDUCTION_CALL_RE = re.compile(
    r"\bsv(?:addv|maxv|minv|orv|andv|eorv)\w*\s*\(",
    flags=re.IGNORECASE
)

TASK_MODE_REDUCTION = "reduction"
TASK_MODE_PER_ELEMENT = "per_element"
TASK_MODE_UNKNOWN = "unknown"

def ensure_target_function_named(code: str, func_name: str) -> Tuple[str, bool, str]:
    """Best-effort fix for serial-reference outputs.

    Some serial-reference models may emit a correct scalar implementation but with a different function
    name (e.g. '*_serial'), which makes the harness fail to link/call the expected entrypoint.
    This helper renames the first detected top-level function definition to `func_name` when the
    expected function is missing.

    Returns: (possibly_modified_code, changed, note)
    """
    if not func_name:
        return code, False, "no_func_name"

    matches = list(_TOPLEVEL_FUNC_DEF_RE.finditer(code))
    for m in matches:
        if m.group(1) == func_name:
            return code, False, "already_present"

    if not matches:
        return code, False, "no_toplevel_function_defs"

    # Prefer renaming the last non-main top-level function (LLM outputs often put helpers first).
    m = None
    for cand in reversed(matches):
        if cand.group(1) != "main":
            m = cand
            break
    if m is None:
        m = matches[-1]

    old = m.group(1)
    new_code = code[: m.start(1)] + func_name + code[m.end(1) :]
    return new_code, True, f"renamed_{old}_to_{func_name}"


def _parse_return_type_from_decl(func_decl: str, func_name: Optional[str]) -> str:
    """
    Best-effort parse return type from something like:
      "void foo(const uint32_t* A, uint8_t* out, size_t n) {"
    """
    if not func_decl or not func_name:
        return ""
    s = str(func_decl).strip()
    # remove trailing "{"
    s = s.split("{", 1)[0].strip()
    m = re.search(rf"^\s*(.*?)\b{re.escape(func_name)}\s*\(", s, flags=re.DOTALL)
    if not m:
        return ""
    rt = m.group(1).strip()
    rt = re.sub(r"\s+", " ", rt)
    return rt

def _infer_task_mode_from_text(spec_text: str, func_name: Optional[str]) -> Tuple[str, float, Dict]:
    """
    Infer whether the task is PER-ELEMENT or REDUCTION-LIKE (including *partial reductions* such as
    per-row / per-output-element dot-products / norms / matmul inner sums).

    Returns (mode, confidence in [0,1], debug_info).

    Key design principle (to avoid hurting pass-rate):
    - Only classify as PER-ELEMENT when per-element evidence is strong and reduction evidence is weak.
    - Treat many "partial reduction" workloads as REDUCTION (i.e., reductions are allowed as intermediate scalars),
      even if the final output is an array (e.g., result[i] per row).

    Unknown -> should NOT automatically trigger semantic-reduction repair (handled in detect()).
    """
    s = (spec_text or "")
    # function name often contains semantics (norm/dot/matmul/conv/etc.)
    if func_name:
        s = s + "\n" + str(func_name)
    s_low = s.lower()

    # Per-element signals (each occurrence adds evidence)
    per_signals = [
        "each element", "for each element", "per-element", "element-wise", "elementwise",
        "for every element", "same length",
        "out[i]", "dst[i]", "result[i]", "output[i]", "mask[i]",
        "store the result in out[i]", "store the result in dst[i]", "store the result in result[i]",
        "write to out[i]", "write to dst[i]", "write to result[i]",
    ]

    # Reduction-like signals (GLOBAL or PARTIAL reductions where sv*addv/etc can be legitimate)
    # We deliberately include partial-reduction workloads to avoid false positives.
    red_signals_strong = [
        "dot product", "inner product",
        "matrix multiply", "matrix multiplication", "matmul", "gemm",
        "convolution", "conv",
        "row-wise", "row wise", "per row", "each row", "across columns",
        "column-wise", "column wise", "per column", "each column", "across rows",
        "norm", "l2", "l1", "rms",
        "histogram",
        "reduce", "reduction",
        "sum of", "sum over", "accumulate", "accumulation",
        "mean", "average",
        "maximum value", "minimum value", "max value", "min value",
        "argmax", "argmin",
        "count total", "total number of", "population count", "popcount",
    ]

    per_score = 0
    red_score = 0

    for k in per_signals:
        if k in s_low:
            # "out[i]"-style evidence should not fully override strong reduction signals (partial reduction)
            if k in ("out[i]", "dst[i]", "result[i]", "output[i]", "mask[i]"):
                per_score += 2
            else:
                per_score += 1

    for k in red_signals_strong:
        if k in s_low:
            # weight a bit higher for workload-structural keywords
            if k in ("dot product", "inner product", "matrix multiply", "matrix multiplication", "matmul", "gemm",
                     "convolution", "conv", "row-wise", "row wise", "per row", "each row", "across columns",
                     "column-wise", "column wise", "per column", "each column", "across rows",
                     "norm", "histogram", "argmax", "argmin"):
                red_score += 3
            else:
                red_score += 2

    # Return-type hint (if available): scalar return type is strong reduction evidence;
    # pointer return type is weak evidence.
    rt = _parse_return_type_from_decl(CURRENT_TARGET_FUNC_DECL, func_name)
    rt_low = (rt or "").lower().replace(" ", "")
    if rt_low and (rt_low not in ("void", "void*", "voidconst")):
        if "*" in rt_low:
            red_score += 1
        else:
            red_score += 5

    dbg = {"per_score": per_score, "red_score": red_score, "return_type": rt}

    # Decide with conservative thresholds
    if red_score >= per_score + 2 and red_score >= 5:
        conf = min(1.0, 0.55 + 0.08 * (red_score - per_score))
        return TASK_MODE_REDUCTION, conf, dbg

    if per_score >= red_score + 2 and per_score >= 5:
        conf = min(1.0, 0.55 + 0.08 * (per_score - red_score))
        return TASK_MODE_PER_ELEMENT, conf, dbg

    return TASK_MODE_UNKNOWN, 0.45, dbg


def detect_reduction_semantic_issue(code: str, spec_text: str, func_name: Optional[str]) -> Tuple[bool, str, Dict]:
    """
    Detect whether code uses SVE horizontal reduction intrinsics (sv*addv/maxv/minv/orv/andv/eorv)
    in a way that is likely SEMANTICALLY WRONG for PER-ELEMENT tasks.

    Returns (is_issue, issue_kind, debug)
      issue_kind in {"hard", "soft", "none"}

    Safety principle (to avoid pass-rate regression):
    - NEVER trigger semantic reduction repair for TASK_MODE_UNKNOWN (too many partial-reduction tasks look "unknown").
    - Allow reductions for TASK_MODE_REDUCTION (includes partial reductions like per-row norms/dot-products/matmul).
    - Still force-trigger on compile-breaking patterns (scalar reduction assigned to SVE vector types).
    """
    gate = str(SEMANTIC_REDUCTION_GATE or "soft")
    if gate == "off":
        return False, "none", {}

    c = code or ""
    if not _SVE_REDUCTION_CALL_RE.search(c):
        return False, "none", {}

    # Compile-breaking misuse: assigning scalar reduction result to an SVE vector type (almost certainly wrong).
    # Example: svint32_t x = svaddv_s32(pg, v);
    if re.search(r"\bsv(?:u?int|float)\d+_t\s+\w+\s*=\s*sv(?:addv|maxv|minv|andv|orv|eorv)_[a-z0-9_]+\s*\(", c):
        dbg = {"reason": "scalar_reduction_assigned_to_vector_type", "gate": gate}
        return True, "hard", dbg

    mode, conf, dbg = _infer_task_mode_from_text(spec_text, func_name)
    dbg["task_mode"] = mode
    dbg["task_conf"] = conf
    dbg["gate"] = gate

    # Reduction-like tasks: reductions are allowed
    if mode == TASK_MODE_REDUCTION:
        return False, "none", dbg

    # Unknown tasks: DO NOT trigger semantic reduction repair (too risky, causes false positives)
    if mode == TASK_MODE_UNKNOWN:
        return False, "none", dbg

    # Per-element tasks:
    if gate == "hard":
        if conf >= 0.70:
            return True, "hard", dbg
        # still allow a softer attempt
        return True, "soft", dbg

    # soft gate: only trigger if we are at least moderately confident it's per-element
    if conf < 0.55:
        return False, "none", dbg

    return True, "soft", dbg


def build_reduction_semantic_repair_prompt(code: str, spec_text: str, dbg: Dict) -> str:
    # Narrowly scoped semantic repair prompt with an explicit "partial reductions are OK" rule.
    return (
        "/*\n"
        "[SEMANTIC_REPAIR_REDUCTION]\n"
        "- You are repairing SVE SIMD C/C++ code to match the SPEC.\n"
        "- The current code uses SVE horizontal reduction intrinsics (sv*addv/maxv/minv/orv/andv/eorv).\n"
        "\n"
        "Rules:\n"
        "1) PER-ELEMENT tasks (one output per input element):\n"
        "   - Horizontal reductions that collapse lanes are usually WRONG.\n"
        "   - Replace them with lane-wise vector ops, and store per-lane outputs using svst1_*.\n"
        "\n"
        "2) PARTIAL reductions (dot/norm/matmul inner sums, per-row/per-output-element scalars):\n"
        "   - Reductions are LEGITIMATE as intermediate scalars.\n"
        "   - If a reduction produces exactly ONE scalar per output element (e.g., result[i]), keep it.\n"
        "\n"
        "3) GLOBAL reductions (one scalar total):\n"
        "   - Reductions are appropriate; return/store exactly one scalar.\n"
        "\n"
        "Additional constraints:\n"
        "- Do NOT vectorize across a non-contiguous dimension unless you use gather/strided loads.\n"
        "- Keep predicate widths consistent (b32 with 32-bit lanes, b64 with 64-bit lanes).\n"
        "- If output is a boolean/byte mask array, store 0/1 bytes (svuint8_t) instead of trying to store svbool_t.\n"
        "- Output ONLY code (respect current completion_mode).\n"
        "[END_SEMANTIC_REPAIR_REDUCTION]\n"
        "*/\n"
        "\n[SPEC]\n" + (spec_text or "").strip() + "\n"
        "\n[INFER]\n" + json.dumps(dbg, ensure_ascii=False, indent=2) + "\n"
        "\n[COMPLETION_TO_FIX]\n" + (code or "").strip() + "\n"
        "\n[END]\n"
    )
@torch.no_grad()
def semantic_repair_reduction_loop(
    model,
    tok,
    *,
    completion_in: str,
    spec_text: str,
    func_name: Optional[str],
    max_iters: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    attempts_per_iter: int = 2,
    print_prompts: bool = False,
    prompt_max_chars: int = 0,
) -> Tuple[str, Dict]:
    """
    Only repairs reduction misuse (does not attempt general semantic fixes).
    Safe to enable by default with small max_iters (e.g., 1).
    """
    info: Dict = {"iters": 0, "triggered": False, "final_issue": "none"}
    code = completion_in

    for it in range(max(0, int(max_iters))):
        is_issue, kind, dbg = detect_reduction_semantic_issue(code, spec_text, func_name)
        info["final_issue"] = kind
        info["last_dbg"] = dbg

        if not is_issue:
            break

        info["triggered"] = True
        prompt = build_reduction_semantic_repair_prompt(code, spec_text, dbg)

        if print_prompts:
            print("\n" + "=" * 120)
            print(f"[SEMANTIC_REDUCTION_PROMPT] iter={it+1} kind={kind}")
            print("-" * 120)
            if prompt_max_chars and len(prompt) > prompt_max_chars:
                print(prompt[:prompt_max_chars] + "\n...[TRUNCATED]...")
            else:
                print(prompt)
            print("=" * 120 + "\n")

        best = None
        best_kind = kind

        tries = max(1, int(attempts_per_iter))
        for _ in range(tries):
            out = generate_text(
                model, tok,
                user_text=prompt,
                system_text="",
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            cand = normalize_completion_snippet(out, func_name)
            cand_issue, cand_kind, _ = detect_reduction_semantic_issue(cand, spec_text, func_name)

            # Prefer candidates that remove the issue entirely; else prefer soft over hard
            if not cand_issue:
                best = cand
                best_kind = "none"
                break
            if best is None and cand_kind == "soft" and kind == "hard":
                best = cand
                best_kind = "soft"

        if best is None:
            # no improvement
            break

        code = best
        info["iters"] += 1
        info["final_issue"] = best_kind
        if best_kind == "none":
            break

    return code, info

def load_whitelist(path: Path) -> Tuple[Set[str], List[str], Dict[str, List[List[str]]], Dict[str, List[str]]]:
    """
    Load whitelist.json.

    Supported formats (best-effort):
      - {"names":[...], "sigs": {"svfoo":[["svbool_t","svuint32_t",...], ...]}}
      - {"names":[...], "sigs": {"svfoo":[{"ret":"svuint32_t","args":[...]} , ...]}}
      - {"intrinsics": {"svfoo":[{"ret":...,"args":[...]} , ...]}}   (recommended)
      - legacy: top-level {"svfoo":[["..."], ...], ...}
    Returns:
      (whitelist_set, whitelist_list, sigs_args, rets)
    where:
      - sigs_args[name] = list of argument-type lists
      - rets[name] = list of return types (optional; may be empty)
    """
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))

    # 1) names list (optional)
    names_raw = obj.get("names", [])
    name_list: List[str] = [str(x) for x in names_raw] if isinstance(names_raw, list) else []

    # 2) signatures / intrinsics mapping
    sigs_raw = obj.get("sigs", None)
    if sigs_raw is None:
        sigs_raw = obj.get("signatures", None)
    if sigs_raw is None:
        sigs_raw = obj.get("intrinsics", None)

    # legacy: top-level { "svfoo": [...] }
    if sigs_raw is None:
        maybe: Dict[str, object] = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("sv") and isinstance(v, list):
                maybe[k] = v
        if maybe:
            sigs_raw = maybe

    sigs2: Dict[str, List[List[str]]] = {}
    rets2: Dict[str, List[str]] = {}

    def _consume_one_sig_list(name: str, v: object) -> None:
        if not isinstance(v, list):
            return
        out_args: List[List[str]] = []
        out_rets: List[str] = []
        for one in v:
            if isinstance(one, list):
                out_args.append([str(t) for t in one])
                continue
            if isinstance(one, dict):
                args = one.get("args", None)
                if args is None:
                    args = one.get("params", None)
                if args is None:
                    args = one.get("signature", None)
                if isinstance(args, list):
                    out_args.append([str(t) for t in args])
                    r = one.get("ret", None)
                    if r is None:
                        r = one.get("return", None)
                    if r is None:
                        r = one.get("returns", None)
                    if isinstance(r, str) and r.strip():
                        r2 = r.strip()
                        # Some clang/arm_sve.h overload shims surface placeholder return types like "__aio".
                        # These are not stable C types; using them for return-mismatch detection creates false positives.
                        if r2.startswith("__"):
                            continue
                        out_rets.append(r2)
                continue
        if out_args:
            sigs2[str(name)] = out_args
        if out_rets:
            rets2[str(name)] = out_rets

    if isinstance(sigs_raw, dict):
        for k, v in sigs_raw.items():
            _consume_one_sig_list(str(k), v)

    elif isinstance(sigs_raw, list):
        # list of {"name":..., "args":[...], "ret":...}
        for one in sigs_raw:
            if not isinstance(one, dict):
                continue
            nm = one.get("name", None)
            if not isinstance(nm, str) or not nm.startswith("sv"):
                continue
            _consume_one_sig_list(nm, [one])

    if not name_list:
        name_list = sorted(sigs2.keys())
    else:
        # Ensure names include anything present in sigs2
        for k in sigs2.keys():
            if k not in name_list:
                name_list.append(k)

    return set(name_list), name_list, sigs2, rets2


def augment_sve_whitelist(
    whitelist_set: Set[str],
    whitelist_list: List[str],
    sigs: Dict[str, List[List[str]]],
    rets: Dict[str, List[str]],
) -> Tuple[Set[str], List[str], Dict[str, List[List[str]]], Dict[str, List[str]]]:
    """Augment a user-provided whitelist with a small set of core SVE intrinsics.

    Some datasets require a handful of fundamental SVE intrinsics (e.g., svwhilelt_* for tail
    predication, load-and-extend forms for byte masks) that might be missing from a generated
    whitelist.json. Adding them here improves repair robustness without requiring the user to
    manually regenerate the whitelist.
    """
    extras: Dict[str, Tuple[str, List[List[str]]]] = {
        # Tail predicates
        "svwhilelt_b8": ("svbool_t", [["uint64_t", "uint64_t"]]),
        "svwhilelt_b16": ("svbool_t", [["uint64_t", "uint64_t"]]),
        "svwhilelt_b32": ("svbool_t", [["uint64_t", "uint64_t"]]),
        "svwhilelt_b64": ("svbool_t", [["uint64_t", "uint64_t"]]),
        # All-true predicates
        "svptrue_b8": ("svbool_t", [[]]),
        "svptrue_b16": ("svbool_t", [[]]),
        "svptrue_b32": ("svbool_t", [[]]),
        "svptrue_b64": ("svbool_t", [[]]),
        # Vector-length queries
        "svcntb": ("size_t", [[]]),
        "svcnth": ("size_t", [[]]),
        "svcntw": ("size_t", [[]]),
        "svcntd": ("size_t", [[]]),
        # Common typed load/store
        "svld1_s64": ("svint64_t", [["svbool_t", "const int64_t *"]]),
        "svst1_s64": ("void", [["svbool_t", "int64_t *", "svint64_t"]]),
        "svld1_u64": ("svuint64_t", [["svbool_t", "const uint64_t *"]]),
        # Load byte mask and extend to 64-bit lanes (useful for bool/uint8 masks controlling 64-bit data)
        "svld1ub_u64": ("svuint64_t", [["svbool_t", "const uint8_t *"]]),
        "svld1sb_s64": ("svint64_t", [["svbool_t", "const int8_t *"]]),
        # Common comparisons that produce predicates
        "svcmpne_n_u64": ("svbool_t", [["svbool_t", "svuint64_t", "uint64_t"]]),
        "svcmpne_n_s64": ("svbool_t", [["svbool_t", "svint64_t", "int64_t"]]),
        "svcmpeq_n_u64": ("svbool_t", [["svbool_t", "svuint64_t", "uint64_t"]]),
        "svcmpeq_n_s64": ("svbool_t", [["svbool_t", "svint64_t", "int64_t"]]),
    }

    for name, (ret, arglists) in extras.items():
        if name not in whitelist_set:
            whitelist_set.add(name)
            whitelist_list.append(name)

        sigs.setdefault(name, [])
        for al in arglists:
            if al not in sigs[name]:
                sigs[name].append(al)

        rets.setdefault(name, [])
        if ret not in rets[name]:
            rets[name].append(ret)

    return whitelist_set, whitelist_list, sigs, rets


def build_op_index(whitelist_list: List[str]) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}
    for n in whitelist_list:
        op = n.split("_")[0]
        idx.setdefault(op, []).append(n)
    return idx

def suggest_for_name(
    bad: str,
    whitelist_set: Set[str],
    whitelist_list: List[str],
    op_index: Dict[str, List[str]],
    code: str,
    top_k: int,
    cutoff: float,
) -> List[str]:
    """Suggest valid intrinsic names for a hallucinated/invalid call name.

    Root-cause policy:
    - Only suggest when we are confident it's a typo/variant of a real intrinsic.
    - For unknown/unsupported ops (e.g., modulus, "between", sort), return [] so the repair step
      is forced to rewrite (often scalar fallback) instead of guessing a different intrinsic.
    - Avoid cross-op fuzzy matches that produce semantically wrong code.
    """
    lane = infer_lane_width(code)
    code_type_tokens = guess_type_tokens(code)

    bad_orig = bad or ""

    # Interpret common prefix mistakes:
    # - missing "sv" prefix: add it
    # - accidental leading "v" (e.g., vdiv_u64): drop the first 'v' and add "sv"
    if bad_orig.startswith("sv"):
        bw = bad_orig
    elif bad_orig.startswith("v") and len(bad_orig) > 1:
        bw = "sv" + bad_orig[1:]
    else:
        bw = "sv" + bad_orig

    # Type tokens explicitly present in the invalid name itself are the strongest hint.
    bad_type_tokens = sorted(
        set(
            re.findall(
                r"_(u8|u16|u32|u64|s8|s16|s32|s64|f16|f32|f64|b8|b16|b32|b64)(?:_|$)",
                bw,
            )
        )
    )

    sugs: List[str] = []

    # 0) Missing prefix exact fix (includes the accidental 'v' prefix normalization above)
    if bad_orig and (not bad_orig.startswith("sv")) and (bw in whitelist_set):
        sugs.append(bw)

    # 1) Common fix heuristics: drop/replace suffixes like _x/_z, or remove an injected '_z_'.
    if bw.endswith("_x") and bw[:-2] in whitelist_set:
        sugs.append(bw[:-2])
    if bw.endswith("_z") and bw[:-2] in whitelist_set:
        sugs.append(bw[:-2])
    if "_z_" in bw:
        cand = bw.replace("_z_", "_")
        if cand in whitelist_set:
            sugs.append(cand)

    # 2) If the base (without predication suffix) exists, prefer it.
    base2 = re.sub(r"_(x|z|m)$", "", bw)
    if base2 != bw and base2 in whitelist_set:
        sugs.append(base2)

    # 3) Prefer _z variant if it exists.
    if (bw + "_z") in whitelist_set:
        sugs.append(bw + "_z")

    # 4) Same-op-group suggestions (most reliable)
    op = bw.split("_")[0]
    group = op_index.get(op, [])
    group_pref = [n for n in group if n.endswith("_z")] or group

    filtered: List[str] = []

    # 4.1) Enforce types from the bad name if present.
    if bad_type_tokens:
        for t in bad_type_tokens:
            filtered += [n for n in group_pref if t in n]

    # 4.2) Otherwise fall back to types seen in the code.
    if not filtered:
        for t in code_type_tokens:
            filtered += [n for n in group_pref if t in n]

    # 4.3) If the bad name had explicit types but nothing matches, don't guess an untyped op.
    if not filtered and (not bad_type_tokens):
        filtered = group_pref[:]

    # Light lane preference (best-effort only).
    if filtered and lane in ("b32", "b64", "b16", "b8"):
        filtered2 = [n for n in filtered if f"_{lane}" in n]
        if filtered2:
            filtered = filtered2

    for n in filtered:
        if n in whitelist_set:
            sugs.append(n)
        if len(sugs) >= top_k:
            break

    # 5) High-confidence global fuzzy match ONLY as last resort (typos).
    if len(sugs) < top_k:
        hi_cut = max(0.92, float(cutoff))

        def _base_op(x: str) -> str:
            op0 = x.split("_")[0]
            return re.sub(r"\d+$", "", op0)

        bad_op = _base_op(bw)
        pool = difflib.get_close_matches(bw, whitelist_list, n=max(12, top_k * 4), cutoff=hi_cut)

        for cand in pool:
            if cand not in whitelist_set:
                continue
            if _base_op(cand) != bad_op:
                continue
            if bad_type_tokens and (not any(t in cand for t in bad_type_tokens)):
                continue
            sugs.append(cand)
            if len(sugs) >= top_k:
                break

    # 6) Deduplicate while preserving order.
    out: List[str] = []
    seen: Set[str] = set()
    for x in sugs:
        if x in whitelist_set and x not in seen:
            out.append(x)
            seen.add(x)
        if len(out) >= top_k:
            break
    return out


def build_name_repair_prompt(code: str, invalid: List[str], suggestions: Dict[str, List[str]], spec_text: str = "") -> str:
    """Build the name-fix prompt.

    IMPORTANT: We do NOT print "candidates: (none found)" because it is ambiguous and can
    encourage the model to "guess" a replacement sv* name. Instead, we explicitly label
    such cases as UNSUPPORTED and require a rewrite / scalar fallback.

    This is a *root-cause* fix: unknown/hallucinated intrinsic names are not typos and cannot
    be solved by name substitution.
    """
    lines: List[str] = []
    rules = STRICT_NAME_RULES_FULL if COMPLETION_MODE == COMPLETION_MODE_FULL else STRICT_NAME_RULES_SNIPPET
    lines.append(rules)
    if spec_text:
        lines.append("\n[SPEC]\n" + spec_text.strip() + "\n")

    with_cand = [b for b in invalid if suggestions.get(b)]
    no_cand = [b for b in invalid if not suggestions.get(b)]

    if with_cand:
        lines.append("\n[INVALID_INTRINSICS_WITH_CANDIDATES]")
        for b in with_cand:
            cand = suggestions.get(b, [])
            lines.append(f"- {b}")
            lines.append(f"  candidates: {', '.join(cand)}")

    if no_cand:
        lines.append("\n[UNSUPPORTED_INTRINSICS_MUST_REMOVE]")
        for b in no_cand:
            lines.append(f"- {b}")
        lines.append(
            "Rule: The above names are NOT valid SVE ACLE intrinsics. Do NOT guess a replacement sv* name.\n"
            "You MUST eliminate them by rewriting the code using VALID SVE intrinsics and/or scalar C/C++ loops.\n"
            "If an operation has no direct SVE intrinsic, scalar fallback is allowed.\n"
            "Hard requirement: your output must not contain ANY new unknown sv* calls."
        )

    lines.append("\n[COMPLETION_TO_FIX]\n" + code.strip() + "\n")
    lines.append("[END]")
    return "\n".join(lines)


@torch.no_grad()
def hard_rewrite_remove_invalid_intrinsics(
    model,
    tok,
    *,
    completion_in: str,
    whitelist_set: Set[str],
    whitelist_list: List[str],
    op_index: Dict[str, List[str]],
    spec_text: str,
    max_rounds: int = 2,
    attempts_per_round: int = 2,
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    sid: str = "",
    rank: int = 0,
    func_name: Optional[str] = None,
    print_prompts: bool = False,
) -> Tuple[str, Dict]:
    """Hard, root-cause cleanup: eliminate ANY invalid sv* calls.

    This is only invoked when the normal name-fix loop cannot reach a fully-valid intrinsic set.
    It is intentionally *strict* and *monotonic*: outputs that introduce NEW invalid sv* tokens
    are rejected.

    Key behavior (what makes it "not just a hint"):
    - We loop until invalid intrinsics are eliminated (or max_rounds exhausted).
    - If the model keeps hallucinating new sv* names, we reject them and retry.
    - Scalar fallback is explicitly allowed for operations that have no SVE intrinsic.
    """
    info: Dict = {"rounds": 0, "invalid_before": [], "invalid_after": [], "skipped_new_invalid": 0}
    code = normalize_completion_snippet(completion_in, func_name)

    def _invalid_list(c: str) -> List[str]:
        return sorted([x for x in extract_calls(c, whitelist_set, context_text=spec_text) if x not in whitelist_set])

    cur_invalid = _invalid_list(code)
    info["invalid_before"] = cur_invalid

    for r in range(max_rounds):
        cur_invalid = _invalid_list(code)
        if not cur_invalid:
            break

        suggestions = {
            b: suggest_for_name(b, whitelist_set, whitelist_list, op_index, code, top_k=8, cutoff=0.55)
            for b in cur_invalid
        }
        prompt = build_name_repair_prompt(code, cur_invalid, suggestions, spec_text=spec_text)
        prompt += (
            "\n[STRICT_ENFORCEMENT]\n"
            "- You MUST eliminate ALL items listed above.\n"
            "- You MUST NOT introduce any new unknown sv* calls.\n"
            "- If an operation has no SVE intrinsic, implement it using scalar C/C++ loops.\n"
            "[END_STRICT_ENFORCEMENT]\n"
        )

        if print_prompts:
            print("\n" + "=" * 120)
            print(f"[HARD_NAME_PROMPT] rank={rank} id={sid} round={r+1} invalid={len(cur_invalid)}")
            print("-" * 120)
            print(prompt)
            print("=" * 120 + "\n")

        best = None
        best_inv = cur_invalid
        best_forb: List[str] = []
        best_score = (len(best_forb) * 1000000) + (len(best_inv) * 1000)

        tries = max(1, int(attempts_per_round))
        for aidx in range(tries):
            out = generate_text(
                model,
                tok,
                user_text=prompt,
                system_text="",
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            cand = normalize_completion_snippet(out, func_name)
            forb = detect_forbidden_structures(cand, func_name)
            inv = _invalid_list(cand)

            new_invalid = [x for x in inv if x not in cur_invalid]
            if new_invalid:
                info["skipped_new_invalid"] += 1
                continue

            # Reject obvious placeholders / truncation markers.
            if ("..." in cand) or ("[TRUNCATED]" in cand) or ("TODO" in cand):
                info.setdefault("skipped_placeholder", 0)
                info["skipped_placeholder"] += 1
                continue

            # Locality preference: even in hard-rewrite, prefer changing near the invalid call sites.
            anchor_lines = _find_call_line_indices(code, cur_invalid)
            nonlocal_segs = 0
            if anchor_lines:
                nonlocal_segs = _count_nonlocal_edit_segments(code, cand, anchor_lines, window=40)

            diff_local = _changed_line_count(code, cand)

            # Avoid non-local rewrites that don't reduce invalid intrinsics.
            if nonlocal_segs > 0 and len(inv) >= len(cur_invalid) and diff_local > 40:
                info.setdefault("skipped_nonlocal_edit", 0)
                info["skipped_nonlocal_edit"] += 1
                continue

            if diff_local > 250 and len(inv) >= len(cur_invalid):
                # Avoid massive algorithm rewrites that don't even reduce invalid intrinsics.
                info["skipped_new_invalid"] += 0  # keep key present
                continue

            score = (len(forb) * 1000000) + (len(inv) * 1000) + diff_local + (nonlocal_segs * 100)
            if score < best_score:
                best_score = score
                best = cand
                best_inv = inv
                best_forb = forb

        if best is None:
            # No monotonic candidate found; stop (we keep current code).
            break

        code = best
        info["rounds"] = r + 1

        # If solved, stop early.
        if (not best_inv) and (not best_forb):
            break

    info["invalid_after"] = _invalid_list(code)
    return code, info


@torch.no_grad()
def rag_fix_names(
    model,
    tok,
    *,
    completion_in: str,
    whitelist_set: Set[str],
    whitelist_list: List[str],
    op_index: Dict[str, List[str]],
    spec_text: str,
    max_iters: int,
    max_new_tokens: int,
    top_k: int,
    cutoff: float,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    sid: str = "",
    rank: int = 0,
    func_name: Optional[str] = None,
    attempts_per_iter: int = 3,
    print_prompts: bool = False,
    dump_prompts_dir: Optional[Path] = None,
    prompt_max_chars: int = 0,
) -> Tuple[str, Dict]:
    """
    Name-level intrinsic repair with monotonic acceptance:
      - Reject candidates that introduce forbidden structures (#include, function re-def, wrong brace delta).
      - Keep the best (lowest-error) completion across iterations.
    """
    info: Dict = {}
    code = normalize_completion_snippet(completion_in, func_name)
    origin_code = code  # for semantic-stability diff penalty (tie-breaker)
    MAX_NAMEFIX_CHANGED_LINES = 80  # hard reject huge rewrites that do not reduce invalids

    def _invalid_list(c: str) -> List[str]:
        # Pass whitelist_set so we can also detect non-"sv" intrinsic-like calls (e.g. "shr_n_s32_z").
        # Pass spec_text as context in snippet mode so we don't mis-classify prompt-prefix declarations as calls.
        return sorted([x for x in extract_calls(c, whitelist_set, context_text=spec_text) if x not in whitelist_set])

    def _score(c: str) -> Tuple[int, List[str], List[str]]:
        invalid = _invalid_list(c)
        forbidden = detect_forbidden_structures(c, func_name)
        # forbidden is "hard" (very expensive)
        score = (len(forbidden) * 1000000) + (len(invalid) * 1000)
        return score, invalid, forbidden

    score0, invalid0, forbid0 = _score(code)
    best_code = code
    best_score = score0

    info["invalid_before"] = invalid0
    info["invalid_before_count"] = len(invalid0)
    info["forbidden_before"] = forbid0

    it = 0
    while it < max_iters:
        cur_score, cur_invalid, cur_forbid = _score(code)
        if (not cur_invalid) and (not cur_forbid):
            break

        if not cur_invalid:
            # nothing to do at name-level; stop
            break

        suggestions = {
            b: suggest_for_name(b, whitelist_set, whitelist_list, op_index, code, top_k, cutoff)
            for b in cur_invalid
        }
        prompt = build_name_repair_prompt(code, cur_invalid, suggestions, spec_text=spec_text)

        iter_no = it + 1

        if print_prompts:
            print("\n" + "=" * 120)
            print(f"[NAME_PROMPT] rank={rank} id={sid} iter={iter_no} invalid={len(cur_invalid)}")
            print("-" * 120)
            if prompt_max_chars and prompt_max_chars > 0 and len(prompt) > prompt_max_chars:
                print(prompt[:prompt_max_chars] + "\n...[TRUNCATED]...")
            else:
                print(prompt)
            print("=" * 120 + "\n")

        if dump_prompts_dir is not None:
            dump_prompts_dir.mkdir(parents=True, exist_ok=True)
            (dump_prompts_dir / f"name_prompt_iter{iter_no}.txt").write_text(prompt, encoding="utf-8")
            (dump_prompts_dir / f"name_invalid_iter{iter_no}.json").write_text(
                json.dumps(
                    {
                        "sid": sid,
                        "rank": rank,
                        "iter": iter_no,
                        "invalid": cur_invalid,
                        "suggestions": suggestions,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        # Try multiple candidates and pick the best monotonic improvement.
        # Locality gate (semantic stability):
        # For pure *name-level* fixes (typos/variants), prefer candidates that only edit near the
        # invalid call sites. This prevents the model from "fixing" a name by rewriting unrelated
        # algorithmic logic.
        anchor_lines = _find_call_line_indices(code, cur_invalid)
        unsupported = [b for b, s in suggestions.items() if not s]
        enforce_locality = bool(anchor_lines) and (len(cur_invalid) <= 3) and (not unsupported)
        local_window = 12

        chosen = None
        chosen_score = cur_score
        chosen_invalid = cur_invalid
        chosen_forbid = cur_forbid

        # Fallback when the model refuses to stay local: keep the best non-local candidate,
        # but only use it if no local candidate improved the score.
        best_nonlocal = None
        best_nonlocal_score = cur_score
        best_nonlocal_invalid = cur_invalid
        best_nonlocal_forbid = cur_forbid

        tries = max(1, int(attempts_per_iter))
        for aidx in range(tries):
            out = generate_text(
                model,
                tok,
                user_text=prompt,
                system_text="",
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            cand = normalize_completion_snippet(out, func_name)
            sc, inv, forb = _score(cand)
            diff_local = _changed_line_count(code, cand)
            diff_origin = _changed_line_count(origin_code, cand)
            # Prefer smaller edits for semantic stability.
            sc_local = sc + diff_local
            sc_global = sc + diff_origin

            # Hard reject: huge rewrite that doesn't even reduce invalid count.
            if diff_local > MAX_NAMEFIX_CHANGED_LINES and len(inv) >= len(cur_invalid):
                info.setdefault("skipped_huge_rewrite", 0)
                info["skipped_huge_rewrite"] += 1
                continue

            # Root-cause enforcement: do NOT accept candidates that introduce NEW invalid sv* names.
            # This prevents "A -> B" drift where B is also invalid (endless hallucination loop).
            new_invalid = [x for x in inv if x not in cur_invalid]
            if new_invalid:
                info.setdefault("skipped_new_invalid", 0)
                info["skipped_new_invalid"] += 1
                continue

            # Reject obvious placeholders / truncation markers.
            if ("..." in cand) or ("[TRUNCATED]" in cand) or ("TODO" in cand):
                info.setdefault("skipped_placeholder", 0)
                info["skipped_placeholder"] += 1
                continue

            # Keep global best (even if it doesn't improve current step).
            if sc_global < best_score:
                best_score = sc_global
                best_code = cand

            nonlocal_segs = 0
            if enforce_locality:
                nonlocal_segs = _count_nonlocal_edit_segments(code, cand, anchor_lines, local_window)

            if enforce_locality and nonlocal_segs > 0:
                info.setdefault("skipped_nonlocal_edit", 0)
                info["skipped_nonlocal_edit"] += 1
                if sc_local < best_nonlocal_score:
                    best_nonlocal = cand
                    best_nonlocal_score = sc_local
                    best_nonlocal_invalid = inv
                    best_nonlocal_forbid = forb
            else:
                if sc_local < chosen_score:
                    chosen = cand
                    chosen_score = sc_local
                    chosen_invalid = inv
                    chosen_forbid = forb

        if chosen is None and best_nonlocal is not None:
            chosen = best_nonlocal
            chosen_score = best_nonlocal_score
            chosen_invalid = best_nonlocal_invalid
            chosen_forbid = best_nonlocal_forbid

        if chosen is None:
            # no improvement found; stop early
            break

        code = chosen
        it += 1

        # If solved, stop.
        if (not chosen_invalid) and (not chosen_forbid):
            break

    # Final: pick best we saw (monotonic over time).
    code_final = best_code

    invalid_after = _invalid_list(code_final)
    forbid_after = detect_forbidden_structures(code_final, func_name)

    info["iters"] = it
    info["invalid_after"] = invalid_after
    info["invalid_after_count"] = len(invalid_after)
    info["forbidden_after"] = forbid_after

    # Root-cause cleanup: if any invalid sv* calls remain, force a rewrite that eliminates them.
    # This is NOT a "hint". We actively reject outputs that keep/introduce invalid intrinsic names.
    if invalid_after:
        code_hard, hard_info = hard_rewrite_remove_invalid_intrinsics(
            model=model,
            tok=tok,
            completion_in=code_final,
            whitelist_set=whitelist_set,
            whitelist_list=whitelist_list,
            op_index=op_index,
            spec_text=spec_text,
            max_rounds=2,
            attempts_per_round=max(2, int(attempts_per_iter)),
            max_new_tokens=max(1024, int(max_new_tokens)),
            do_sample=True,
            temperature=max(0.6, float(temperature)),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            sid=sid,
            rank=rank,
            func_name=func_name,
            print_prompts=print_prompts,
        )
        info["hard_rewrite"] = hard_info
        code_final = code_hard
        invalid_after = sorted([x for x in extract_calls(code_final, whitelist_set, context_text=spec_text) if x not in whitelist_set])
        forbid_after = detect_forbidden_structures(code_final, func_name)
        info["invalid_after"] = invalid_after
        info["invalid_after_count"] = len(invalid_after)
        info["forbidden_after"] = forbid_after

    return code_final, info


# -------------------------
# Signature / call-shape repair helpers
# -------------------------

def build_arity_index(sigs: Dict[str, List[List[str]]]) -> Dict[int, List[str]]:
    idx: Dict[int, List[str]] = {}
    for name, sig_list in sigs.items():
        for sig in sig_list:
            idx.setdefault(len(sig), []).append(name)
    return idx


# -------------------------
# Type inference helpers (decls + function parameters)
# -------------------------

_DECL_QUALIFIERS_RE = re.compile(
    r"\b(?:const|volatile|restrict|__restrict__|__restrict|register|static|inline|constexpr)\b"
)


def _normalize_type_token(t: str) -> str:
    """Normalize a C/C++ type snippet for our coarse categorizer.

    We intentionally keep '*' (pointer) and the 'sv..._t' vector types intact,
    but strip common qualifiers and all spaces.
    """
    if not t:
        return ""
    s = str(t)
    s = _DECL_QUALIFIERS_RE.sub("", s)
    s = re.sub(r"\s+", "", s)
    # References: treat like the underlying type for our category checks.
    s = s.replace("&", "")
    return s


def _find_matching_paren(src: str, lparen_pos: int) -> int:
    """Return index of matching ')' for src[lparen_pos]=='(', or -1."""
    if lparen_pos < 0 or lparen_pos >= len(src) or src[lparen_pos] != "(":
        return -1
    depth = 0
    in_str = False
    str_ch = ""
    i = lparen_pos
    while i < len(src):
        ch = src[i]
        if in_str:
            if ch == str_ch and (i == 0 or src[i - 1] != "\\"):
                in_str = False
            i += 1
            continue
        if ch in ("'", '"'):
            in_str = True
            str_ch = ch
            i += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _parse_one_param_decl(param: str) -> Optional[Tuple[str, str]]:
    """Parse a single function parameter declaration into (name, normalized_type).

    This is best-effort. We accept the common forms used in SimdBench prompts:
      - const float *a
      - int64_t n
      - svbool_t pg
    Unnamed params (rare) are ignored.
    """
    if not param:
        return None
    p = str(param).strip()
    if not p or p == "void" or p == "...":
        return None

    # Drop default values (very rare in this dataset, but safe).
    if "=" in p:
        p = p.split("=", 1)[0].strip()

    # Drop C++11 attributes [[...]] (best-effort, non-nested).
    p = re.sub(r"\[\[[^\]]*\]\]", "", p).strip()

    # Identify the parameter name (last identifier token), allowing trailing array brackets.
    m = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)*$", p)
    if not m:
        return None
    name = m.group(1)
    type_part = p[:m.start(1)].strip()
    if not type_part:
        return None

    # If it's an array parameter (e.g., "int a[]"), treat as pointer.
    if re.search(r"\[[^\]]*\]\s*$", p[m.end(1):] or ""):
        if "*" not in type_part:
            type_part = type_part + "*"

    t_norm = _normalize_type_token(type_part)
    if not t_norm:
        return None

    # Sanity filter: avoid capturing expressions from call sites.
    # We only keep things that look like a type.
    if not (
        t_norm.startswith("sv")
        or "*" in t_norm
        or t_norm in ("int", "unsigned", "size_t", "ptrdiff_t", "float", "double", "bool")
        or re.match(r"^(?:u?int(?:8|16|32|64)_t|(?:u?int|u?long|u?longlong|longlong)|(?:u?intptr|u?intmax)_t)$", t_norm)
    ):
        return None

    return name, t_norm


def _extract_param_types_for_function(text: str, func_name: str) -> Dict[str, str]:
    """Extract (param_name -> normalized_type) for the target function from a text blob.

    Works on either:
      - Full translation unit (function definition present)
      - Prompt prefix (function signature present)

    This is best-effort and intentionally conservative.
    """
    if not text or not func_name:
        return {}
    s = strip_preprocessor_lines(strip_comments(text))

    # Prefer matches that look like a function declaration/definition line. This avoids
    # mistakenly treating a call site such as "return foo(x);" as the function signature.
    decl_hits: List[re.Match] = []
    for dm in _TOPLEVEL_FUNC_DECL_RE.finditer(s):
        if dm.group(1) != func_name:
            continue
        head = s[dm.start():dm.end()]
        if re.match(r"(?m)^\s*return\b", head):
            continue
        decl_hits.append(dm)

    lpar = -1
    if decl_hits:
        dm = decl_hits[-1]
        name_pos = s.find(func_name, dm.start(), dm.end())
        if name_pos < 0:
            name_pos = dm.start()
        lpar = s.find("(", name_pos + len(func_name), dm.end())
        if lpar < 0:
            lpar = s.find("(", dm.start(), dm.end())
    else:
        # Fallback: last raw occurrence of "func_name(".
        hits = list(re.finditer(rf"\b{re.escape(func_name)}\s*\(", s))
        if not hits:
            return {}
        m = hits[-1]
        lpar = s.find("(", m.start())

    if lpar < 0:
        return {}
    rpar = _find_matching_paren(s, lpar)
    if rpar < 0:
        return {}

    inside = s[lpar + 1 : rpar]
    params = split_args(inside)
    out: Dict[str, str] = {}
    for p in params:
        one = _parse_one_param_decl(p)
        if one is None:
            continue
        nm, ty = one
        if nm and ty:
            out[nm] = ty
    return out

def parse_decl_table(code: str, *, context_text: str = "", func_name: Optional[str] = None) -> Dict[str, str]:
    """Best-effort variable -> type table.

    The original implementation only recognized local declarations, which misses
    *function parameters* (e.g., `int n`, `const float *a`) and hurts shape-fix accuracy.

    - In FULL mode, we can often recover parameter types from the emitted function definition.
    - In SNIPPET mode, the completion does not contain the signature; we fall back to
      `context_text` (usually spec_text which includes the prompt prefix).
    """
    code2 = strip_comments(code)

    table: Dict[str, str] = {}

    # 1) Seed with target function parameters (if we can infer them).
    if func_name:
        if context_text:
            table.update(_extract_param_types_for_function(context_text, func_name))
        table.update(_extract_param_types_for_function(code2, func_name))

    # 2) Local declarations / simple assignments.
    for m in DECL_RE.finditer(code2):
        t = _normalize_type_token(m.group(1).strip())
        v = m.group(2).strip()
        if v:
            table[v] = t

    return table

def arg_category(arg: str, decls: Dict[str, str]) -> str:
    a = arg.strip()
    if not a:
        return "unknown"

    if re.match(r"^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?[fF]?$", a) or a in ("0", "1"):
        return "scalar"
    # Treat common constructors as vector-typed expressions
    if re.match(r"^svdup(_n)?_[A-Za-z0-9_]+\s*\(", a):
        return "vec"
    if re.match(r"^svreinterpret_[A-Za-z0-9_]+\s*\(", a):
        return "vec"   # best-effort; good enough for shape checking
    if a.startswith("&"):
        return "ptr"
    if "*" in a or "->" in a:
        return "ptr"

    if a.startswith("(") and ")" in a:
        t = a[1:a.find(")")]
        if "svbool_t" in t:
            return "pred"
        if "sv" in t and "_t" in t:
            return "vec"
        if "*" in t:
            return "ptr"
        return "scalar"

    name = re.split(r"[\[\].]", a)[0]
    if name in decls:
        t = decls[name]
        if t == "svbool_t":
            return "pred"
        if t.startswith("sv") and t.endswith("_t") and t != "svbool_t":
            return "vec"
        if "*" in t:
            return "ptr"
        if (
            t in (
                "int",
                "unsigned",
                "size_t",
                "ptrdiff_t",
                "long",
                "longlong",
                "float",
                "double",
                "bool",
                "uint8_t",
                "uint16_t",
                "uint32_t",
                "uint64_t",
                "int8_t",
                "int16_t",
                "int32_t",
                "int64_t",
            )
            or (t.endswith("_t") and (not t.startswith("sv")))
        ):
            return "scalar"

    if a.startswith("svptrue") or a.startswith("svwhilelt") or a.startswith("svptest") or a.startswith("svcmp"):
        return "pred"

    return "unknown"

def sig_category(t: str) -> str:
    tt = t.strip()
    if tt == "svbool_t":
        return "pred"
    if tt.startswith("sv") and tt.endswith("_t") and tt != "svbool_t":
        return "vec"
    if "*" in tt:
        return "ptr"
    return "scalar"

def split_args(arg_str: str) -> List[str]:
    s = arg_str.strip()
    if not s:
        return []
    out: List[str] = []
    cur: List[str] = []
    depth_par = 0
    depth_brk = 0
    depth_brc = 0
    in_str = False
    str_ch = ""
    i = 0
    while i < len(s):
        ch = s[i]
        if in_str:
            cur.append(ch)
            if ch == str_ch and (i == 0 or s[i - 1] != "\\"):
                in_str = False
            i += 1
            continue
        if ch in ("'", '"'):
            in_str = True
            str_ch = ch
            cur.append(ch)
            i += 1
            continue

        if ch == "(":
            depth_par += 1
        elif ch == ")":
            depth_par = max(0, depth_par - 1)
        elif ch == "[":
            depth_brk += 1
        elif ch == "]":
            depth_brk = max(0, depth_brk - 1)
        elif ch == "{":
            depth_brc += 1
        elif ch == "}":
            depth_brc = max(0, depth_brc - 1)

        if ch == "," and depth_par == 0 and depth_brk == 0 and depth_brc == 0:
            out.append("".join(cur).strip())
            cur = []
            i += 1
            continue

        cur.append(ch)
        i += 1

    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out

def find_call_sites(code: str) -> List[Tuple[str, str]]:
    code2 = strip_comments(code)
    defs = set(DEF_RE.findall(code2))

    sites: List[Tuple[str, str]] = []
    # IMPORTANT: capture nested call sites too.
    # A previous implementation advanced the scan index to the end of each call expression, which
    # skipped intrinsics used as arguments to other intrinsics (a common pattern in SVE codegen).
    for m in CALL_RE.finditer(code2):
        name = m.group(1)
        if name in defs:
            continue
        lpar = m.end() - 1
        if lpar < 0 or lpar >= len(code2) or code2[lpar] != "(":
            continue
        rpar = _find_matching_paren(code2, lpar)
        if rpar < 0:
            continue
        arg_str = code2[m.end() : rpar]
        sites.append((name, arg_str))

    return sites

def type_category_from_type_str(t: str) -> str:
    tt = (t or "").strip()
    if not tt:
        return "unknown"
    # normalize spaces
    tt = re.sub(r"\s+", "", tt)
    if "auto" in tt or "decltype" in tt:
        return "unknown"
    return sig_category(tt)

_DECL_INIT_CALL_RE = re.compile(
    r"(?m)^\s*(?:const\s+)?"
    r"([A-Za-z_]\w*(?:\s*<[^;]+?>)?(?:\s*[*&]\s*)*)\s+"
    r"([A-Za-z_]\w*)\s*=\s*"
    r"(sv[a-zA-Z0-9_]+)\s*\("
)

_ASSIGN_CALL_RE = re.compile(
    r"(?m)^\s*([A-Za-z_]\w*)\s*=\s*(sv[a-zA-Z0-9_]+)\s*\("
)

def detect_return_mismatches(
    code: str,
    whitelist_set: Set[str],
    rets: Dict[str, List[str]],
    *,
    context_text: str = "",
    func_name: Optional[str] = None,
) -> List[Dict]:
    """
    Detect obvious return-type category mismatches such as:
        svbool_t pg = svindex_u32(...);   // vec -> pred
    We only do a coarse category check: pred/vec/ptr/scalar.
    """
    if not rets:
        return []

    code2 = strip_comments(code)
    decls = parse_decl_table(code2, context_text=context_text, func_name=func_name)
    mism: List[Dict] = []

    def _ret_cats(name: str) -> Set[str]:
        rs = rets.get(name, [])
        out: Set[str] = set()
        for r in rs:
            out.add(type_category_from_type_str(r))
        # drop unknown
        out.discard("unknown")
        return out

    # 1) Declarations with initialization: <type> <var> = svfoo(...)
    for m in _DECL_INIT_CALL_RE.finditer(code2):
        lhs_type = m.group(1).strip()
        lhs_var = m.group(2).strip()
        name = m.group(3).strip()
        if name not in whitelist_set:
            continue
        if name not in rets:
            continue
        lhs_cat = type_category_from_type_str(lhs_type)
        if lhs_cat == "unknown":
            continue
        rcats = _ret_cats(name)
        if rcats and (lhs_cat not in rcats):
            mism.append({
                "name": name,
                "reason": "return_mismatch",
                "lhs_var": lhs_var,
                "lhs_type": lhs_type,
                "lhs_cat": lhs_cat,
                "ret_types": rets.get(name, [])[:8],
                "ret_cats": sorted(list(rcats))[:8],
            })

    # 2) Assignments to existing variable: <var> = svfoo(...)
    for m in _ASSIGN_CALL_RE.finditer(code2):
        lhs_var = m.group(1).strip()
        name = m.group(2).strip()
        if name not in whitelist_set:
            continue
        if name not in rets:
            continue
        lhs_type = decls.get(lhs_var, "")
        lhs_cat = type_category_from_type_str(lhs_type)
        if lhs_cat == "unknown":
            continue
        rcats = _ret_cats(name)
        if rcats and (lhs_cat not in rcats):
            mism.append({
                "name": name,
                "reason": "return_mismatch",
                "lhs_var": lhs_var,
                "lhs_type": lhs_type,
                "lhs_cat": lhs_cat,
                "ret_types": rets.get(name, [])[:8],
                "ret_cats": sorted(list(rcats))[:8],
            })

    return mism

def detect_shape_mismatches(
    code: str,
    whitelist_set: Set[str],
    sigs: Dict[str, List[List[str]]],
    *,
    rets: Optional[Dict[str, List[str]]] = None,
    check_types: bool = True,
    check_returns: bool = True,
    context_text: str = "",
    func_name: Optional[str] = None,
) -> List[Dict]:
    decls = parse_decl_table(code, context_text=context_text, func_name=func_name)
    mismatches: List[Dict] = []

    for name, arg_str in find_call_sites(code):
        if name not in whitelist_set:
            continue
        if name not in sigs:
            continue

        args = split_args(arg_str)
        argc = len(args)

        sig_list = sigs[name]
        allowed_arities = sorted({len(s) for s in sig_list})

        if argc not in allowed_arities:
            mismatches.append({
                "name": name,
                "argc": argc,
                "args": args,
                "allowed_arities": allowed_arities,
                "allowed_sigs": sig_list,
                "reason": "argc_mismatch",
            })
            continue

        if not check_types:
            continue

        arg_cats = [arg_category(a, decls) for a in args]
        if any(c == "unknown" for c in arg_cats):
            continue

        ok_any = False
        for sig in sig_list:
            if len(sig) != argc:
                continue
            sig_cats = [sig_category(t) for t in sig]
            if sig_cats == arg_cats:
                ok_any = True
                break

        if not ok_any:
            mismatches.append({
                "name": name,
                "argc": argc,
                "args": args,
                "allowed_arities": allowed_arities,
                "allowed_sigs": sig_list,
                "reason": "type_mismatch",
                "arg_cats": arg_cats,
            })

    if check_returns and rets:
        mismatches += detect_return_mismatches(
            code,
            whitelist_set,
            rets,
            context_text=context_text,
            func_name=func_name,
        )

    return mismatches

def build_shape_repair_prompt(
    code: str,
    mismatches: List[Dict],
    sigs: Dict[str, List[List[str]]],
    arity_index: Dict[int, List[str]],
    spec_text: str,
    *,
    invalid_intrinsics: Optional[List[str]] = None,
    invalid_suggestions: Optional[Dict[str, List[str]]] = None,
    max_alts_per_call: int = 6,
) -> str:
    lines: List[str] = []
    rules = (
        FUNCTIONAL_REPAIR_RULES_FULL
        if COMPLETION_MODE == COMPLETION_MODE_FULL
        else FUNCTIONAL_REPAIR_RULES_SNIPPET
    )
    lines.append(rules)
    lines.append("\n[SPEC]\n" + spec_text.strip() + "\n")

    # Include invalid-name diagnostics here as well (shape-fix often introduces new sv* tokens).
    if invalid_intrinsics:
        lines.append("\n[INVALID_INTRINSICS]")
        for b in invalid_intrinsics:
            cand = (invalid_suggestions or {}).get(b, [])
            lines.append(f"- {b}")
            if cand:
                lines.append(f"  candidates: {', '.join(cand)}")
            else:
                lines.append("  candidates: (none found)")
        lines.append("")

    lines.append("\n[MISMATCHED_CALLS]")
    for mm in mismatches:
        name = mm.get("name", "")
        reason = mm.get("reason", "")

        if reason == "return_mismatch":
            lines.append(f"- {name}  (reason=return_mismatch)")
            lhs_var = mm.get("lhs_var", "")
            lhs_type = mm.get("lhs_type", "")
            lhs_cat = mm.get("lhs_cat", "")
            lines.append(f"  lhs: {lhs_type} {lhs_var}   (cat={lhs_cat})")
            lines.append(f"  allowed_return_categories: {mm.get('ret_cats', [])}")
            rts = mm.get("ret_types", [])
            if rts:
                lines.append(f"  known_return_types: {rts}")
            # Heuristic guidance for common predicate-related return mismatches
            if lhs_cat == "pred":
                lines.append(
                    "  hint: This intrinsic returns a vector type, so it cannot be assigned to svbool_t."
                )
                lines.append(
                    "  hint: If you intended a loop tail predicate, use svwhilelt_b8/b16/b32/b64 (pick the element width)."
                )
                lines.append(
                    "  hint: If you intended to build a predicate from a bool/byte mask array for 64-bit elements, a common pattern is:"
                )
                lines.append("        svbool_t pg = svwhilelt_b64(i, length);")
                lines.append(
                    "        svuint64_t m = svld1ub_u64(pg, (const uint8_t*)mask + i);"
                )
                lines.append("        svbool_t pm = svcmpne_n_u64(pg, m, 0);")
            continue

        argc = mm.get("argc", None)
        args = mm.get("args", [])
        args_preview = ", ".join(args)[:200] if isinstance(args, list) else str(args)[:200]
        lines.append(f"- {name}  (argc={argc}, reason={reason})")
        lines.append(f"  call_args: {args_preview}")

        sig_list = sigs.get(name, [])
        show = sig_list[:8]
        lines.append("  valid_signatures:")
        for sig in show:
            lines.append("    - " + name + "(" + ", ".join(sig) + ")")
        if len(sig_list) > len(show):
            lines.append(f"    ... ({len(sig_list) - len(show)} more)")

        # Suggest alternatives with the same argc
        try:
            argc_i = int(argc) if argc is not None else -1
        except Exception:
            argc_i = -1

        pool = arity_index.get(argc_i, []) if argc_i >= 0 else []
        alts = difflib.get_close_matches(name, pool, n=max_alts_per_call, cutoff=0.55)

        mtype = None
        m = re.search(r"_(u8|u16|u32|u64|s8|s16|s32|s64|f16|f32|f64)(?:_|$)", name)
        if m:
            mtype = m.group(1)
        if mtype and pool:
            typed = [x for x in pool if f"_{mtype}_" in x or x.endswith(f"_{mtype}") or x.endswith(f"_{mtype}_z")]
            typed = typed[:max_alts_per_call]
            for x in typed:
                if x not in alts:
                    alts.append(x)

        out_alts: List[str] = []
        seen: Set[str] = set()
        for a in alts:
            if a not in seen:
                out_alts.append(a)
                seen.add(a)
            if len(out_alts) >= max_alts_per_call:
                break
        if out_alts:
            lines.append("  possible_alternatives_same_argc: " + ", ".join(out_alts))

    lines.append("\n[COMPLETION_TO_FIX]\n" + code.strip() + "\n")
    lines.append("[END]")
    return "\n".join(lines)

@torch.no_grad()
def rag_fix_shapes(
    model,
    tok,
    *,
    completion_in: str,
    whitelist_set: Set[str],
    whitelist_list: List[str],
    sigs: Dict[str, List[List[str]]],
    rets: Optional[Dict[str, List[str]]] = None,
    max_iters: int,
    spec_text: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    check_types: bool,
    sid: str = "",
    rank: int = 0,
    func_name: Optional[str] = None,
    attempts_per_iter: int = 3,
    print_summary: bool = False,
    print_prompts: bool = False,
    dump_prompts_dir: Optional[Path] = None,
    prompt_max_chars: int = 0,
) -> Tuple[str, Dict]:
    info: Dict = {
        "iters": 0,
        "mismatch_before_count": 0,
        "mismatch_after_count": 0,
        "invalid_before_count": 0,
        "invalid_after_count": 0,
    }

    code = normalize_completion_snippet(completion_in, func_name)

    if not sigs:
        info["skipped"] = "no sigs in whitelist.json (shape-fix cannot run)"
        if print_summary:
            print("=" * 120)
            print(f"[SHAPE_FIX] rank={rank} id={sid} SKIP: {info['skipped']}")
            print("=" * 120)
        return code, info

    arity_index = build_arity_index(sigs)

    def _invalid_list(c: str) -> List[str]:
        return sorted([x for x in extract_calls(c, whitelist_set, context_text=spec_text) if x not in whitelist_set])

    def _mismatches(c: str) -> List[Dict]:
        return detect_shape_mismatches(
            c,
            whitelist_set,
            sigs,
            rets=rets,
            check_types=check_types,
            check_returns=True,
            context_text=spec_text,
            func_name=func_name,
        )

    def _score(c: str) -> Tuple[int, List[str], List[Dict], List[str]]:
        invalid = _invalid_list(c)
        mism = _mismatches(c)
        forbidden = detect_forbidden_structures(c, func_name)
        score = (len(forbidden) * 1000000) + (len(invalid) * 1000) + (len(mism) * 10)
        return score, invalid, mism, forbidden

    score0, invalid0, mism0, forbid0 = _score(code)
    info["invalid_before_count"] = len(invalid0)
    info["invalid_before"] = invalid0[:50]
    info["mismatch_before_count"] = len(mism0)
    info["mismatch_before"] = mism0[:20]
    info["forbidden_before"] = forbid0

    best_code = code
    best_score = score0

    if print_summary:
        try:
            print("=" * 120)
            print(f"[SHAPE_FIX] rank={rank} id={sid} invalid_before={len(invalid0)} mismatch_before={len(mism0)} max_iters={max_iters} check_types={check_types}")
            for mm in mism0[:10]:
                print(f"  - {mm.get('name')} reason={mm.get('reason')} argc={mm.get('argc')} allowed={mm.get('allowed_arities')}")
            print("=" * 120)
        except OSError:
            # stdout/stderr can be closed in long runs; avoid crashing the whole job
            print_summary = False

    if (not invalid0) and (not mism0) and (not forbid0):
        info["mismatch_after_count"] = 0
        info["invalid_after_count"] = 0
        info["forbidden_after"] = []
        return code, info

    if max_iters <= 0:
        info["mismatch_after_count"] = len(mism0)
        info["invalid_after_count"] = len(invalid0)
        info["forbidden_after"] = forbid0
        return code, info

    it = 0
    while it < max_iters:
        cur_score, cur_invalid, cur_mism, cur_forbid = _score(code)
        if (not cur_invalid) and (not cur_mism) and (not cur_forbid):
            break

        # Build invalid suggestions (to discourage inventing new sv* calls)
        invalid_sugs: Dict[str, List[str]] = {}
        if cur_invalid:
            op_index = build_op_index(whitelist_list)

        # Force-remove a few known-invalid/invented SVE call names that sometimes slip into LLM output.
        # This keeps the whitelist conservative and helps the repair loop converge.
        for _bad in sorted(_FORCE_INVALID_SVE_CALLS):
            if _bad in whitelist_set:
                whitelist_set.discard(_bad)
                if sigs is not None:
                    sigs.pop(_bad, None)
                if rets is not None:
                    rets.pop(_bad, None)
            invalid_sugs = {
                b: suggest_for_name(b, whitelist_set, whitelist_list, op_index, code, top_k=8, cutoff=0.55)
                for b in cur_invalid
            }

        prompt = build_shape_repair_prompt(
            code,
            cur_mism,
            sigs,
            arity_index,
            spec_text=spec_text,
            invalid_intrinsics=cur_invalid,
            invalid_suggestions=invalid_sugs,
        )
        iter_no = it + 1

        if print_prompts:
            print("\n" + "=" * 120)
            print(f"[SHAPE_PROMPT] rank={rank} id={sid} iter={iter_no} invalid={len(cur_invalid)} mismatches={len(cur_mism)}")
            print("-" * 120)
            if prompt_max_chars and prompt_max_chars > 0 and len(prompt) > prompt_max_chars:
                print(prompt[:prompt_max_chars] + "\n...[TRUNCATED]...")
            else:
                print(prompt)
            print("=" * 120 + "\n")

        if dump_prompts_dir is not None:
            dump_prompts_dir.mkdir(parents=True, exist_ok=True)
            (dump_prompts_dir / f"shape_prompt_iter{iter_no}.txt").write_text(prompt, encoding="utf-8")
            (dump_prompts_dir / f"shape_mismatches_iter{iter_no}.json").write_text(
                json.dumps({"sid": sid, "rank": rank, "iter": iter_no, "invalid": cur_invalid, "mismatches": cur_mism}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        chosen = None
        chosen_score = cur_score

        tries = max(1, int(attempts_per_iter))
        for aidx in range(tries):
            out = generate_text(
                model,
                tok,
                user_text=prompt,
                system_text="",
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            cand = normalize_completion_snippet(out, func_name)
            sc, inv, mm, forb = _score(cand)

            if sc < best_score:
                best_score = sc
                best_code = cand

            if sc < chosen_score:
                chosen = cand
                chosen_score = sc

        if chosen is None:
            # no improvement found; stop
            break

        code = chosen
        it += 1

    # Final selection
    code_final = best_code
    invalid_after = _invalid_list(code_final)
    mism_after = _mismatches(code_final)
    forbid_after = detect_forbidden_structures(code_final, func_name)

    info["iters"] = it
    info["invalid_after_count"] = len(invalid_after)
    info["invalid_after"] = invalid_after[:50]
    info["mismatch_after_count"] = len(mism_after)
    info["mismatch_after"] = mism_after[:20]
    info["forbidden_after"] = forbid_after

    if print_summary:
        try:
            print("=" * 120)
            print(f"[SHAPE_FIX_DONE] rank={rank} id={sid} iters={it} invalid_after={len(invalid_after)} mismatch_after={len(mism_after)}")
            print("=" * 120)
        except OSError:
            pass

    return code_final, info


# =============================================================================
# Remote evaluation feedback loop (per-sample) via SimdBench evaluate_functional_correctness
# =============================================================================

_REMOTE_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_@%+=:,./~$-]+$")

def _quote_remote_token(tok: str) -> str:
    if _REMOTE_SAFE_TOKEN_RE.match(tok):
        return tok
    return shlex.quote(tok)

def _cmd_to_str(cmd: List[str]) -> str:
    try:
        return shlex.join(cmd)
    except Exception:
        return " ".join(shlex.quote(x) for x in cmd)

def _tail_chars(s: str, n: int) -> str:
    if not s:
        return ""
    if n <= 0:
        return s
    return s[-n:]

def _safe_decode_process_output(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return bytes(value).decode("utf-8", errors="replace")

def _run_cmd_capture(
    cmd: List[str],
    timeout_s: int = 0,
    *,
    input_text: Optional[str] = None,
    log_path: Optional[Path] = None,
    print_cmd: bool = False,
    tail_chars: int = 4000,
) -> Tuple[int, str, str, str]:
    cmd_str = _cmd_to_str(cmd)
    t0 = time.time()

    try:
        p = subprocess.run(
            cmd,
            input=input_text.encode("utf-8") if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=None if timeout_s <= 0 else timeout_s,
        )
        rc = p.returncode
        out = _safe_decode_process_output(p.stdout)
        err = _safe_decode_process_output(p.stderr)
    except subprocess.TimeoutExpired as e:
        rc = 124
        out = _safe_decode_process_output(e.stdout)
        err = _safe_decode_process_output(e.stderr) + f"\n[TIMEOUT] {timeout_s}s\n"
    except Exception as e:
        rc, out, err = 1, "", f"[EXCEPTION] {repr(e)}\n"

    dt = time.time() - t0

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "[CMD]\n" + cmd_str + "\n\n"
            "[RC]\n" + str(rc) + "\n\n"
            f"[ELAPSED_S]\n{dt:.3f}\n\n"
            "[STDOUT]\n" + out + "\n\n"
            "[STDERR]\n" + err + "\n",
            encoding="utf-8",
            errors="replace",
        )

    if print_cmd:
        def _safe_print(*args, **kwargs) -> None:
            try:
                print(*args, **kwargs)
            except OSError:
                pass
        _safe_print("=" * 120)
        _safe_print("[CMD]", cmd_str)
        _safe_print("[RC]", rc, f"(elapsed {dt:.3f}s)")
        if out.strip():
            _safe_print("[STDOUT_TAIL]\n" + _tail_chars(out, tail_chars))
        if err.strip():
            _safe_print("[STDERR_TAIL]\n" + _tail_chars(err, tail_chars))
        if log_path is not None:
            _safe_print("[LOG_PATH]", str(log_path))
        _safe_print("=" * 120)

    return rc, out, err, cmd_str

def _ssh_scp_common_opts(no_strict_hostkey: bool) -> List[str]:
    opts = [
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=3",
    ]
    if no_strict_hostkey:
        opts += ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    return opts

def remote_mkdir(
    user: str,
    host: str,
    port: int,
    key_path: str,
    remote_dir: str,
    *,
    no_strict_hostkey: bool = False,
    log_path: Optional[Path] = None,
    print_cmd: bool = False,
) -> Tuple[bool, str]:
    remote = f"{user}@{host}"
    cmd = ["ssh", "-i", key_path, "-p", str(port)] + _ssh_scp_common_opts(no_strict_hostkey) + [
        remote,
        f"mkdir -p {_quote_remote_token(remote_dir)}",
    ]
    rc, out, err, cmd_str = _run_cmd_capture(cmd, timeout_s=60, log_path=log_path, print_cmd=print_cmd)
    ok = (rc == 0)
    msg = (
        f"[REMOTE_MKDIR]\n"
        f"[CMD]\n{cmd_str}\n"
        f"[RC] {rc}\n"
        f"[STDOUT_TAIL]\n{_tail_chars(out, 2000)}\n"
        f"[STDERR_TAIL]\n{_tail_chars(err, 2000)}\n"
    )
    return ok, msg

def remote_scp_file(
    user: str,
    host: str,
    port: int,
    key_path: str,
    local_path: Path,
    remote_path: str,
    *,
    no_strict_hostkey: bool = False,
    log_path: Optional[Path] = None,
    print_cmd: bool = False,
) -> Tuple[bool, str]:
    remote = f"{user}@{host}"
    cmd = ["scp", "-i", key_path, "-P", str(port)] + _ssh_scp_common_opts(no_strict_hostkey) + [
        str(local_path),
        f"{remote}:{remote_path}",
    ]
    rc, out, err, cmd_str = _run_cmd_capture(cmd, timeout_s=180, log_path=log_path, print_cmd=print_cmd)
    ok = (rc == 0)
    msg = (
        f"[SCP_FILE]\n"
        f"[CMD]\n{cmd_str}\n"
        f"[RC] {rc}\n"
        f"[STDOUT_TAIL]\n{_tail_chars(out, 2000)}\n"
        f"[STDERR_TAIL]\n{_tail_chars(err, 2000)}\n"
    )
    return ok, msg

def _parse_last_json_line(stdout: str) -> Optional[Dict]:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            try:
                return json.loads(ln)
            except Exception:
                continue
    return None

def remote_eval_compile_only(
    *,
    user: str,
    host: str,
    port: int,
    key_path: str,
    remote_src_path: str,
    remote_obj_path: str,
    compiler: str,
    cflags: str,
    timeout_s: int,
    flock_path: str,
    no_strict_hostkey: bool,
    log_path: Optional[Path],
    print_cmd: bool,
) -> Dict:
    compile_args = [compiler] + shlex.split(cflags) + ["-c", remote_src_path, "-o", remote_obj_path]
    remote_cmd = " ".join(_quote_remote_token(x) for x in compile_args)

    if flock_path:
        remote_cmd = f"flock -x {_quote_remote_token(flock_path)} -c {shlex.quote(remote_cmd)}"

    remote = f"{user}@{host}"
    cmd = ["ssh", "-i", key_path, "-p", str(port)] + _ssh_scp_common_opts(no_strict_hostkey) + [remote, remote_cmd]
    rc, out, err, cmd_str = _run_cmd_capture(
        cmd,
        timeout_s=max(60, timeout_s + 30),
        log_path=log_path,
        print_cmd=print_cmd,
        tail_chars=8000,
    )

    compile_ok = 1 if rc == 0 else 0
    res = {
        "compile_ok": compile_ok,
        "run_ok": compile_ok,
        "reason": "compile_only_ok" if compile_ok else "compile_fail",
        "compile_rc": rc,
        "run_rc": None,
        "compiler": compiler,
        "cflags": cflags,
        "compile_log_tail": _tail_chars(out + "\n" + err, 8000),
        "run_log_tail": "",
        "_ssh_cmd": cmd_str,
        "_remote_cmd": remote_cmd,
    }
    return res

def remote_eval_cmd_json(
    *,
    user: str,
    host: str,
    port: int,
    key_path: str,
    remote_cmd_template: str,
    format_vars: Dict[str, str],
    timeout_s: int,
    flock_path: str,
    no_strict_hostkey: bool,
    log_path: Optional[Path],
    print_cmd: bool,
) -> Tuple[Optional[Dict], str]:
    remote = f"{user}@{host}"

    try:
        remote_cmd = remote_cmd_template.format(**format_vars)
    except KeyError as e:
        return None, f"[ERROR] remote_cmd_template missing format key: {e}\n"

    if flock_path:
        remote_cmd = f"flock -x {_quote_remote_token(flock_path)} -c {shlex.quote(remote_cmd)}"

    cmd = ["ssh", "-i", key_path, "-p", str(port)] + _ssh_scp_common_opts(no_strict_hostkey) + [remote, remote_cmd]
    rc, out, err, cmd_str = _run_cmd_capture(
        cmd,
        timeout_s=max(60, timeout_s + 30),
        log_path=log_path,
        print_cmd=print_cmd,
        tail_chars=8000,
    )

    blob = (
        "[REMOTE_EVAL_CMD]\n"
        f"[SSH_CMD]\n{cmd_str}\n"
        f"[REMOTE_CMD]\n{remote_cmd}\n"
        f"[RC] {rc}\n"
        f"[STDOUT_TAIL]\n{_tail_chars(out, 8000)}\n"
        f"[STDERR_TAIL]\n{_tail_chars(err, 8000)}\n"
    )

    if rc != 0:
        return None, blob

    js = _parse_last_json_line(out)
    if js is None:
        return None, blob + "\n[ERROR] json_parse_fail: no valid JSON line found in stdout\n"

    js["_ssh_cmd"] = cmd_str
    js["_remote_cmd"] = remote_cmd
    js["_ssh_stdout_tail"] = _tail_chars(out, 8000)
    js["_ssh_stderr_tail"] = _tail_chars(err, 8000)
    return js, blob

def _extract_interesting_log_lines(log: str, *, max_lines: int = 40) -> str:
    """
    Extract the most helpful lines from a compile/run log.

    - Prefer compiler errors/warnings and obvious runtime failure markers.
    - If only a tiny number of "interesting" lines are found (e.g., just "logical bug"),
      also include the tail of the log to provide additional context.
    """
    if not log:
        return ""
    lines = [ln.rstrip("\n") for ln in str(log).splitlines()]
    if not lines:
        return ""

    key_patterns = [
        r"\berror\b",
        r"\bwarning\b",
        r"\bundefined\b",
        r"\bundeclared\b",
        r"no matching function",
        r"candidate function",
        r"note:",
        r"did you mean",
        r"runtime failed",
        r"SIGSEGV|segmentation fault",
        r"SIGABRT",
        r"glibc",
        r"malloc\(",
        r"corrupt",
        r"logical bug",
        r"mismatch|expected|Expected|got|Got",
    ]
    key_re = re.compile("|".join(key_patterns))

    interesting = [ln for ln in lines if key_re.search(ln)]
    tail = lines[-max_lines:]

    if interesting:
        # If we only have a couple of lines (common for "logical bug"), append tail context too.
        if len(interesting) < 3:
            merged: List[str] = []
            seen = set()
            for ln in interesting + tail:
                if ln not in seen:
                    merged.append(ln)
                    seen.add(ln)
            return "\n".join(merged[:max_lines]).strip()
        return "\n".join(interesting[:max_lines]).strip()

    return "\n".join(tail).strip()


_DID_YOU_MEAN_RE = re.compile(
    r"use of undeclared identifier '([^']+)'; did you mean '([^']+)'\?"
)
_UNDECLARED_IDENTIFIER_RE = re.compile(r"use of undeclared identifier '([^']+)'")
_NO_MATCHING_CALL_RE = re.compile(r"no matching function for call to '([^']+)'")
_EXPECTED_TOKEN_RE = re.compile(r"expected '([^']+)'")
_CLANG_DIAG_LOCATION_RE = re.compile(
    r"^(?:\[compile_mode=[^\]]+\]\s+compilation failed:\s+)?"
    r"(?P<path>(?:\[REDACTED\]|[^:\n]+).*?):(?P<line>\d+):(?P<col>\d+):\s+"
    r"(?P<kind>error|warning|note):\s+(?P<message>.+)$"
)
_LOCAL_HELPER_NAME_RE = re.compile(
    r"^(?:"
    r"[A-Za-z_]\w*_local|"
    r"(?:round|shift|round_shift|shift_then|avg\d*|avg|clamp|abs|bit_mask|pack|unpack|saturate|narrow|widen)_[A-Za-z0-9_]+"
    r")$"
)
_INDEX_HELPER_NAME_RE = re.compile(r"^idx\d+_local$")


def _dedup_text_items(items: Sequence[Any], *, limit: int = 12) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        txt = str(item).strip()
        if not txt or txt in seen:
            continue
        seen.add(txt)
        out.append(txt)
        if len(out) >= max(0, int(limit)):
            break
    return out


def _looks_like_local_helper_symbol(name: str) -> bool:
    return bool(_LOCAL_HELPER_NAME_RE.match(str(name or "").strip()))


def _looks_like_index_helper_symbol(name: str) -> bool:
    return bool(_INDEX_HELPER_NAME_RE.match(str(name or "").strip()))


def _looks_like_source_line(line: str) -> bool:
    text = str(line or "").rstrip()
    if not text.strip():
        return False
    stripped = text.lstrip()
    low = stripped.lower()
    if stripped.startswith("^") or set(stripped) <= {"^", "~", " "}:
        return False
    if low.startswith(("in file included from", "from ", "candidate ", "note:", "error:", "warning:")):
        return False
    if _CLANG_DIAG_LOCATION_RE.match(stripped):
        return False
    return True


def _compact_source_context(text: str, *, limit: int = 160, from_right: bool = False) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    if len(cleaned) <= max(0, int(limit)):
        return cleaned
    keep = max(0, int(limit) - 3)
    if from_right:
        return "..." + cleaned[-keep:]
    return cleaned[:keep] + "..."


def _source_focus_token_from_message(message: str) -> str:
    msg = str(message or "")
    for pat in [
        r"call to ['‘]([^'’]+)['’]",
        r"no member named ['‘]([^'’]+)['’]",
        r"use of undeclared identifier ['‘]([^'’]+)['’]",
        r"unknown type name ['‘]([^'’]+)['’]",
        r"redefinition of ['‘]([^'’]+)['’]",
    ]:
        m = re.search(pat, msg)
        if not m:
            continue
        name = str(m.group(1) or "").strip()
        if "(" in name:
            name = name.split("(", 1)[0].strip()
        if "::" in name:
            name = name.rsplit("::", 1)[-1].strip()
        if name:
            return name
    return ""


def _source_focus_for_column(source_line: str, column: str, *, long_threshold: int = 220, focus_token: str = "") -> str:
    """Render a source line, focusing around clang's column for one-line functions."""
    src = str(source_line or "").rstrip()
    if not src:
        return ""
    stripped = src.strip()
    if len(stripped) <= long_threshold:
        return f"source: {stripped}"

    try:
        col_idx = max(0, int(str(column or "1")) - 1)
    except Exception:
        col_idx = 0
    col_idx = min(max(0, col_idx), max(0, len(src) - 1))

    token = str(focus_token or "").strip()
    if token and token in src:
        token_positions = [m.start() for m in re.finditer(re.escape(token), src)]
        if token_positions:
            col_idx = min(token_positions, key=lambda pos: abs(pos - col_idx))

    left_candidates = [src.rfind(ch, 0, col_idx + 1) for ch in (";", "{", "}")]
    left = max(left_candidates)
    focus_start = left + 1 if left >= 0 else max(0, col_idx - 80)

    right_positions = [pos for pos in (src.find(ch, col_idx) for ch in (";", "{", "}")) if pos >= 0]
    focus_end = (min(right_positions) + 1) if right_positions else min(len(src), col_idx + 120)

    if focus_end <= focus_start:
        focus_start = max(0, col_idx - 80)
        focus_end = min(len(src), col_idx + 120)

    focus = _compact_source_context(src[focus_start:focus_end], limit=220)
    before = _compact_source_context(src[:focus_start], limit=140, from_right=True)
    after = _compact_source_context(src[focus_end:], limit=140)

    pieces = [f"source_focus: {focus}"]
    if before:
        pieces.append(f"context_before: {before}")
    if after:
        pieces.append(f"context_after: {after}")
    return " | ".join(pieces)


def _compile_diag_origin_from_path(path: str) -> str:
    """Classify a diagnostic path without retaining the raw path in prompts."""
    low = str(path or "").strip().lower()
    if not low:
        return "user"
    if "[redacted]" in low or "/tmp" in low or "tmp" in low and ".cpp" in low:
        return "user"
    if "arm_sve.h" in low or "arm_neon.h" in low:
        return "acle"
    if "/usr/include/c++" in low or "/bits/" in low or "bits/" in low or "__gnu_cxx" in low:
        return "stl"
    if "/usr/include" in low or "/lib/clang/" in low or "/usr/lib/" in low:
        return "system"
    return "user"


def _extract_compile_location_details(compile_log_tail: str, *, limit: int = 8) -> List[str]:
    """Keep source locations/source lines without leaking raw temp paths."""
    lines = str(compile_log_tail or "").splitlines()
    out: List[str] = []
    for idx, line in enumerate(lines):
        match = _CLANG_DIAG_LOCATION_RE.match(line.strip())
        if not match:
            continue
        kind = match.group("kind")
        if kind not in {"error", "note"}:
            continue
        message = str(match.group("message") or "").strip()
        if kind == "note" and not (
            "to match this" in message.lower()
            or "candidate" in message.lower()
            or "conversion" in message.lower()
        ):
            continue
        origin = _compile_diag_origin_from_path(match.group("path"))
        source_line = ""
        for next_line in lines[idx + 1 : idx + 4]:
            if _looks_like_source_line(next_line):
                source_line = str(next_line or "").rstrip()
                break
        origin_prefix = f" origin={origin}" if origin != "user" else ""
        detail = f"{kind}:{origin_prefix} line {match.group('line')}, column {match.group('col')}: {message}"
        if source_line:
            focus_token = _source_focus_token_from_message(message)
            source_detail = _source_focus_for_column(source_line, match.group("col"), focus_token=focus_token)
            if source_detail:
                detail += f" | {source_detail}"
        out.append(detail)
        if len(out) >= max(0, int(limit)):
            break
    return _dedup_text_items(out, limit=limit)


def _extract_compile_diagnostic_supplements(compile_log_tail: str) -> Dict[str, Any]:
    """Small local supplement so shared/fallback extractors keep the same prompt-facing semantics."""
    text = str(compile_log_tail or "")
    if not text.strip():
        return {}

    no_matching_calls: List[str] = []
    expected_tokens: List[str] = []
    syntax_signals: List[str] = []
    invalid_cast_messages: List[str] = []
    invalid_cpp_construct_messages: List[str] = []
    type_mismatch_messages: List[str] = []
    sve_tuple_access_messages: List[str] = []
    sizeless_type_messages: List[str] = []
    remote_evaluator_artifacts: List[str] = []

    for match in _NO_MATCHING_CALL_RE.finditer(text):
        name = str(match.group(1) or "").strip()
        if "(" in name:
            name = name.split("(", 1)[0].strip()
        if "::" in name:
            name = name.rsplit("::", 1)[-1].strip()
        if name:
            no_matching_calls.append(name)
    for match in _EXPECTED_TOKEN_RE.finditer(text):
        tok = str(match.group(1) or "").strip()
        if tok:
            expected_tokens.append(tok)

    for raw_line in text.splitlines():
        line = str(raw_line or "").strip()
        low = line.lower()
        if "[tmp_src_missing]" in low or "tmp_src_missing" in low:
            remote_evaluator_artifacts.append("tmp_src_missing")
        if "expecting value: line 1 column 1" in low or "json parse" in low and "empty" in low:
            remote_evaluator_artifacts.append("json_parse_empty_output")
        if "remote_cmd_fail" in low:
            remote_evaluator_artifacts.append("remote_cmd_fail")
        if "ssh" in low and ("timed out" in low or "connection reset" in low or "connection timed out" in low):
            remote_evaluator_artifacts.append("ssh_transport_failure")

        if (
            "expected ';'" in low
            or "expected '}'" in low
            or "expected ')'" in low
            or "expected ']'" in low
            or "expected expression" in low
            or "function definition is not allowed here" in low
            or "extraneous closing brace" in low
            or "at end of input" in low
        ):
            syntax_signals.append(line)
        if "error:" not in low:
            continue
        if (
            "reinterpret_cast" in low
            or "static_cast from" in low
            or "c-style cast" in low
            or "cannot cast" in low
            or "casts away qualifiers" in low
            or "assignment to cast is illegal" in low
        ):
            invalid_cast_messages.append(line)
        if (
            "called object type" in low and "is not a function" in low
            or "variable-sized object may not be initialized" in low
            or "expression is not assignable" in low
            or "cannot delete expression" in low
            or "integer literal is too large" in low
            or "no matching literal operator" in low
            or "invalid digit" in low
        ):
            invalid_cpp_construct_messages.append(line)
        if (
            "invalid operands to binary expression" in low
            or "not contextually convertible" in low
            or "cannot initialize" in low
            or "cannot assign" in low
            or "invalid argument type" in low
            or "subscripted value is not an array" in low
            or "cannot be narrowed" in low
            or "cannot convert between scalar type" in low and "vector type" in low
        ):
            type_mismatch_messages.append(line)
        if "member reference base type" in low and (
            "__clang_sv" in low or "__sv" in low or re.search(r"\bsv\w*x\d+_t\b", low)
        ):
            sve_tuple_access_messages.append(line)
        if (
            "array has sizeless element type" in low
            or "address of vector element requested" in low
            or "subscript of svbool_t is not allowed" in low
            or "sizeless type" in low
        ):
            sizeless_type_messages.append(line)

    out = {
        "no_matching_calls": _dedup_text_items(no_matching_calls, limit=12),
        "expected_tokens": _dedup_text_items(expected_tokens, limit=8),
        "syntax_signals": _dedup_text_items(syntax_signals, limit=8),
        "invalid_cast_messages": _dedup_text_items(invalid_cast_messages, limit=8),
        "invalid_cpp_construct_messages": _dedup_text_items(invalid_cpp_construct_messages, limit=8),
        "type_mismatch_messages": _dedup_text_items(type_mismatch_messages, limit=8),
        "sve_tuple_access_messages": _dedup_text_items(sve_tuple_access_messages, limit=8),
        "sizeless_type_messages": _dedup_text_items(sizeless_type_messages, limit=8),
        "remote_evaluator_artifacts": _dedup_text_items(remote_evaluator_artifacts, limit=8),
    }
    return {k: v for k, v in out.items() if v}


def _merge_compile_diagnostic_maps(*maps: Dict[str, Any]) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}
    for raw in maps:
        clean = _clean_named_diag_map(raw or {}, per_key_limit=12)
        for key, values in clean.items():
            if not values:
                continue
            merged.setdefault(key, []).extend(values)
    return {k: _dedup_text_items(v, limit=12) for k, v in merged.items() if v}


def extract_remote_compile_diagnostics(compile_log_tail: str) -> Dict[str, Any]:
    if extract_shared_compile_diagnostics is not None:
        try:
            shared = extract_shared_compile_diagnostics(compile_log_tail)
        except Exception:
            shared = None
        if isinstance(shared, dict):
            out_shared = {str(k): v for k, v in shared.items() if v}
            supplements = _extract_compile_diagnostic_supplements(compile_log_tail)
            if supplements:
                out_shared = _merge_compile_diagnostic_maps(out_shared, supplements)
            location_details = _extract_compile_location_details(compile_log_tail)
            if location_details:
                out_shared["diagnostic_locations"] = location_details
            return out_shared

    text = str(compile_log_tail or "")
    if not text.strip():
        return {}

    undeclared_identifiers: List[str] = []
    did_you_mean_pairs: List[str] = []
    no_matching_calls: List[str] = []
    expected_tokens: List[str] = []
    syntax_signals: List[str] = []
    diagnostic_locations: List[str] = _extract_compile_location_details(text)

    for match in _UNDECLARED_IDENTIFIER_RE.finditer(text):
        undeclared_identifiers.append(match.group(1))
    for match in _DID_YOU_MEAN_RE.finditer(text):
        did_you_mean_pairs.append(f"{match.group(1)}->{match.group(2)}")
    for match in _NO_MATCHING_CALL_RE.finditer(text):
        no_matching_calls.append(match.group(1))
    for match in _EXPECTED_TOKEN_RE.finditer(text):
        tok = str(match.group(1) or "").strip()
        if tok:
            expected_tokens.append(tok)

    for line in text.splitlines():
        low = line.lower()
        if (
            "expected ';'" in low
            or "expected '}'" in low
            or "expected ')'" in low
            or "expected expression" in low
            or "function definition is not allowed here" in low
        ):
            syntax_signals.append(line.strip())

    missing_helper_symbols = [
        name for name in undeclared_identifiers if _looks_like_local_helper_symbol(name)
    ]
    missing_index_symbols = [
        name for name in undeclared_identifiers if _looks_like_index_helper_symbol(name)
    ]
    unsupported_symbols = [
        name
        for name in list(undeclared_identifiers)
        if name.startswith("__builtin_") or name.startswith("sv")
    ]
    unsupported_symbols.extend(
        name for name in no_matching_calls if str(name or "").startswith("__builtin_")
    )

    out = {
        "undeclared_identifiers": _dedup_text_items(undeclared_identifiers, limit=16),
        "missing_helper_symbols": _dedup_text_items(missing_helper_symbols, limit=12),
        "missing_index_symbols": _dedup_text_items(missing_index_symbols, limit=12),
        "unsupported_symbols": _dedup_text_items(unsupported_symbols, limit=12),
        "did_you_mean_pairs": _dedup_text_items(did_you_mean_pairs, limit=12),
        "no_matching_calls": _dedup_text_items(no_matching_calls, limit=12),
        "expected_tokens": _dedup_text_items(expected_tokens, limit=8),
        "syntax_signals": _dedup_text_items(syntax_signals, limit=8),
        "diagnostic_locations": _dedup_text_items(diagnostic_locations, limit=8),
    }
    out = {k: v for k, v in out.items() if v}
    supplements = _extract_compile_diagnostic_supplements(text)
    return _merge_compile_diagnostic_maps(out, supplements) if supplements else out

def apply_did_you_mean_renames(code: str, compile_log_tail: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Apply clang-style "did you mean" identifier suggestions to the code.

    This is intentionally conservative: it only applies replacements when both sides look like valid
    C/C++ identifiers (to avoid rewriting intrinsics/macros accidentally).
    """
    if not code or not compile_log_tail:
        return code, []

    renames: List[Tuple[str, str]] = []
    for m in _DID_YOU_MEAN_RE.finditer(str(compile_log_tail)):
        old, new = m.group(1), m.group(2)
        if old == new:
            continue
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", old):
            continue
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", new):
            continue
        renames.append((old, new))

    if not renames:
        return code, []

    out = code
    # Apply in order; later replacements can override earlier ones if they overlap (rare).
    for old, new in renames:
        out = re.sub(rf"\b{re.escape(old)}\b", new, out)

    return out, renames



# -------------------------
# SERIAL reference (scalar) helpers + cache
# -------------------------

# NOTE:
# When we ask a *second* model (e.g., DeepSeek) to generate a **serial C reference**
# implementation, we must aggressively remove SIMD/SVE hints from the context.
# Otherwise some models will "helpfully" output SVE/NEON intrinsics again.
#
# We keep this as a simple substring filter (case-insensitive) because the upstream
# prompts can be arbitrary mixtures of JSON, comments, and code stubs.
_SERIAL_FORBIDDEN_LINE_SUBSTRINGS: List[str] = [
    # SIMD-related headers / ecosystems
    "arm_sve.h",
    "arm_neon.h",
    "immintrin.h",
    "xmmintrin.h",
    "emmintrin.h",
    "smmintrin.h",
    "avxintrin.h",
    "avx512",
    "avx2",
    "sse4",
    "sse3",
    "sse2",

    # SIMD buzzwords that frequently appear in the dataset descriptions
    "scalable vector extension",
    "arm scalable vector extension",
    "arm c language extensions",
    "acle",
    "sve",
    "neon",
    "simd intrinsics",
    "intrinsic",
    "intrinsics",
    "vectorize",
    "vectorization",
    "parallelism",

    # Common intrinsic spellings / types (SVE/NEON/x86)
    "svbool_t",
    "svint",
    "svuint",
    "svfloat",
    "svld",
    "svst",
    "svmla",
    "svmad",
    "svcnt",
    "_mm",
    "__m128",
    "__m256",
    "__m512",
    "vld1",
    "vst1",
    "#pragma omp simd",
]

_SERIAL_FORBIDDEN_LINE_REGEXES: List[re.Pattern] = [
    # Any explicit sv* tokens (intrinsics or sv* types)
    re.compile(r"\bsv[a-zA-Z0-9_]*\b"),
    # SIMD family words (standalone), case-insensitive
    re.compile(r"\b(sve|simd|neon|avx|sse|rvv)\b", re.IGNORECASE),
    # x86 SIMD types / intrinsics
    re.compile(r"\b__m(?:128|256|512)\b"),
    re.compile(r"\b_mm[a-zA-Z0-9_]*\b"),
]

def _serial_line_has_simd_cues(line: str) -> bool:
    if line is None:
        return False
    ll = line.lower()
    if any(bad in ll for bad in _SERIAL_FORBIDDEN_LINE_SUBSTRINGS):
        return True
    for rx in _SERIAL_FORBIDDEN_LINE_REGEXES:
        if rx.search(line):
            return True
    return False

def sanitize_text_for_serial_context(text: str) -> str:
    """Remove obvious SIMD/SVE cues from a text blob before using it as context for serial code gen.

    This is intentionally conservative: if a line contains strong SIMD hints, we drop the line
    entirely. The goal is to *avoid biasing* the serial-reference model toward vector intrinsics.
    """
    if not text:
        return ""
    out_lines: List[str] = []
    for raw in text.splitlines():
        l = raw.rstrip("\n")
        if _serial_line_has_simd_cues(l):
            continue
        out_lines.append(l)
    return "\n".join(out_lines).strip()

def sanitize_prompt_prefix_for_serial(prompt_prefix: str) -> str:
    """Same as sanitize_text_for_serial_context, but for the C/C++ prompt prefix stub."""
    if not prompt_prefix:
        return ""
    out_lines: List[str] = []
    for raw in prompt_prefix.splitlines():
        l = raw.rstrip("\n")
        if _serial_line_has_simd_cues(l):
            continue
        out_lines.append(l)
    return "\n".join(out_lines).strip()

_SIMD_OUTPUT_REGEXES: List[re.Pattern] = [
    re.compile(r"#\s*include\s*<\s*arm_sve\.h\s*>"),
    re.compile(r"#\s*include\s*<\s*arm_neon\.h\s*>"),
    re.compile(r"\bsv[a-zA-Z0-9_]*\b"),
    re.compile(r"\bsv(?:bool_t|int\d+_t|uint\d+_t|float\d+_t)\b"),
    re.compile(r"\bfloat\d+x\d+_t\b"),  # NEON vector types
    re.compile(r"\b(?:vld1|vst1|vadd|vsub|vmul|vmla)[a-z0-9_]*\b"),
    re.compile(r"\b__m(?:128|256|512)\b"),
    re.compile(r"\b_mm(?:\d+)?[a-z0-9_]*\b"),
    re.compile(r"#\s*pragma\s+omp\s+simd"),
    re.compile(r"\b(sve|simd|neon|avx|sse|rvv)\b", re.IGNORECASE),
]

def looks_like_simd_output(code: str, *, func_name: Optional[str] = None) -> bool:
    """Heuristic check to reject non-serial outputs from the serial reference model."""
    if not code:
        return False
    s = str(code)
    if func_name:
        # Allow entrypoint names that contain "simd"/"sve" substrings.
        s = re.sub(rf"\b{re.escape(func_name)}\b", "FUNC_NAME", s)
    for rx in _SIMD_OUTPUT_REGEXES:
        if rx.search(s):
            return True
    return False


def strip_noncode(text: str) -> str:
    """Extract the most likely C/C++ code from an LLM response.

    Many chat models wrap code with Markdown fences. This helper pulls out fenced blocks
    if present; otherwise it returns the original text stripped.
    """
    if text is None:
        return ""
    s = str(text)

    if "```" in s:
        # Extract all fenced blocks; keep language tags if any.
        blocks = re.findall(r"```(?:[a-zA-Z0-9_+\-]+)?\s*\n(.*?)```", s, flags=re.S)
        blocks = [b.strip() for b in blocks if b and b.strip()]
        if blocks:
            return "\n\n".join(blocks).strip()

    return s.strip()

def _sha256_text(s: str) -> str:
    try:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    except Exception:
        return ""

class SerialRefCache:
    """A tiny jsonl-backed cache for PASSED serial reference completions (keyed by task_id).

    The cache is append-only; the latest record for a given task_id wins.
    """

    def __init__(self, path: str, *, reload_on_miss: bool = True, rank: int = 0):
        self.path = Path(path) if path else None
        self.reload_on_miss = bool(reload_on_miss)
        self.rank = int(rank)
        self._loaded = False
        self._map: Dict[str, Dict[str, Any]] = {}
        self._attempts: Dict[str, int] = {}  # per-process attempts for this run

    def load(self) -> None:
        if self.path is None:
            self._loaded = True
            return
        if not self.path.exists():
            self._loaded = True
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                if fcntl is not None:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    except Exception:
                        pass
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    tid = rec.get("task_id")
                    if not tid:
                        continue
                    ok = rec.get("ok")
                    passed = rec.get("passed")
                    # accept either {"ok": true} or {"passed": true}
                    if (ok is True) or (passed is True) or (isinstance(rec.get("result"), dict) and rec["result"].get("passed") is True):
                        self._map[str(tid)] = rec
        finally:
            self._loaded = True

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        if self.path is None:
            return None
        if not self._loaded:
            self.load()
        rec = self._map.get(task_id)
        if rec is None and self.reload_on_miss:
            # Re-read to pick up other ranks' writes.
            self._loaded = False
            self.load()
            rec = self._map.get(task_id)
        return rec

    def attempts(self, task_id: str) -> int:
        return int(self._attempts.get(task_id, 0))

    def bump_attempt(self, task_id: str) -> int:
        n = int(self._attempts.get(task_id, 0)) + 1
        self._attempts[task_id] = n
        return n

    def save_passed(
        self,
        *,
        task_id: str,
        func_name: Optional[str],
        completion: str,
        backend: Optional[str],
        model: Optional[str],
        result: Optional[Dict[str, Any]],
    ) -> bool:
        if self.path is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        code_hash = _sha256_text(completion)
        existing = self._map.get(task_id)
        if existing and code_hash and existing.get("code_sha256") == code_hash:
            return False

        # Keep only a small summary of result in the cache to avoid bloating jsonl.
        result_summary: Dict[str, Any] = {}
        if isinstance(result, dict):
            for k in ["passed", "compile_ok", "run_ok", "reason", "compile_log_tail", "run_log_tail"]:
                if k in result:
                    result_summary[k] = result.get(k)

        rec = {
            "task_id": task_id,
            "func_name": func_name,
            "completion": completion,
            "backend": backend,
            "model": model,
            "ok": True,
            "passed": True,
            "result": result_summary,
            "code_sha256": code_hash,
            "saved_at": _dt.datetime.now().isoformat(),
            "rank": self.rank,
        }

        line = json.dumps(rec, ensure_ascii=False) + "\n"
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                if fcntl is not None:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    except Exception:
                        pass
                f.write(line)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
        except Exception:
            return False

        self._map[task_id] = rec
        return True


def build_serial_reference_prompt(
    *,
    spec_text: str,
    prompt_prefix: str,
    func_name: Optional[str] = None,
    attempt_idx: int = 1,
    prev_serial_completion: Optional[str] = None,
    prev_serial_feedback: str = "",
    simd_failure_feedback: str = "",
    **_ignored: Any,
) -> str:
    """Build a prompt for a *scalar-only* serial C/C++ reference implementation.

    Critical design goal: remove/neutralize SVE/SIMD cues from the upstream task text.
    Many models will otherwise output vector intrinsics again (even when asked for scalar).
    """
    def _extract_spec_block(text: str, name: str) -> str:
        if not text:
            return ""
        m = re.search(
            rf"(?s)\[{re.escape(name)}\]\s*(.*?)(?=\n\[[A-Z_]+\]|\Z)",
            text,
        )
        return m.group(1).strip() if m else ""

    # If caller accidentally passes a JSON line from the dataset (it may contain `"intrinsic":"SVE"`),
    # extract only the human-readable `prompt` field to avoid leaking SIMD hints.
    spec_for_serial = spec_text or ""
    st = spec_for_serial.strip()
    if st.startswith("{") and ("\"prompt\"" in st or "'prompt'" in st):
        try:
            obj = json.loads(st)
            if isinstance(obj, dict) and isinstance(obj.get("prompt"), str):
                spec_for_serial = obj.get("prompt", "")
        except Exception:
            pass

    # Prefer compact semantic blocks over the full SPEC (which can contain SIMD cues).
    task_blk = _extract_spec_block(spec_for_serial, "TASK")
    exp_blk = _extract_spec_block(spec_for_serial, "EXPANDED_SPEC_AND_SVE_PLAN")
    if task_blk or exp_blk:
        spec_for_serial = "\n\n".join([x for x in [task_blk, exp_blk] if x]).strip()

    spec_text_serial = sanitize_text_for_serial_context(spec_for_serial)
    prefix_serial = sanitize_prompt_prefix_for_serial(prompt_prefix)

    # Provide an explicit target signature line if we can find it, to reduce drift.
    signature_hint = ""
    if func_name:
        signature_hint = extract_function_decl_from_prompt_prefix(prompt_prefix, func_name) or ""
        if not signature_hint:
            for ln in prefix_serial.splitlines():
                if func_name in ln and "(" in ln and "{" in ln:
                    signature_hint = ln.strip()
                    break

    parts: List[str] = []
    parts.append("You are generating a **SERIAL scalar C/C++ reference implementation**.")
    parts.append("")
    parts.append("HARD RULES (automatic checker will reject violations):")
    parts.append("1) Scalar-only: use plain loops and scalar operations.")
    parts.append("2) ABSOLUTELY NO SIMD/vector intrinsics (SVE/NEON/AVX/SSE/etc), no vector types, no inline asm.")
    parts.append("3) Do NOT include any SIMD headers (e.g., <arm_sve.h>, <arm_neon.h>, <immintrin.h>).")
    parts.append("4) Keep the function name/signature EXACTLY as given (even if it ends with '_simd').")
    parts.append("5) The function name may contain 'simd' or 'sve' but it is ONLY an API name; you MUST still write scalar code.")
    parts.append("6) Output ONLY code. No explanations. No markdown. No analysis.")
    parts.append("7) Self-check before final output: if you see any token like 'sv', '_mm', 'vld', 'vst', 'sve', 'simd', 'neon', 'avx', 'sse', REMOVE and rewrite.")
    parts.append("")
    parts.append("If the task text/prefix mentions SVE/SIMD/intrinsics, IGNORE that and still write scalar code.")
    parts.append("")
    if func_name:
        parts.append(f"TARGET FUNCTION: {func_name}")
        if signature_hint:
            parts.append(f"SIGNATURE HINT: {signature_hint}")
        parts.append("")
    parts.append("Forbidden tokens/examples (must not appear in your output):")
    parts.append("- svbool_t, svint*, svuint*, svfloat*")
    parts.append("- svld*, svst*, svcnt*, svmla*, svmad*")
    parts.append("- vld1*, vst1*")
    parts.append("- _mm*, __m128/__m256/__m512")
    parts.append("- #include <arm_sve.h> / <arm_neon.h> / <immintrin.h>")
    parts.append("")

    if simd_failure_feedback:
        parts.append("[SIMD_FAILURE_FEEDBACK] (sanitized; for debugging only; do not copy SIMD)")
        parts.append(sanitize_text_for_serial_context(simd_failure_feedback).strip())
        parts.append("[/SIMD_FAILURE_FEEDBACK]")
        parts.append("")

    if prev_serial_feedback:
        parts.append("[PREV_FEEDBACK]")
        parts.append(sanitize_text_for_serial_context(prev_serial_feedback).strip())
        parts.append("[/PREV_FEEDBACK]")
        parts.append("")

    if prev_serial_completion:
        if looks_like_simd_output(prev_serial_completion, func_name=func_name):
            parts.append("[PREV_BAD_OUTPUT_NOTICE]")
            parts.append("Previous attempt contained SIMD tokens. DO NOT include any SIMD tokens in your output.")
            parts.append("[/PREV_BAD_OUTPUT_NOTICE]")
            parts.append("")
        else:
            parts.append("[PREV_BAD_OUTPUT] (for reference only; DO NOT copy SIMD patterns)")
            parts.append(prev_serial_completion.strip())
            parts.append("[/PREV_BAD_OUTPUT]")
            parts.append("")

    parts.append("[SPEC] (sanitized)")
    parts.append(spec_text_serial.strip())
    parts.append("[/SPEC]")
    parts.append("")
    parts.append("[PROMPT_PREFIX] (sanitized)")
    parts.append(prefix_serial.strip())
    parts.append("[/PROMPT_PREFIX]")
    parts.append("")
    parts.append(f"Attempt {attempt_idx}: Now output the SERIAL scalar implementation code:")

    return "\n".join(parts)

def _extract_function_definition(code: str, name: str) -> str:
    """Best-effort extraction of a single C/C++ function definition by name.

    Returns the substring covering the declarator + body (matching braces,
    string/comment-aware). Returns "" if not found.

    Used by build_serial_reference_from_solution to pull JUST the entrypoint
    function from solution_scalar — without re-emitting the helpers
    (clamp_*_local, mirror_index_local, etc.) which otherwise duplicate when
    the remote evaluator concatenates `solution_scalar + completion`.
    """
    if not code or not name:
        return ""
    decl_pattern = re.compile(
        r"^[ \t]*"
        r"(?:(?:static|inline|extern|__attribute__\s*\([^)]*\))\s+)*"
        r"(?:[A-Za-z_][\w]*\s*[*&]*\s+)+"        # return type tokens
        + rf"({re.escape(name)})"                 # the target name
        + r"\s*\([^)]*\)\s*\{",
        re.MULTILINE,
    )
    matches = list(decl_pattern.finditer(code))
    if not matches:
        return ""

    def _candidate_score(match: re.Match) -> Tuple[int, int, int]:
        declarator = code[match.start():match.end()]
        is_static = 1 if re.match(r"^[ \t]*static\b", declarator) else 0
        params_m = re.search(r"\(([^)]*)\)\s*\{", declarator, re.DOTALL)
        params = params_m.group(1).strip() if params_m else ""
        if not params or params == "void":
            param_count = 0
        else:
            param_count = len([p for p in params.split(",") if p.strip()])
        # Prefer the public entrypoint over same-name local helper overloads.
        # If there are still ties, later definitions usually wrap earlier
        # helpers in generated scalar references.
        return (0 if is_static else 1, param_count, match.start())

    m = max(matches, key=_candidate_score)
    start = m.start()
    depth = 1
    i = m.end()  # right after opening '{'
    n = len(code)
    while i < n and depth > 0:
        c = code[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "/" and i + 1 < n and code[i + 1] == "/":
            while i < n and code[i] != "\n":
                i += 1
            continue
        elif c == "/" and i + 1 < n and code[i + 1] == "*":
            i += 2
            while i + 1 < n and not (code[i] == "*" and code[i + 1] == "/"):
                i += 1
            i = min(i + 2, n)
            continue
        elif c == '"' or c == "'":
            quote = c
            i += 1
            while i < n and code[i] != quote:
                if code[i] == "\\" and i + 1 < n:
                    i += 2
                else:
                    i += 1
            i += 1
            continue
        i += 1
    if depth != 0:
        return ""
    return code[start:i]


def build_serial_reference_from_solution(
    solution_scalar: str,
    entrypoint_scalar: str,
    entrypoint_simd: str,
) -> str:
    """Build a serial reference completion from the dataset scalar solution.

    The SimdBench scalar eval path still uses the problem's test harness, and
    that harness calls entrypoint_simd as the candidate under test.  Therefore
    the dataset scalar implementation must be renamed to entrypoint_simd, not
    ref_<entrypoint_simd>.

    NOTE: This returns ONLY the entrypoint function definition (renamed),
    without any helper functions or #include lines that solution_scalar may
    contain. The remote evaluator builds the cpp as
        header + solution_scalar + completion + test_correctness
    — `solution_scalar` is already concatenated in full earlier, so emitting
    its helpers a second time inside `completion` causes
    `redefinition of '<helper>'` link errors. Those helpers are visible to the
    renamed entrypoint via normal forward-declaration scoping.
    """
    if not solution_scalar:
        return ""
    target = entrypoint_simd if entrypoint_simd else entrypoint_scalar
    # Try to extract just the entrypoint_scalar function, then rename it.
    body = _extract_function_definition(str(solution_scalar), entrypoint_scalar) if entrypoint_scalar else ""
    if body and target and entrypoint_scalar and entrypoint_scalar != target:
        # Rename only the extracted function's declarator.  Do not rewrite the
        # function body: some scalar references call local helper overloads with
        # the same base name (for example softmax_scalar_ref(ptr, size)).  Those
        # helpers remain available from problem["solution_scalar"] and must keep
        # their original names.
        body = re.sub(
            rf"(^[ \t]*(?:(?:static|inline|extern|__attribute__\s*\([^)]*\))\s+)*(?:[A-Za-z_][\w]*\s*[*&]*\s+)+){re.escape(entrypoint_scalar)}(\s*\([^)]*\)\s*\{{)",
            rf"\1{target}\2",
            body,
            count=1,
            flags=re.MULTILINE,
        )
    if body:
        return body.strip()
    # Fallback: legacy whole-text rename (preserved for tasks where the
    # extractor can't find the entrypoint, e.g. unusual signatures).
    code = str(solution_scalar)
    if entrypoint_scalar and target and entrypoint_scalar != target:
        code = re.sub(rf"\b{re.escape(entrypoint_scalar)}\b", target, code)
    return code.strip()


def build_serial_reference_pseudocode(
    serial_ref_completion: str,
    *,
    func_name: Optional[str] = None,
    prompt_prefix: str = "",
    max_chars: int = 6000,
) -> str:
    """Convert a validated scalar/serial reference into non-compilable semantic pseudocode.

    This is intentionally line-oriented and conservative.  It keeps the loop/control structure
    and index formulas, but removes includes, preprocessor lines, braces, semicolons, and the
    serial function body as compilable C/C++ code.  The resulting block is safe to show to the
    repair model as a semantic oracle without teaching it to paste serial wrappers or harnesses.
    """
    src = str(serial_ref_completion or "").strip()
    if not src:
        return ""

    target_sig = ""
    if func_name and prompt_prefix:
        try:
            target_sig = target_signature_line_from_decl(
                extract_function_decl_from_prompt_prefix(prompt_prefix, func_name) or ""
            )
        except Exception:
            target_sig = ""

    try:
        src = strip_comments(src)
    except Exception:
        src = str(serial_ref_completion or "")

    lines: List[str] = []
    header_added = False
    indent = 0
    pending_unbraced_blocks = 0
    block_extra_close_stack: List[int] = []

    def emit(text: str) -> None:
        nonlocal header_added
        t = " ".join(str(text or "").strip().split())
        if not t:
            return
        if not header_added:
            header = target_sig or (t[:-1] if t.endswith("{") else t)
            header = re.sub(r"\s*\{\s*$", "", header).strip()
            header = " ".join(header.split())
            lines.append(f"function {header}:")
            header_added = True
            return
        lines.append("  " * max(1, indent) + t)

    def split_top_level_semicolon(text: str) -> List[str]:
        parts: List[str] = []
        cur: List[str] = []
        depth_paren = depth_bracket = 0
        for ch in str(text or ""):
            if ch == "(":
                depth_paren += 1
            elif ch == ")" and depth_paren > 0:
                depth_paren -= 1
            elif ch == "[":
                depth_bracket += 1
            elif ch == "]" and depth_bracket > 0:
                depth_bracket -= 1
            if ch == ";" and depth_paren == 0 and depth_bracket == 0:
                parts.append("".join(cur).strip())
                cur = []
                continue
            cur.append(ch)
        parts.append("".join(cur).strip())
        return parts

    def parse_parenthesized_control(s: str, keyword: str) -> Optional[Tuple[str, str]]:
        rest = str(s or "").strip()
        if not rest.startswith(keyword):
            return None
        pos = len(keyword)
        while pos < len(rest) and rest[pos].isspace():
            pos += 1
        if pos >= len(rest) or rest[pos] != "(":
            return None
        close = _find_matching_paren(rest, pos)
        if close < 0:
            return None
        cond = rest[pos + 1 : close].strip()
        tail = rest[close + 1 :].strip()
        return cond, tail

    def pseudo_statement(stmt: str) -> str:
        s = " ".join(str(stmt or "").strip().split())
        s = s.rstrip(";").strip()
        if not s:
            return ""
        s = re.sub(r"\breturn\s+\[(.*)\]$", r"return [\1]", s)
        s = re.sub(r"\breturn\s+empty\s+collection$", "return empty collection", s)
        s = re.sub(
            r"^(?:std::)?vector\s*<[^>]+>\s+([A-Za-z_]\w*)\s*=\s*empty\s+collection$",
            r"\1 = empty collection",
            s,
        )
        s = re.sub(
            r"^(?:std::)?vector\s*<[^>]+>\s+([A-Za-z_]\w*)\s*=\s*\[(.*)\]$",
            r"\1 = [\2]",
            s,
        )
        s = re.sub(
            r"^(?:std::)?vector\s*<[^>]+>\s+([A-Za-z_]\w*)\s*\(\s*0\s*\)$",
            r"\1 = empty collection",
            s,
        )
        s = re.sub(r"^([A-Za-z_]\w*)\s*=\s*\[(.*)\]$", r"\1 = [\2]", s)
        s = re.sub(r"^([A-Za-z_]\w*)\s*=\s*empty\s+collection$", r"\1 = empty collection", s)
        m_bit_src = re.match(r"^u\.i\s*=\s*(.+)$", s)
        if m_bit_src:
            return "bit_source = " + m_bit_src.group(1).strip()
        m_bit_dst = re.match(r"^(.+?)\s*=\s*u\.f$", s)
        if m_bit_dst:
            return m_bit_dst.group(1).strip() + " = reinterpret bit_source as float"
        if s.startswith("union ") and "uint32_t" in s and "float" in s:
            return "prepare uint32-to-float bit reinterpret container"

        parsed_for = parse_parenthesized_control(s, "for")
        if parsed_for:
            head, tail = parsed_for
            pieces = split_top_level_semicolon(head)
            if len(pieces) < 3 and ":" in head:
                out = f"for each ({head.strip()}):"
                if tail:
                    out += " " + pseudo_statement(tail)
                return out
            while len(pieces) < 3:
                pieces.append("")
            init, cond, step = (x.strip() for x in pieces[:3])
            out = f"for ({init}; while {cond}; step {step}):"
            if tail:
                out += " " + pseudo_statement(tail)
            return out
        parsed_while = parse_parenthesized_control(s, "while")
        if parsed_while:
            cond, tail = parsed_while
            out = f"while {cond}:"
            if tail:
                out += " " + pseudo_statement(tail)
            return out
        m = re.match(r"do\s*$", s)
        if m:
            return "do:"
        parsed_if = parse_parenthesized_control(s, "if")
        if parsed_if:
            cond, tail = parsed_if
            out = f"if {cond}:"
            if tail:
                out += " " + pseudo_statement(tail)
            return out
        if s.startswith("else if"):
            parsed_else_if = parse_parenthesized_control(s[5:].lstrip(), "if")
            if parsed_else_if:
                cond, tail = parsed_else_if
                out = f"else if {cond}:"
                if tail:
                    out += " " + pseudo_statement(tail)
                return out
        if s == "else":
            return "else:"
        if s.startswith("else "):
            return "else: " + pseudo_statement(s[5:].strip())
        if s.startswith("return "):
            return s
        if s == "return":
            return "return"

        # Make obvious declarations read like algorithm variables, not compilable C++.
        s = re.sub(
            r"^(?:const\s+)?(?:std::)?(?:size_t|uint\d+_t|int\d+_t|int|long|short|float|double|bool|char|auto)\s+",
            "",
            s,
        )
        return s

    raw_lines = src.splitlines()
    header_accum: List[str] = []

    def next_nonempty_stripped(start_idx: int) -> str:
        for j in range(start_idx + 1, len(raw_lines)):
            nxt = raw_lines[j].strip()
            if nxt:
                return nxt
        return ""

    for line_idx, raw in enumerate(raw_lines):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith(("using ", "namespace ", "extern ")):
            continue
        if line.startswith("int main") or re.match(r"^(?:static\s+)?int\s+\w*main\s*\(", line):
            break

        # Preserve C++ initializer-list semantics before structural braces are stripped.
        line = re.sub(r"\breturn\s*\{\s*\}\s*;?", "return empty collection", line)
        line = re.sub(r"\breturn\s*\{([^{}]*)\}\s*;?", r"return [\1]", line)
        line = re.sub(r"=\s*\{\s*\}", "= empty collection", line)
        line = re.sub(r"=\s*\{([^{}]*)\}", r"= [\1]", line)

        # The serial entrypoint signature may span several lines. Emit it once
        # as the pseudocode header and skip the continuation lines; otherwise
        # parameters such as "JDIMENSION input_row, JSAMPARRAY output_buf,"
        # become bogus semantic statements in the feedback prompt.
        if not header_added:
            header_accum.append(line)
            if "{" not in line:
                continue
            combined_header = " ".join(header_accum)
            header_before_brace, _, header_after_brace = combined_header.partition("{")
            emit(header_before_brace.strip() + " {")
            block_extra_close_stack.append(0)
            indent += 1
            line = header_after_brace.strip()
            if not line:
                continue

        if line == "{":
            if pending_unbraced_blocks:
                block_extra_close_stack.append(max(0, pending_unbraced_blocks - 1))
                pending_unbraced_blocks = 0
            else:
                block_extra_close_stack.append(0)
                indent += 1
            continue

        while line.startswith("}"):
            extra_close = block_extra_close_stack.pop() if block_extra_close_stack else 0
            indent = max(0, indent - 1 - extra_close)
            line = line[1:].strip()
            if line.startswith("else"):
                break
        if not line:
            continue

        opens = line.count("{")
        closes = line.count("}")
        line = line.replace("{", "").replace("}", "").strip()
        stmt = pseudo_statement(line)
        if stmt:
            emit(stmt)
        is_unbraced_control = bool(stmt and stmt.endswith(":") and opens <= closes)
        if opens > closes:
            for _ in range(opens - closes):
                block_extra_close_stack.append(0)
            indent += opens - closes
        elif closes > opens:
            indent = max(0, indent - (closes - opens))
        if is_unbraced_control:
            indent += 1
            pending_unbraced_blocks += 1
        elif stmt and pending_unbraced_blocks and opens <= closes:
            next_line = next_nonempty_stripped(line_idx)
            keep_for_else = bool(
                (stmt.startswith("if ") or stmt.startswith("else if "))
                and ": " in stmt
                and next_line.startswith("else")
            )
            if not keep_for_else:
                indent = max(0, indent - pending_unbraced_blocks)
                pending_unbraced_blocks = 0

    if not lines and src:
        lines.append("function scalar_semantics:")
        for raw in src.splitlines()[:80]:
            line = pseudo_statement(raw)
            if line and not line.startswith("#"):
                lines.append("  " + line)

    out = "\n".join(lines).strip()
    if max_chars and max_chars > 0 and len(out) > int(max_chars):
        out = _truncate_middle(out, int(max_chars))
    return out


_CODE_LIKE_PSEUDOCODE_SCRIPT = Path("/home/user/selective_repo/scripts/build_simdbench_code_like_pseudocode.py")
_CODE_LIKE_PSEUDOCODE_MOD = None


def _load_code_like_pseudocode_module():
    global _CODE_LIKE_PSEUDOCODE_MOD
    if _CODE_LIKE_PSEUDOCODE_MOD is not None:
        return _CODE_LIKE_PSEUDOCODE_MOD
    if not _CODE_LIKE_PSEUDOCODE_SCRIPT.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            "build_simdbench_code_like_pseudocode_runtime",
            str(_CODE_LIKE_PSEUDOCODE_SCRIPT),
        )
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        _CODE_LIKE_PSEUDOCODE_MOD = mod
        return mod
    except Exception:
        return None


def _first_function_name_from_source(src: str) -> Optional[str]:
    text = strip_comments(str(src or ""))
    matches = list(_TOPLEVEL_FUNC_DEF_RE.finditer(text))
    if matches:
        return matches[0].group(1)
    return None


def build_dataflow_serial_reference_pseudocode(
    serial_ref_completion: str,
    *,
    func_name: Optional[str] = None,
    prompt_prefix: str = "",
    max_chars: int = 6000,
) -> str:
    """Build code-like scalar semantics with deterministic dataflow/DSL lifts.

    This path intentionally does not append natural-language SVE constraint
    prose. It only returns the code-like pseudocode produced by the shared
    builder, including conservative algebraic/dataflow rewrites such as
    store_map(...), sum(...), all(...), exists(...), prefix_scan(...).
    """
    src = str(serial_ref_completion or "").strip()
    if not src:
        return ""

    mod = _load_code_like_pseudocode_module()
    if mod is None:
        return build_serial_reference_pseudocode(
            serial_ref_completion,
            func_name=func_name,
            prompt_prefix=prompt_prefix,
            max_chars=max_chars,
        )

    target_sig = ""
    try:
        target_sig = target_signature_line_from_decl(
            extract_function_decl_from_prompt_prefix(prompt_prefix, func_name) or ""
        )
    except Exception:
        target_sig = ""

    body = ""
    scalar_header = ""
    candidate_names: List[str] = []
    if func_name:
        candidate_names.append(str(func_name))
    first_name = _first_function_name_from_source(src)
    if first_name and first_name not in candidate_names:
        candidate_names.append(first_name)
    try:
        for name in candidate_names:
            scalar_header, body = mod.extract_function(src, name)
            if scalar_header and body:
                break
    except Exception:
        scalar_header, body = "", ""
    if not body:
        return build_serial_reference_pseudocode(
            serial_ref_completion,
            func_name=func_name,
            prompt_prefix=prompt_prefix,
            max_chars=max_chars,
        )

    display_header = target_sig or scalar_header
    try:
        pseudocode = mod.build_code_like_pseudocode(display_header, body)
        out = pseudocode.strip()
    except Exception:
        return build_serial_reference_pseudocode(
            serial_ref_completion,
            func_name=func_name,
            prompt_prefix=prompt_prefix,
            max_chars=max_chars,
        )

    if max_chars and max_chars > 0 and len(out) > int(max_chars):
        out = _truncate_middle(out, int(max_chars))
    return out


def build_constrained_serial_reference_pseudocode(
    serial_ref_completion: str,
    *,
    func_name: Optional[str] = None,
    prompt_prefix: str = "",
    max_chars: int = 6000,
) -> str:
    """Backward-compatible alias for old run commands.

    Historically this appended natural-language API hint prose. That behavior is
    disabled; use --serial_feedback_style dataflow_pseudocode for the same
    deterministic DSL/dataflow output with a clearer name.
    """
    return build_dataflow_serial_reference_pseudocode(
        serial_ref_completion,
        func_name=func_name,
        prompt_prefix=prompt_prefix,
        max_chars=max_chars,
    )


def _serial_route_old_metadata_fallback(
    *,
    task_id: str = "",
    problem_type: str = "",
    problem_subtype: str = "",
    source_name: str = "",
    source_type: str = "",
    entrypoint_simd: str = "",
    prompt_prefix: str = "",
) -> Dict[str, str]:
    """Conservative fallback when no validated scalar reference is available."""
    tid = str(task_id or "")
    typ = str(problem_type or "").strip()
    subtype = str(problem_subtype or "").strip()
    src_name = str(source_name or "")
    src_type = str(source_type or "")
    entry = str(entrypoint_simd or "")
    prompt_text = str(prompt_prefix or "")
    key = " ".join([tid, entry, src_name, src_type, prompt_text[:2000]]).lower()

    if tid.startswith("SimdBench_") or typ in {"1", "2"} or subtype.startswith(("1-", "HumanEval/")):
        if typ == "1" or subtype.startswith("1-"):
            return {
                "style": "dataflow_pseudocode",
                "pattern": "metadata_numeric_kernel",
                "reason": "fallback: SimdBench type=1 numeric kernel",
            }
        if typ == "2" or subtype.startswith("HumanEval/"):
            return {
                "style": "pseudocode",
                "pattern": "metadata_humaneval_control",
                "reason": "fallback: SimdBench type=2/HumanEval-like task",
            }

    is_vecintrin = (
        tid.startswith("VecIntrinBench_")
        or src_name == "VecIntrinBench"
        or "vecintrinbench" in src_type.lower()
        or "vecintrinbench" in key
    )
    if is_vecintrin:
        dataflow_names = {
            "absval",
            "bias",
            "clip",
            "dropout",
            "eltwise",
            "flatten",
            "merge",
            "split",
            "packing",
            "batchnorm",
            "instancenorm",
            "innerproduct",
            "pooling",
            "relu",
            "prelu",
            "gelu",
            "mish",
            "swish",
            "sigmoid",
            "tanh",
            "selu",
            "hardsigmoid",
            "hardswish",
            "magnitude",
            "magnitude32f",
            "fastatan32f",
            "atan",
        }
        if any(name in key for name in dataflow_names):
            return {
                "style": "dataflow_pseudocode",
                "pattern": "metadata_vecintrin_pointwise",
                "reason": "fallback: VecIntrinBench pointwise/tensor/math name",
            }
        return {
            "style": "pseudocode",
            "pattern": "metadata_vecintrin_side_effect_or_macro",
            "reason": "fallback: VecIntrinBench side-effect/JPEG/color-style task",
        }

    control_keywords = [
        "string",
        "std::string",
        "to_string",
        "digit",
        "palindrome",
        "prime",
        "factor",
        "gcd",
        "dictionary",
        "map",
        "sort",
        "bracket",
        "vowel",
        "encrypt",
        "decode",
    ]
    if any(k in key for k in control_keywords):
        return {
            "style": "pseudocode",
            "pattern": "metadata_control_keyword",
            "reason": "fallback: control/string/number-theory keyword",
        }

    dataflow_keywords = [
        "matrix",
        "tensor",
        "vector",
        "array",
        "row",
        "column",
        "sum",
        "min",
        "max",
        "transpose",
        "normalize",
        "load",
        "store",
    ]
    if any(k in key for k in dataflow_keywords):
        return {
            "style": "dataflow_pseudocode",
            "pattern": "metadata_dataflow_keyword",
            "reason": "fallback: array/matrix/reduction keyword",
        }

    return {
        "style": "pseudocode",
        "pattern": "metadata_default",
        "reason": "fallback: no scalar pattern available",
    }


def _extract_serial_route_body(
    serial_ref_source: str,
    *,
    func_name: Optional[str] = None,
    prompt_prefix: str = "",
) -> Tuple[str, str]:
    """Return (display_header, body) for route classification."""
    src = str(serial_ref_source or "").strip()
    if not src:
        return "", ""

    mod = _load_code_like_pseudocode_module()
    target_sig = ""
    try:
        target_sig = target_signature_line_from_decl(
            extract_function_decl_from_prompt_prefix(prompt_prefix, func_name) or ""
        )
    except Exception:
        target_sig = ""

    candidate_names: List[str] = []
    if func_name:
        candidate_names.append(str(func_name))
    first_name = _first_function_name_from_source(src)
    if first_name and first_name not in candidate_names:
        candidate_names.append(first_name)

    if mod is not None:
        try:
            for name in candidate_names:
                scalar_header, body = mod.extract_function(src, name)
                if scalar_header and body:
                    return target_sig or scalar_header, body
        except Exception:
            pass

    # Fallback to the local extractor if the standalone builder is unavailable.
    try:
        for name in candidate_names:
            body = _extract_function_definition(src, name)
            if body:
                lb = body.find("{")
                rb = body.rfind("}")
                header = body[:lb].strip() if lb >= 0 else ""
                inner = body[lb + 1 : rb] if lb >= 0 and rb > lb else body
                return target_sig or normalize_ws(header), inner
    except Exception:
        pass
    return target_sig, src


def _build_route_dataflow_preview(display_header: str, body: str) -> str:
    mod = _load_code_like_pseudocode_module()
    if mod is None or not body:
        return ""
    try:
        return str(mod.build_code_like_pseudocode(display_header or "scalar_semantics", body) or "")
    except Exception:
        return ""


def _route_expr_norm(expr: str) -> str:
    s = re.sub(r"\s+", "", str(expr or ""))
    changed = True
    while changed and s.startswith("(") and s.endswith(")"):
        changed = False
        depth = 0
        ok = True
        for idx, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx != len(s) - 1:
                    ok = False
                    break
        if ok:
            s = s[1:-1]
            changed = True
    return s


def _route_array_refs(expr: str) -> List[Tuple[str, str]]:
    """Return array references as (array_name, index_expr) from a C-ish expr."""
    text = str(expr or "")
    refs: List[Tuple[str, str]] = []
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\[", text):
        lbr = text.find("[", m.end() - 1)
        if lbr < 0:
            continue
        depth = 0
        end = -1
        for pos in range(lbr, len(text)):
            ch = text[pos]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = pos
                    break
        if end >= 0:
            refs.append((m.group(1), text[lbr + 1 : end].strip()))
    return refs


def _route_iter_indexed_assignments(body: str) -> List[Tuple[int, str, str]]:
    """Find simple indexed assignments and return (position, lhs, rhs)."""
    out: List[Tuple[int, str, str]] = []
    text = str(body or "")
    # This intentionally ignores declarations/comparisons and keeps only
    # statements whose left-hand side is an indexed array write.
    pat = re.compile(
        r"(?P<lhs>\b[A-Za-z_]\w*\s*\[[^\n;=]+?\])\s*=\s*(?!=)(?P<rhs>[^;]+);",
        re.MULTILINE,
    )
    for m in pat.finditer(text):
        out.append((m.start(), m.group("lhs").strip(), m.group("rhs").strip()))
    return out


def _route_recent_loop_context(body: str, pos: int) -> str:
    """Small local context before an assignment, enough for loop bounds/steps."""
    start = max(0, int(pos) - 900)
    return str(body or "")[start:int(pos)]


def _route_has_loop_start_step(ctx: str, var: str, *, start: str = "", step2: bool = False) -> bool:
    v = re.escape(var)
    s = str(ctx or "")
    if start:
        start_re = re.escape(start)
        if not re.search(rf"for\s*\([^;]*\b{v}\s*=\s*{start_re}\b", s):
            return False
    if step2 and not re.search(rf"\b{v}\s*(?:\+=\s*2|\+\+)", s):
        return False
    return True


def _route_rhs_is_exact_anchor(arr: str, rhs: str, anchor_idx: str) -> bool:
    return _route_expr_norm(rhs) == f"{arr}[{_route_expr_norm(anchor_idx)}]"


def _route_is_known_nonoverlap_same_array_read(
    *,
    arr: str,
    lhs_idx: str,
    rhs_idx: str,
    rhs: str,
    ctx: str,
) -> bool:
    """Recognize a few provably non-overlapping TSVC-style read/write regions."""
    lhs = _route_expr_norm(lhs_idx)
    ridx = _route_expr_norm(rhs_idx)
    compact_ctx = _route_expr_norm(ctx)

    if lhs == ridx:
        return True

    # Odd/even split: for (i = 1; i < n; i += 2) a[i] reads a[i-1].
    if lhs == "i" and ridx in {"i-1", "(i)-1"} and re.search(r"for\s*\([^;]*\bi\s*=\s*1\b[^;]*;[^;]*;[^)]*\bi\s*\+=\s*2", ctx):
        return True

    # Fixed first element is not overwritten by loops that start from i=1.
    if lhs == "i" and ridx == "0" and re.search(r"for\s*\([^;]*\bi\s*=\s*1\b", ctx):
        return True

    # Exact self-copy anchor: a[i] = a[0] with i starting at 0 leaves a[0] unchanged.
    if lhs == "i" and ridx == "0" and _route_rhs_is_exact_anchor(arr, rhs, "0"):
        return True

    # Split halves: for (i = 0; i < M; ++i) a[i+M] reads a[i].
    if lhs in {"i+M", "M+i"} and ridx == "i" and re.search(r"for\s*\([^)]*\bi\s*=\s*0\b[^)]*\bi\s*<\s*M\b", ctx):
        return True

    # Lower-triangle write reads the symmetric upper-triangle element.
    if "j<i" in compact_ctx:
        lhs_tri = re.sub(r"[()]", "", lhs)
        rhs_tri = re.sub(r"[()]", "", ridx)
        if "i*len2+j" in lhs_tri and "j*len2+i" in rhs_tri:
            return True

    return False


def _route_same_array_dependency_guard(body: str) -> Optional[Dict[str, str]]:
    """Detect same-array cross-element dependencies before treating loops as maps.

    A store-map lift is only valid when each written element is independent, or
    when read/write regions are provably disjoint.  TSVC contains many loops
    that look like indexed stores but are actually recurrences, stencils, or
    direction-sensitive shifted overwrites.
    """
    text = str(body or "")
    for pos, lhs, rhs in _route_iter_indexed_assignments(text):
        lhs_refs = _route_array_refs(lhs)
        if not lhs_refs:
            continue
        lhs_arr, lhs_idx = lhs_refs[0]
        ctx = _route_recent_loop_context(text, pos)
        for rhs_arr, rhs_idx in _route_array_refs(rhs):
            if rhs_arr != lhs_arr:
                continue
            if _route_is_known_nonoverlap_same_array_read(
                arr=lhs_arr,
                lhs_idx=lhs_idx,
                rhs_idx=rhs_idx,
                rhs=rhs,
                ctx=ctx,
            ):
                continue

            lhs_n = _route_expr_norm(lhs_idx)
            rhs_n = _route_expr_norm(rhs_idx)
            evidence = f"{lhs_arr}[{lhs_idx}] reads {rhs_arr}[{rhs_idx}]"
            lower_evidence = evidence.lower()
            if re.search(r"[ij]\s*[+-]\s*\d|[+-]\s*[ij]", rhs_idx) or re.search(r"\b[a-z]\s*[+-]\s*(?:k|m|inc)\b", rhs_n):
                pattern = "recurrence_or_prefix_dependency"
                if any(token in lower_evidence for token in ("j-1", "i-1", "i+1", "j+1")) and ("len2" in lower_evidence or lhs_arr in {"aa", "bb", "cc", "tt"}):
                    pattern = "stencil_or_wavefront_dependency"
                elif re.search(r"\+\s*(?:k|m|inc)|-\s*(?:k|m|inc)", rhs_idx, flags=re.IGNORECASE) or re.search(
                    r"\+\s*(?:k|m|inc)|-\s*(?:k|m|inc)", lhs_idx, flags=re.IGNORECASE
                ):
                    pattern = "overlap_shifted_store_dependency"
            elif re.search(r"\b(?:len1|len2)\b|^\d+$", rhs_n):
                pattern = "anchored_broadcast_write_dependency"
            else:
                pattern = "overlap_shifted_store_dependency"

            return {
                "style": "pseudocode",
                "pattern": pattern,
                "reason": (
                    "dependency guard: indexed assignment reads and writes the same array "
                    f"at different positions ({evidence}); do not treat it as independent store_map"
                ),
            }
    return None


def _route_add_feature(features: List[str], feature: str) -> None:
    if feature and feature not in features:
        features.append(feature)


def _route_collect_semantic_features(
    *,
    display_header: str,
    clean_body: str,
    preview: str,
) -> List[str]:
    """Collect composable scalar semantics for routed serial feedback.

    These tags are deliberately feature-like rather than task-like.  A single
    kernel may be row-pointer + edge-stencil + fixed-point + interleaved-store;
    treating the first matched keyword as the whole route is exactly how JPEG
    and color conversion tasks were previously mislabeled as generic bit shifts.
    """
    features: List[str] = []
    text = str(clean_body or "")
    text_l = text.lower()
    preview_l = str(preview or "").lower()
    all_text = "\n".join([str(display_header or ""), text, str(preview or "")])
    all_l = all_text.lower()

    if re.search(r"\b(?:JSAMPARRAY|JSAMPIMAGE|JSAMPROW|JSAMPLE|JDIMENSION)\b", all_text) or re.search(
        r"\b(?:input_buf|output_buf|inptr|outptr|inrow|outrow)\b", all_l
    ):
        _route_add_feature(features, "row_pointer_layout")

    if re.search(
        r"\b(?:RGB_RED|RGB_GREEN|RGB_BLUE|RGB_PIXELSIZE|BGR|YUV|YCC|YCbCr|Cr|Cb|blueIdx|scn|srccn|dstcn|cn)\b",
        all_text,
        flags=re.IGNORECASE,
    ) or re.search(r"\b(?:src|dst|inptr|outptr)\s*(?:\+=|-=)\s*(?:3|4|cn|scn|srccn|dstcn)\b", text):
        _route_add_feature(features, "interleaved_channel_map")

    has_neighbor_ref = bool(re.search(
        r"\[[^\]]*(?:[A-Za-z_]\w*)\s*[+-]\s*1[^\]]*\]|\[[^\]]*(?:[+-]\s*1)\s*[+-]\s*[A-Za-z_]\w*[^\]]*\]",
        text,
    ))
    has_edge_words = bool(re.search(r"\b(?:upsample|downsample|fancy|edge|border|padding)\b", all_l))
    if has_edge_words or (
        has_neighbor_ref
        and (
            "row_pointer_layout" in features
            or re.search(r"\b(?:stencil|rolling|prefix|scan|window|neighbor)\b", all_l)
            or re.search(r"\b[A-Za-z_]\w*\s*\[\s*[A-Za-z_]\w*\s*\]\s*=", text)
        )
    ):
        _route_add_feature(features, "edge_stencil_or_upsample")

    has_shift = bool(re.search(r">>|<<", text))
    has_fixed_point_markers = bool(
        re.search(r"\b(?:SCALEBITS|ONE_HALF|FIX|DESCALE|RANGE_LIMIT|CENTERJSAMPLE|MAXJSAMPLE)\b", all_text)
    )
    has_clamp = bool(
        re.search(r"\b(?:min|max|std::min|std::max|clip|clamp)\s*\(", text_l)
        or re.search(r"\b(?:if|else if)\s*\([^)]*[<>]=?\s*(?:0|255|MAXJSAMPLE|RANGE_LIMIT)", text)
    )
    if (has_shift and re.search(r"\*", text) and (has_clamp or has_fixed_point_markers)) or has_fixed_point_markers:
        _route_add_feature(features, "fixed_point_numeric_map")

    if re.search(r"\b(?:switch|case)\b", text) and re.search(r"\b(?:SUM|PROD|MAX|MIN|Operation_|op|operation)\b", all_text):
        _route_add_feature(features, "branch_accumulate")

    if re.search(r"\b(?:fdct|idct|dctsize|dctelem|workspace|z1|z2|butterfly)\b", all_l):
        _route_add_feature(features, "fixed_point_butterfly")

    if "prefix_scan(" in preview_l:
        _route_add_feature(features, "scan_or_prefix_monoid")
    if re.search(r"\b(?:sum|min|max|product)\s*\(", preview_l):
        _route_add_feature(features, "reduction_or_two_pass_statistic")
    if re.search(r"\b(?:all|exists|argmax_index|max_index|argmin_pair)\s*\(", preview_l):
        _route_add_feature(features, "predicate_or_arg_reduction")
    if "store_map(" in preview_l:
        _route_add_feature(features, "store_map")

    # Only call it a bit-shift/rotate route when it is not really fixed-point
    # arithmetic, row-pointer image code, channel packing, or edge interpolation.
    bit_token = r"(?:\^|~|(?<!\|)\|(?!\|)|(?<!&)&(?!&)|>>|<<)"
    indexed_write = bool(re.search(r"\b[A-Za-z_]\w*\s*\[[^\]]+\]\s*=", text))
    bitwise_expr = bool(
        re.search(rf"\[[^\]]+\].*{bit_token}|{bit_token}.*\[[^\]]+\]", text)
        or (indexed_write and re.search(bit_token, text))
    )
    rotate_expr = bool(
        re.search(r"\(\s*[^()]+\s*(?:<<|>>)\s*[^()]+\)\s*\|\s*\(\s*[^()]+\s*(?:>>|<<)\s*[^()]+\)", text)
        or re.search(r"\b(?:rot|rotate|ror|rol)\b", all_l)
    )
    if (bitwise_expr or rotate_expr) and not any(
        f in features
        for f in {
            "row_pointer_layout",
            "interleaved_channel_map",
            "edge_stencil_or_upsample",
            "fixed_point_numeric_map",
            "fixed_point_butterfly",
        }
    ):
        _route_add_feature(features, "bit_shift_or_rotate")

    return features


def _route_composite_pattern(features: List[str]) -> str:
    priority = [
        "branch_accumulate",
        "row_pointer_layout",
        "interleaved_channel_map",
        "edge_stencil_or_upsample",
        "fixed_point_butterfly",
        "fixed_point_numeric_map",
        "scan_or_prefix_monoid",
        "reduction_or_two_pass_statistic",
        "predicate_or_arg_reduction",
        "store_map",
        "bit_shift_or_rotate",
    ]
    ordered = [f for f in priority if f in set(features)]
    return "+".join(ordered[:4]) if ordered else ""


def _classify_serial_pattern_from_source(
    serial_ref_source: str,
    *,
    func_name: Optional[str] = None,
    prompt_prefix: str = "",
) -> Dict[str, str]:
    """Classify the scalar reference into a bootstrap style by AST-like shape.

    This does not route by benchmark ID. It uses the scalar function body and
    deterministic code-like/dataflow lifts to distinguish map/index transforms,
    reductions, predicate reductions, scans, compact outputs, affine
    recurrences, conversions, and sequential decimal/string algorithms.
    """
    display_header, body = _extract_serial_route_body(
        serial_ref_source,
        func_name=func_name,
        prompt_prefix=prompt_prefix,
    )
    clean_body = strip_comments(body or "")
    if not clean_body.strip():
        return {
            "style": "",
            "pattern": "no_scalar_body",
            "reason": "no validated scalar body to classify",
        }

    preview = _build_route_dataflow_preview(display_header, clean_body)
    body_l = clean_body.lower()
    preview_l = preview.lower()
    all_l = "\n".join([display_header or "", clean_body, preview]).lower()
    dependency_guard_route = _route_same_array_dependency_guard(clean_body)

    # Hard sequential/control cases. These should keep code-shaped scalar
    # pseudocode unless a more specific vectorizable lift below fires first.
    has_string_container = bool(
        re.search(r"\bstd::string\b|\bstring\b|std::vector\s*<\s*std::string", all_l)
    )
    has_decimal_loop = bool(
        re.search(r"%\s*10\b|/\s*10\b|/=\s*10\b|\bto_string\s*\(|\bstoi\s*\(", all_l)
    )
    has_number_theory = bool(
        re.search(r"\bprime\b|\bfactor\b|\bgcd\b|\bfactorial\b|\bhappy\b|\bbase\b", all_l)
    )
    has_container_mutation = bool(
        re.search(r"\.push_back\s*\(|\.erase\s*\(|\.insert\s*\(|std::sort\s*\(|\bsort\s*\(", all_l)
    )
    conversion_scan_body = re.sub(
        r"\breturn\s+\(\s*float\s*\)\s*\(?\s*0(?:\.0f?|f)?\s*\)?\s*;",
        "",
        clean_body,
    )
    float_numeric_cast_re = (
        r"\bstatic_cast\s*<\s*float\s*>\s*\(\s*[A-Za-z_*(]"
        r"|\(\s*float\s*\)\s*(?!\(?\s*[-+]?(?:\d|\.\d))\(?\s*[A-Za-z_*]"
    )
    feature_tags = _route_collect_semantic_features(
        display_header=display_header,
        clean_body=clean_body,
        preview=preview,
    )

    # Direct high-risk/vectorizable recognizers get first chance so the route
    # metadata names the actual issue instead of a generic store_map fallback.
    if re.search(
        r"([A-Za-z_]\w*)\s*\[\s*0\s*\]\s*=[^;]+;\s*for\s*\([^;]*\b([A-Za-z_]\w*)\s*=\s*1\b[\s\S]{0,400}?\1\s*\[\s*\2\s*\]\s*=\s*\1\s*\[\s*\2\s*-\s*1\s*\]\s*[+-]",
        clean_body,
    ):
        return {
            "style": "dataflow_pseudocode",
            "pattern": "affine_recurrence_closed_form",
            "features": ",".join(feature_tags),
            "reason": "scalar body is a first-order affine recurrence that can be expressed per index",
        }
    if re.search(
        r"while\s*\([^)]*<[^)]*\)\s*\{[\s\S]{0,500}?\w+\s*\[\s*\w+\s*\]\s*=\s*\w+\s*\[\s*\w+\s*\][\s\S]{0,300}?\+\+[\s\S]{0,100}?--",
        clean_body,
    ):
        return {
            "style": "dataflow_pseudocode",
            "pattern": "reverse_or_two_pointer_swap",
            "features": ",".join(feature_tags),
            "reason": "scalar body is a converging two-pointer reverse/swap",
        }
    if re.search(float_numeric_cast_re, conversion_scan_body) and re.search(
        r"\bdouble\b|\bfloat\b|\bint(?:8|16|32|64)?_t\b|\buint(?:8|16|32|64)?_t\b",
        all_l,
    ):
        return {
            "style": "dataflow_pseudocode",
            "pattern": "numeric_conversion_lane_mapping",
            "features": ",".join(feature_tags),
            "reason": "scalar body contains numeric conversion/narrowing over typed data",
        }
    if re.search(r"\broundf?\s*\(|\bceilf?\s*\(|\bfloorf?\s*\(|\bnearbyintf?\s*\(|\brintf?\s*\(", body_l):
        return {
            "style": "dataflow_pseudocode",
            "pattern": "rounding_exact",
            "features": ",".join(feature_tags),
            "reason": "scalar body contains explicit rounding-mode operation",
        }

    if has_string_container:
        return {
            "style": "pseudocode",
            "pattern": "string_container_or_text_algorithm",
            "features": ",".join(feature_tags),
            "reason": "scalar body uses string/container text semantics",
        }

    composite_pattern = _route_composite_pattern(feature_tags)
    high_priority_features = {
        "branch_accumulate",
        "row_pointer_layout",
        "interleaved_channel_map",
        "fixed_point_butterfly",
        "fixed_point_numeric_map",
    }
    edge_composite = "edge_stencil_or_upsample" in feature_tags and any(
        f in feature_tags
        for f in {
            "store_map",
            "predicate_or_arg_reduction",
            "reduction_or_two_pass_statistic",
            "row_pointer_layout",
            "interleaved_channel_map",
            "fixed_point_numeric_map",
        }
    )
    if any(f in feature_tags for f in high_priority_features) or edge_composite:
        if dependency_guard_route is not None and not any(f in feature_tags for f in high_priority_features):
            return dependency_guard_route
        return {
            "style": "dataflow_pseudocode",
            "pattern": composite_pattern or "composite_dataflow_kernel",
            "features": ",".join(feature_tags),
            "reason": "scalar body has composable layout/kernel features; do not collapse it to generic store_map or bit_shift",
        }
    if "bit_shift_or_rotate" in feature_tags:
        return {
            "style": "dataflow_pseudocode",
            "pattern": "bit_shift_or_rotate",
            "features": ",".join(feature_tags),
            "reason": "scalar body contains true bitwise/rotate array expression without higher-priority layout/fixed-point features",
        }

    # Strong vectorizable/dataflow lifts produced from the scalar body.
    if "prefix_scan(" in preview_l:
        return {
            "style": "dataflow_pseudocode",
            "pattern": "scan_or_prefix_monoid",
            "reason": "scalar body lifts to prefix_scan(...)",
        }
    if "argmin_pair(" in preview_l:
        return {
            "style": "dataflow_pseudocode",
            "pattern": "pair_domain_argmin",
            "reason": "scalar body lifts to pair-domain argmin over candidate pairs",
        }
    if preview_l.count("exists(") >= 2 and preview_l.count(" in range(") >= 2:
        return {
            "style": "dataflow_pseudocode",
            "pattern": "pair_domain_exists",
            "reason": "scalar body lifts to nested exists over candidate pairs",
        }
    if "store_map(" in preview_l:
        if dependency_guard_route is not None:
            return dependency_guard_route
        return {
            "style": "dataflow_pseudocode",
            "pattern": "map_or_index_transform",
            "features": ",".join(feature_tags),
            "reason": "scalar body lifts to store_map(...), i.e. independent index/write mapping",
        }
    if re.search(r"\b(?:sum|min|max|product)\s*\(", preview_l):
        return {
            "style": "dataflow_pseudocode",
            "pattern": "reduction_or_two_pass_statistic",
            "features": ",".join(feature_tags),
            "reason": "scalar body lifts to compact reduction expression",
        }
    if re.search(r"\b(?:all|exists|argmax_index|max_index|argmin_pair)\s*\(", preview_l):
        return {
            "style": "dataflow_pseudocode",
            "pattern": "predicate_or_arg_reduction",
            "features": ",".join(feature_tags),
            "reason": "scalar body lifts to all/exists/index reduction expression",
        }
    if "select(" in preview_l and re.search(r"\[[^\]]+\]\s*=", preview_l):
        return {
            "style": "dataflow_pseudocode",
            "pattern": "predicate_select_store",
            "features": ",".join(feature_tags),
            "reason": "scalar body lifts to select(...) feeding indexed store",
        }

    if has_string_container:
        return {
            "style": "pseudocode",
            "pattern": "string_container_or_text_algorithm",
            "features": ",".join(feature_tags),
            "reason": "scalar body uses string/container text semantics",
        }

    if re.search(
        r"([A-Za-z_]\w*)\s*\[\s*0\s*\]\s*=[^;]+;\s*for\s*\([^;]*\b([A-Za-z_]\w*)\s*=\s*1\b[\s\S]{0,400}?\1\s*\[\s*\2\s*\]\s*=\s*\1\s*\[\s*\2\s*-\s*1\s*\]\s*[+-]",
        clean_body,
    ):
        return {
            "style": "dataflow_pseudocode",
            "pattern": "affine_recurrence_closed_form",
            "reason": "scalar body is a first-order affine recurrence that can be expressed per index",
        }
    if re.search(r"\[[^\]]*\+\+\s*\]\s*=", clean_body):
        return {
            "style": "dataflow_pseudocode",
            "pattern": "compact_filter_output",
            "reason": "scalar body packs selected outputs with an incrementing write index",
        }

    # Conversion/widen/rounding/bit formulas are dataflow-like even when the
    # standalone preview could not compress them.
    if re.search(float_numeric_cast_re, conversion_scan_body) and re.search(
        r"\bdouble\b|\bfloat\b|\bint(?:8|16|32|64)?_t\b|\buint(?:8|16|32|64)?_t\b",
        all_l,
    ):
        return {
            "style": "dataflow_pseudocode",
            "pattern": "numeric_conversion_lane_mapping",
            "features": ",".join(feature_tags),
            "reason": "scalar body contains numeric conversion/narrowing over typed data",
        }
    if re.search(r"\broundf?\s*\(|\bceilf?\s*\(|\bfloorf?\s*\(|\bnearbyintf?\s*\(|\brintf?\s*\(", body_l):
        return {
            "style": "dataflow_pseudocode",
            "pattern": "rounding_exact",
            "features": ",".join(feature_tags),
            "reason": "scalar body contains explicit rounding-mode operation",
        }
    if "bit_shift_or_rotate" in feature_tags:
        return {
            "style": "dataflow_pseudocode",
            "pattern": "bit_shift_or_rotate",
            "features": ",".join(feature_tags),
            "reason": "scalar body contains true bitwise/rotate array expression without higher-priority layout/fixed-point features",
        }
    if re.search(
        r"\bfor\s*\([^)]*\)\s*\{?[\s\S]{0,800}?[A-Za-z_]\w*\s*\[[^\]]+\]\s*=",
        clean_body,
    ):
        if dependency_guard_route is not None:
            return dependency_guard_route
        return {
            "style": "dataflow_pseudocode",
            "pattern": "map_or_index_transform",
            "features": ",".join(feature_tags),
            "reason": "scalar body assigns indexed outputs inside loop nests",
        }

    # Sequential-ish cases after stronger vectorizable lifts have had a chance.
    if has_decimal_loop:
        return {
            "style": "pseudocode",
            "pattern": "decimal_digit_loop",
            "reason": "scalar body uses base-10 digit extraction/conversion",
        }
    if has_number_theory:
        return {
            "style": "pseudocode",
            "pattern": "number_theory_or_scalar_algorithm",
            "reason": "scalar body is scalar number-theory/control-flow style",
        }
    if has_container_mutation:
        return {
            "style": "pseudocode",
            "pattern": "container_mutation",
            "reason": "scalar body mutates variable-size containers",
        }

    # Nested search with early return is usually better as explicit pair-domain
    # semantics than a line-by-line loop dump.
    if len(re.findall(r"\bfor\s*\(", clean_body)) >= 2 and re.search(r"return\s+(?:true|false)\s*;", clean_body):
        return {
            "style": "dataflow_pseudocode",
            "pattern": "outer_scalar_inner_domain_search",
            "reason": "nested early-return search can be expressed as exists/all over a candidate domain",
        }

    # Default to dataflow for simple array loops, pseudocode for remaining
    # scalar-only/control code.
    if re.search(r"\[[^\]]+\]", clean_body) and re.search(r"\bfor\s*\(", clean_body):
        return {
            "style": "dataflow_pseudocode",
            "pattern": "array_loop_default",
            "reason": "scalar body is an array loop without recognized sequential hazards",
        }

    return {
        "style": "pseudocode",
        "pattern": "scalar_control_default",
        "reason": "no vectorizable dataflow pattern recognized",
    }


def classify_serial_feedback_route(
    requested_style: str,
    *,
    task_id: str = "",
    problem_type: str = "",
    problem_subtype: str = "",
    source_name: str = "",
    source_type: str = "",
    entrypoint_simd: str = "",
    prompt_prefix: str = "",
    serial_ref_source: str = "",
) -> Dict[str, str]:
    """Resolve routed/auto SERIAL feedback style with scalar-pattern routing."""
    style = str(requested_style or "pseudocode").strip().lower()
    aliases = {
        "auto": "routed",
        "route": "routed",
        "routing": "routed",
        "routed_by_type": "routed",
        "type_routed": "routed",
    }
    style = aliases.get(style, style)
    if style != "routed":
        return {
            "style": style or "pseudocode",
            "pattern": "explicit_style",
            "reason": "serial_feedback_style is not routed/auto",
        }

    if str(serial_ref_source or "").strip():
        route = _classify_serial_pattern_from_source(
            serial_ref_source,
            func_name=entrypoint_simd or None,
            prompt_prefix=prompt_prefix,
        )
        if route.get("style") in {"pseudocode", "dataflow_pseudocode"}:
            route["source"] = "scalar_ast_pattern"
            return route

    route = _serial_route_old_metadata_fallback(
        task_id=task_id,
        problem_type=problem_type,
        problem_subtype=problem_subtype,
        source_name=source_name,
        source_type=source_type,
        entrypoint_simd=entrypoint_simd,
        prompt_prefix=prompt_prefix,
    )
    route["source"] = "metadata_fallback"
    return route


def resolve_serial_feedback_style(
    requested_style: str,
    *,
    task_id: str = "",
    problem_type: str = "",
    problem_subtype: str = "",
    source_name: str = "",
    source_type: str = "",
    entrypoint_simd: str = "",
    prompt_prefix: str = "",
    serial_ref_source: str = "",
) -> str:
    """Resolve routed/auto SERIAL feedback style to a concrete prompt style."""
    return classify_serial_feedback_route(
        requested_style,
        task_id=task_id,
        problem_type=problem_type,
        problem_subtype=problem_subtype,
        source_name=source_name,
        source_type=source_type,
        entrypoint_simd=entrypoint_simd,
        prompt_prefix=prompt_prefix,
        serial_ref_source=serial_ref_source,
    ).get("style", "pseudocode") or "pseudocode"


def build_serial_mismatch_harness_prompt(
    *,
    spec_text: str,
    prompt_prefix: str,
    func_name: Optional[str],
    serial_ref_completion: str,
    simd_completion: str,
) -> str:
    """Prompt for generating a deterministic mismatch harness comparing serial vs SIMD outputs."""
    def _rename_entry(code: str, old: Optional[str], new: str) -> str:
        if not code or not old:
            return code or ""
        return re.sub(rf"\b{re.escape(old)}\b", new, code)

    parts: List[str] = []
    parts.append("You will write a standalone C++17 test harness to compare TWO implementations.")
    parts.append("")
    parts.append("HARD RULES:")
    parts.append("1) Output ONLY a single C++ source file. No markdown.")
    parts.append("2) The harness must be deterministic (no random unless fixed seed).")
    parts.append("3) MUST include a valid int main() entry point.")
    parts.append("4) Print exactly ONE JSON line to stdout as the LAST line (no other prints).")
    parts.append("5) JSON keys: mismatch(bool), index(int), expected(string/number), got(string/number), note(string), mismatch_examples(array).")
    parts.append("6) mismatch_examples must contain up to the FIRST 16 mismatches, each as {index, expected, got, note}.")
    parts.append("7) If no mismatch, set mismatch=false, index=-1, and mismatch_examples=[].")
    parts.append("8) If you cannot compare, set mismatch=true, index=-1, note with the reason.")
    parts.append("")
    parts.append("Task:")
    parts.append("- Embed BOTH implementations in the same file.")
    parts.append("- The code blocks provided below are ALREADY renamed:")
    parts.append("  * serial version => <FUNC>_serial")
    parts.append("  * simd version   => <FUNC>_simd")
    parts.append("  Do NOT rename them again.")
    parts.append("- Build a small fixed input (size <= 16) consistent with the SPEC.")
    parts.append("- Call both versions and compare outputs.")
    parts.append("- Report the FIRST mismatch index/value.")
    parts.append("- Also include up to the FIRST 16 mismatch examples if there is more than one mismatch.")
    parts.append("")
    parts.append("Required output shape (example):")
    parts.append("int main(){")
    parts.append("  // build inputs")
    parts.append("  auto exp = <FUNC>_serial(...);")
    parts.append("  auto got = <FUNC>_simd(...);")
    parts.append("  if (mismatch) std::cout << \"{\\\"mismatch\\\":true,\\\"index\\\":0,\\\"expected\\\":...,\\\"got\\\":...,\\\"note\\\":\\\"...\\\",\\\"mismatch_examples\\\":[{...}]}\" << std::endl;")
    parts.append("  else std::cout << \"{\\\"mismatch\\\":false,\\\"index\\\":-1,\\\"expected\\\":0,\\\"got\\\":0,\\\"note\\\":\\\"ok\\\",\\\"mismatch_examples\\\":[]}\" << std::endl;")
    parts.append("  return 0;")
    parts.append("}")
    parts.append("")
    if func_name:
        parts.append(f"TARGET FUNCTION NAME: {func_name}")
        parts.append("")

    def _extract_spec_block(text: str, name: str) -> str:
        if not text:
            return ""
        m = re.search(
            rf"(?s)\[{re.escape(name)}\]\s*(.*?)(?=\n\[[A-Z_]+\]|\Z)",
            text,
        )
        return m.group(1).strip() if m else ""

    task_blk = _extract_spec_block(spec_text or "", "TASK")
    if task_blk:
        parts.append("[TASK_SUMMARY]")
        parts.append(task_blk)
        parts.append("[/TASK_SUMMARY]")
        parts.append("")
    parts.append("[SERIAL_REFERENCE_CODE_RENAMED]")
    parts.append(_rename_entry(serial_ref_completion or "", func_name, f"{func_name}_serial"))
    parts.append("[/SERIAL_REFERENCE_CODE_RENAMED]")
    parts.append("")
    parts.append("[SIMD_COMPLETION_CODE_RENAMED]")
    parts.append(_rename_entry(simd_completion or "", func_name, f"{func_name}_simd"))
    parts.append("[/SIMD_COMPLETION_CODE_RENAMED]")
    parts.append("")
    parts.append("Now output the harness code:")
    return "\n".join(parts)

def build_serial_mismatch_harness_prompt_minimal(
    *,
    func_name: Optional[str],
    serial_ref_completion: str,
    simd_completion: str,
) -> str:
    """Minimal fallback prompt for mismatch harness (kept short to avoid empty outputs)."""
    def _rename_entry(code: str, old: Optional[str], new: str) -> str:
        if not code or not old:
            return code or ""
        return re.sub(rf"\b{re.escape(old)}\b", new, code)

    parts: List[str] = []
    parts.append("Write ONE C++17 source file.")
    parts.append("MUST include int main() and print ONE JSON line as the LAST line.")
    parts.append("JSON keys: mismatch, index, expected, got, note, mismatch_examples.")
    parts.append("Include up to the first 16 mismatches in mismatch_examples.")
    parts.append("Use the two functions below (already renamed). Do NOT rename again.")
    parts.append("")
    if func_name:
        parts.append(f"TARGET FUNCTION NAME: {func_name}")
        parts.append("")
    parts.append("[SERIAL_REFERENCE_CODE_RENAMED]")
    parts.append(_rename_entry(serial_ref_completion or "", func_name, f"{func_name}_serial"))
    parts.append("[/SERIAL_REFERENCE_CODE_RENAMED]")
    parts.append("")
    parts.append("[SIMD_COMPLETION_CODE_RENAMED]")
    parts.append(_rename_entry(simd_completion or "", func_name, f"{func_name}_simd"))
    parts.append("[/SIMD_COMPLETION_CODE_RENAMED]")
    parts.append("")
    parts.append("Now output the harness code:")
    return "\n".join(parts)


_SERIAL_MISMATCH_HARNESS_REQUIRED_INCLUDES = [
    "#include <iostream>",
    "#include <sstream>",
    "#include <string>",
    "#include <vector>",
    "#include <cstdint>",
    "#include <cstddef>",
    "#include <cmath>",
    "#include <cfloat>",
    "#include <float.h>",
    "#include <climits>",
    "#include <limits>",
    "#include <algorithm>",
]


def ensure_serial_mismatch_harness_preamble(source: str) -> str:
    """Add only harness-level standard includes before serial/SIMD comparison code.

    This is deliberately scoped to the serial mismatch harness path. It must not
    normalize or rewrite the candidate SVE completion; it only makes the
    standalone comparison harness compile when the serial reference uses standard
    constants such as FLT_MAX before the SIMD block's own includes appear.
    """
    s = strip_noncode(source or "")
    if not s.strip():
        return s
    return "\n".join(_SERIAL_MISMATCH_HARNESS_REQUIRED_INCLUDES) + "\n\n" + s


def try_build_auto_mismatch_harness(
    *,
    func_name: Optional[str],
    serial_ref_completion: str,
    simd_completion: str,
) -> Optional[str]:
    """Best-effort auto harness when LLM fails to include main()."""
    if not func_name:
        return None

    def _rename_entry(code: str, old: Optional[str], new: str) -> str:
        if not code or not old:
            return code or ""
        return re.sub(rf"\b{re.escape(old)}\b", new, code)

    def _split_params(s: str) -> List[str]:
        params: List[str] = []
        cur = []
        depth_angle = depth_paren = depth_brack = 0
        for ch in s:
            if ch == "<":
                depth_angle += 1
            elif ch == ">" and depth_angle > 0:
                depth_angle -= 1
            elif ch == "(":
                depth_paren += 1
            elif ch == ")" and depth_paren > 0:
                depth_paren -= 1
            elif ch == "[":
                depth_brack += 1
            elif ch == "]" and depth_brack > 0:
                depth_brack -= 1
            if ch == "," and depth_angle == 0 and depth_paren == 0 and depth_brack == 0:
                p = "".join(cur).strip()
                if p:
                    params.append(p)
                cur = []
                continue
            cur.append(ch)
        tail = "".join(cur).strip()
        if tail:
            params.append(tail)
        return params

    def _parse_signature(code: str, target: str) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        m = re.search(
            rf"(?s)^\s*([A-Za-z_][\w\s:<>\*&]+?)\b{re.escape(target)}\s*\((.*?)\)\s*\{{",
            code,
            flags=re.MULTILINE,
        )
        if not m:
            return None
        ret = m.group(1).strip()
        params_str = m.group(2).strip()
        if not params_str or params_str == "void":
            return ret, []
        params = []
        for idx, raw in enumerate(_split_params(params_str)):
            p = raw.split("=", 1)[0].strip()
            if not p:
                continue
            name_m = re.search(r"([A-Za-z_]\w*)\s*$", p)
            name = name_m.group(1) if name_m else f"arg{idx}"
            type_part = p[: name_m.start()].strip() if name_m else p
            is_const = "const" in type_part.split()
            is_ptr = "*" in type_part
            is_ref = "&" in type_part
            base = type_part.replace("const", "").replace("*", "").replace("&", "").strip()
            params.append(
                {
                    "name": name,
                    "type": type_part.strip(),
                    "base": base,
                    "is_const": is_const,
                    "is_ptr": is_ptr,
                    "is_ref": is_ref,
                }
            )
        return ret, params

    serial_code = _rename_entry(serial_ref_completion or "", func_name, f"{func_name}_serial")
    simd_code = _rename_entry(simd_completion or "", func_name, f"{func_name}_simd")

    sig = _parse_signature(serial_code, f"{func_name}_serial")
    if sig is None:
        return None
    ret_type, params = sig

    # Basic type checks
    def _is_numeric(t: str) -> bool:
        t = t.replace("unsigned", "uint").replace("signed", "int")
        return any(x in t for x in ["int", "uint", "size_t", "long", "short", "float", "double", "bool"])

    for p in params:
        if not p["is_ptr"] and not _is_numeric(p["base"]):
            return None
        if p["is_ptr"] and not _is_numeric(p["base"]):
            return None

    # Build size hints
    size_vals: Dict[str, str] = {}
    for p in params:
        n = p["name"].lower()
        if not p["is_ptr"] and _is_numeric(p["base"]):
            if any(k in n for k in ["rows", "cols"]):
                size_vals[p["name"]] = "4"
            elif any(k in n for k in ["n", "len", "size", "count"]):
                size_vals[p["name"]] = "8"
            else:
                size_vals[p["name"]] = "7"

    def _guess_ptr_len(pname: str) -> str:
        n = pname.lower()
        # Common SimdBench pattern: dim1/dim2/dim3 describe a flattened tensor length.
        # Use a product of the constant size hints to keep the harness memory-safe even when
        # pointer params appear before the dim params in the signature.
        dim_keys = [k for k in size_vals.keys() if re.match(r"^dim\d+$", k.lower())]
        if len(dim_keys) >= 2:
            def _dim_num(x: str) -> int:
                m = re.search(r"\d+", x)
                return int(m.group(0)) if m else 0
            dim_keys_sorted = sorted(dim_keys, key=_dim_num)
            return "(" + "*".join(size_vals[k] for k in dim_keys_sorted) + ")"
        if "matrix" in n and "rows" in size_vals and "cols" in size_vals:
            return f"({size_vals['rows']}*{size_vals['cols']})"
        if ("vector" in n or "vec" in n) and "cols" in size_vals:
            return size_vals["cols"]
        for key in ["n", "len", "size", "count"]:
            for k, v in size_vals.items():
                if key == k.lower() or key in k.lower():
                    return v
        return "8"

    decls: List[str] = []
    fill: List[str] = []
    args_serial: List[str] = []
    args_simd: List[str] = []
    output_ptrs: List[Tuple[str, str]] = []

    def _printable_expr(var: str, base: str) -> str:
        """Ensure small integer types print as numbers (not chars) in JSON."""
        b = (base or "").replace("unsigned", "uint").replace("signed", "int").strip()
        if b in {"uint8_t", "int8_t", "char", "unsigned char", "signed char", "bool"}:
            return f"static_cast<int>({var})"
        return var

    for p in params:
        name = p["name"]
        base = p["base"] or "int"
        if p["is_ptr"]:
            n_expr = _guess_ptr_len(name)
            decls.append(f"size_t {name}_n = {n_expr};")
            if p["is_const"]:
                decls.append(f"std::vector<{base}> {name}_buf({name}_n);")
                fill.append(f"for (size_t i=0;i<{name}_n;i++) {name}_buf[i] = static_cast<{base}>(i+1);")
                arg = f"{name}_buf.data()"
                args_serial.append(arg)
                args_simd.append(arg)
            else:
                decls.append(f"std::vector<{base}> {name}_buf_serial({name}_n);")
                decls.append(f"std::vector<{base}> {name}_buf_simd({name}_n);")
                fill.append(f"for (size_t i=0;i<{name}_n;i++) {name}_buf_serial[i] = static_cast<{base}>(i+1);")
                fill.append(f"for (size_t i=0;i<{name}_n;i++) {name}_buf_simd[i] = static_cast<{base}>(i+1);")
                args_serial.append(f"{name}_buf_serial.data()")
                args_simd.append(f"{name}_buf_simd.data()")
                output_ptrs.append((name, base))
        else:
            val = size_vals.get(name, "7")
            decls.append(f"{base} {name} = static_cast<{base}>({val});")
            args_serial.append(name)
            args_simd.append(name)

    # Assemble harness
    lines: List[str] = []
    lines.append("#include <iostream>")
    lines.append("#include <sstream>")
    lines.append("#include <string>")
    lines.append("#include <vector>")
    lines.append("#include <cstdint>")
    lines.append("#include <cstddef>")
    lines.append("#include <cmath>")
    lines.append("#include <cfloat>")
    lines.append("#include <float.h>")
    lines.append("#include <climits>")
    lines.append("#include <limits>")
    lines.append("#include <algorithm>")
    lines.append("")
    lines.append(serial_code.strip())
    lines.append("")
    lines.append(simd_code.strip())
    lines.append("")
    lines.append("int main(){")
    for d in decls:
        lines.append(f"  {d}")
    for f in fill:
        lines.append(f"  {f}")
    lines.append("  auto to_json_value = [](const auto& v) -> std::string {")
    lines.append("    std::ostringstream oss;")
    lines.append("    oss << v;")
    lines.append("    return oss.str();")
    lines.append("  };")
    lines.append("  bool mismatch = false;")
    lines.append("  long long mismatch_index = -1;")
    lines.append("  std::string mismatch_expected = \"0\";")
    lines.append("  std::string mismatch_got = \"0\";")
    lines.append("  std::string mismatch_note = \"ok\";")
    lines.append("  std::vector<std::string> mismatch_examples;")
    lines.append("  auto record_mismatch = [&](long long idx, const auto& exp_v, const auto& got_v, const char* note) {")
    lines.append("    if (!mismatch) {")
    lines.append("      mismatch = true;")
    lines.append("      mismatch_index = idx;")
    lines.append("      mismatch_expected = to_json_value(exp_v);")
    lines.append("      mismatch_got = to_json_value(got_v);")
    lines.append("      mismatch_note = note;")
    lines.append("    }")
    lines.append("    if (mismatch_examples.size() < 16) {")
    lines.append("      mismatch_examples.push_back(")
    lines.append("          std::string(\"{\\\"index\\\":\") + std::to_string(idx) +")
    lines.append("          \",\\\"expected\\\":\" + to_json_value(exp_v) +")
    lines.append("          \",\\\"got\\\":\" + to_json_value(got_v) +")
    lines.append("          \",\\\"note\\\":\\\"\" + note + \"\\\"}\");")
    lines.append("    }")
    lines.append("  };")
    args_s = ", ".join(args_serial)
    args_v = ", ".join(args_simd)
    if "void" not in ret_type.replace(" ", ""):
        lines.append(f"  auto exp = {func_name}_serial({args_s});")
        lines.append(f"  auto got = {func_name}_simd({args_v});")
        lines.append("  if (exp != got) {")
        exp_p = _printable_expr("exp", ret_type)
        got_p = _printable_expr("got", ret_type)
        lines.append(f"    record_mismatch(-1, {exp_p}, {got_p}, \"ret\");")
        lines.append("  }")
    else:
        lines.append(f"  {func_name}_serial({args_s});")
        lines.append(f"  {func_name}_simd({args_v});")

    for name, base in output_ptrs:
        lines.append(f"  for (size_t i=0;i<{name}_n;i++) {{")
        if "float" in base or "double" in base:
            lines.append(f"    if (std::fabs({name}_buf_serial[i] - {name}_buf_simd[i]) > 1e-6) {{")
        else:
            lines.append(f"    if ({name}_buf_serial[i] != {name}_buf_simd[i]) {{")
        exp_p = _printable_expr(f"{name}_buf_serial[i]", base)
        got_p = _printable_expr(f"{name}_buf_simd[i]", base)
        lines.append(f"      record_mismatch(static_cast<long long>(i), {exp_p}, {got_p}, \"out\");")
        lines.append("    }")
        lines.append("  }")

    lines.append("  std::cout << \"{\\\"mismatch\\\":\" << (mismatch ? \"true\" : \"false\")")
    lines.append("            << \",\\\"index\\\":\" << mismatch_index")
    lines.append("            << \",\\\"expected\\\":\" << mismatch_expected")
    lines.append("            << \",\\\"got\\\":\" << mismatch_got")
    lines.append("            << \",\\\"note\\\":\\\"\" << mismatch_note << \"\\\"\"")
    lines.append("            << \",\\\"mismatch_examples\\\":[\";")
    lines.append("  for (size_t i = 0; i < mismatch_examples.size(); ++i) {")
    lines.append("    if (i) std::cout << \",\";")
    lines.append("    std::cout << mismatch_examples[i];")
    lines.append("  }")
    lines.append("  std::cout << \"]}\" << std::endl;")
    lines.append("  return 0;")
    lines.append("}")
    return "\n".join(lines)

def make_unified_diff(
    a: str,
    b: str,
    *,
    fromfile: str = "serial",
    tofile: str = "sve",
    context_lines: int = 3,
    max_chars: int = 4000,
) -> str:
    """Return a unified diff between two code strings, optionally middle-truncated."""
    if not a or not b:
        return ""
    try:
        diff_lines = list(
            difflib.unified_diff(
                str(a).splitlines(),
                str(b).splitlines(),
                fromfile=fromfile,
                tofile=tofile,
                n=max(0, int(context_lines or 0)),
                lineterm="",
            )
        )
        diff = "\n".join(diff_lines).strip()
    except Exception:
        return ""

    if max_chars and max_chars > 0 and len(diff) > max_chars:
        diff = _truncate_middle(diff, int(max_chars))
    return diff


_CONTROLLER_REPAIR_TEMPLATE_SPECS: Dict[str, Dict[str, List[str] | str]] = {
    "compile_bitcount_api_fix": {
        "title": "Bitcount And Counting API Fix",
        "objective": "Repair misuse of counting/bitcount-related intrinsics without rewriting unrelated loop logic.",
        "must": [
            "Distinguish vector-length counting intrinsics from per-element bitcount/popcount operations.",
            "Keep loop order, indexing, and output shape unchanged while replacing the wrong counting API family.",
            "Use a real per-element bitcount path or a conservative scalar fallback if no valid SVE path is clear.",
        ],
        "avoid": [
            "Do not use svcnt* length-count intrinsics as if they computed per-element population count.",
            "Do not widen this into a full algorithm rewrite.",
        ],
        "checklist": [
            "bitcount/popcount semantics are implemented with the correct API family",
            "vector-length counting intrinsics are only used for lane-count/step logic",
            "store width matches the actual result element width",
        ],
    },
    "compile_acle_api_fix": {
        "title": "ACLE Call Shape Fix",
        "objective": "Repair invalid Arm SVE intrinsic names and call signatures before touching broader logic.",
        "must": [
            "Change only invalid sv* names, overloads, argument counts, argument ordering, or predicate/value operand placement.",
            "Keep loop structure, indexing, and arithmetic intact unless a local API fix forces a matching width change.",
            "Use only real <arm_sve.h> APIs; do not invent helper intrinsics, wrappers, or replacement sv* names.",
        ],
        "avoid": [
            "Do not rewrite the algorithm just to silence one bad intrinsic call.",
            "Do not change the function signature or output semantics.",
        ],
        "checklist": [
            "every sv* symbol exists in <arm_sve.h>",
            "argument count and order match the chosen intrinsic",
            "predicate operands are svbool_t and value operands are vector/scalar values",
            "load/store width matches the vector element width",
        ],
    },
    "compile_symbol_closure_fix": {
        "title": "Standalone Symbol Closure Fix",
        "objective": "Repair missing in-file helper definitions, missing index helpers, and nearby syntax closure damage without widening the rewrite.",
        "must": [
            "Every non-stdlib, non-<arm_sve.h> helper you call must be defined in this same file before first use, or be inlined away.",
            "If a helper-like symbol is missing, either emit its full definition in-file or replace the call with a local inline expression.",
            "Preserve existing helper definitions unless you also remove every call site that depends on them.",
            "Keep the edit scope local to missing helpers, invalid builtin names, and syntax closure problems.",
        ],
        "avoid": [
            "Do not invent new *_local helper names unless you also emit their full definition in-file.",
            "Do not widen the repair into an unrelated algorithm rewrite.",
            "Do not delete a helper block and leave the original call sites behind.",
        ],
        "checklist": [
            "every helper-like symbol referenced by the final file has an in-file definition or has been inlined away",
            "missing index helpers are resolved in-file rather than left as undeclared names",
            "unsupported builtin/intrinsic spellings are replaced with valid alternatives or removed",
            "parentheses, braces, and obvious statement terminators are syntactically closed",
        ],
    },
    "compile_cpp17_syntax_fix": {
        "title": "Local C++17 Syntax Closure Fix",
        "objective": "Repair the exact local clang++ syntax-closure failures before sending the candidate to remote eval.",
        "must": [
            "Prioritize the exact local clang++ syntax errors first: missing semicolons, mismatched delimiters, broken declarations, and nested function/body structure damage.",
            "Keep the repair local and minimal; restore a coherent standalone C++17 file before touching broader logic.",
            "Preserve the exact target function signature from the prompt prefix, the algorithm intent, and existing helper blocks unless the syntax error is inside them.",
        ],
        "avoid": [
            "Do not widen a local syntax fix into an unrelated algorithm rewrite.",
            "Do not invent new helpers or rename symbols unless the syntax error directly forces it.",
        ],
        "checklist": [
            "the file parses as a closed C++17 translation unit",
            "statements and declarations are terminated correctly",
            "parentheses, braces, and brackets are balanced",
        ],
    },
    "compile_signature_shape_fix": {
        "title": "Target Signature Closure Fix",
        "objective": "Restore the exact target function signature from the prompt prefix before fixing any downstream compile noise.",
        "must": [
            "Treat the prompt-prefix target signature as the only source of truth, even if the current completion already changed it.",
            "Repair the function header first so return type, parameter count, parameter types, parameter order, and array rank match the expected target signature exactly.",
            "Keep the function body and loop logic as intact as possible unless a tiny local alias is needed after restoring the signature.",
        ],
        "avoid": [
            "Do not preserve the current broken signature just because it already appears in the completion.",
            "Do not widen a signature repair into a broader algorithm rewrite.",
        ],
        "checklist": [
            "the final target function header exactly matches the prompt-prefix signature",
            "parameter types, order, and array rank are restored before any broader repair",
            "the body still implements the same intended algorithm after the header repair",
        ],
    },
    "compile_predicate_api_fix": {
        "title": "Predicate API Fix",
        "objective": "Repair predicate creation and predicate/data separation without rewriting unrelated arithmetic.",
        "must": [
            "Fix bool or predicate loads, predicate construction, and predicate consumers first.",
            "Keep which values are computed the same; only repair how active lanes are represented and consumed.",
            "Use predicate-producing intrinsics for predicates and data intrinsics for data vectors.",
        ],
        "avoid": [
            "Do not reinterpret predicates as data vectors or data vectors as predicates.",
            "Do not widen the repair into unrelated math or reduction rewrites.",
        ],
        "checklist": [
            "bool/mask inputs are converted into valid svbool_t predicates",
            "svptest*/svcntp* operate on predicates, not data vectors",
            "predicate width matches the element width of the guarded operation",
        ],
    },
    "compile_loop_shape_fix": {
        "title": "Loop Shape And Lane Width Fix",
        "objective": "Make the loop step, predicate width, and load/store width agree on the same element type.",
        "must": [
            "Align svwhilelt_b*, vector type width, load/store intrinsic width, and svcnt* step with the same element size.",
            "Change only the local loop-shape pieces needed to remove the mismatch.",
            "Preserve the original iteration order and tail behavior.",
        ],
        "avoid": [
            "Do not change arithmetic or indexing semantics unless they are directly tied to the width mismatch.",
            "Do not mix b8/b16/b32/b64 predicates with a different element-width loop step.",
        ],
        "checklist": [
            "svwhilelt_b* width matches the vector element type",
            "loop increment uses the matching svcntb/h/w/d",
            "tail predicate and store predicate use the same logical lane width",
        ],
    },
    "compile_width_store_fix": {
        "title": "Reinterpret Width Store Fix",
        "objective": "Repair width mismatches introduced by reinterpret/cast/store combinations.",
        "must": [
            "Keep the data flow but make reinterpret, load/store, and destination element width consistent.",
            "If narrowing is required, use a valid narrowing or packing path instead of raw reinterpret.",
            "Store through the intrinsic whose element width matches the destination buffer type.",
        ],
        "avoid": [
            "Do not dump a reinterpreted wider vector directly into a narrower buffer.",
            "Do not change the meaning of the values while fixing width mismatches.",
        ],
        "checklist": [
            "destination pointer type matches the final store intrinsic",
            "reinterpret does not silently change the lane count expected by the store",
            "any narrowing/widening path is explicit and valid",
        ],
    },
    "compile_addressing_fix": {
        "title": "Addressing And Gather/Scatter Fix",
        "objective": "Repair pointer arithmetic and gather/scatter addressing without rewriting the whole algorithm.",
        "must": [
            "Fix base pointer, index expression, stride usage, and gather/scatter addressing only.",
            "Use element indexing on typed pointers; never multiply typed-pointer offsets by sizeof(T).",
            "If true gather/scatter is unclear, keep the repair anchored on supported u32-index ACLE forms instead of inventing addresses or switching to scalar fallback.",
        ],
        "avoid": [
            "Do not cast integers into pointers or dereference integer addresses.",
            "Do not access vector lanes through non-portable lane members.",
        ],
        "checklist": [
            "gather/scatter indices are element indices, not raw byte addresses",
            "base pointer and stride correspond to the target buffer layout",
            "typed pointer arithmetic does not apply sizeof(T) manually",
        ],
    },
    "local_compile_fix": {
        "title": "Local Compile Hazard Fix",
        "objective": "Clear the local static verifier hazards with the smallest possible compile-oriented edits.",
        "must": [
            "Prioritize the tagged compile hazards before remote-only symptoms.",
            "Keep semantics stable while making the code compile into a coherent SVE implementation.",
        ],
        "avoid": [
            "Do not widen the repair into an unnecessary algorithm rewrite.",
        ],
        "checklist": [
            "flagged local compile hazards are resolved first",
            "function signature and high-level algorithm remain unchanged",
        ],
    },
    "access_semantic_fix": {
        "title": "Access Map Semantic Fix",
        "objective": "Make code-side indexing, stride usage, and gather/scatter behavior match semantic_plan.access_map exactly.",
        "must": [
            "Repair only index expressions, buffer selection, stride usage, and gather/scatter direction.",
            "Follow the plan's access_map and uncertain_fields; if the plan is conservative, keep the code conservative too.",
            "Preserve reduction logic and arithmetic unless they directly depend on a wrong access pattern.",
        ],
        "avoid": [
            "Do not rewrite predicates or reduction structure unless access semantics force it.",
            "Do not invent a more complex memory layout than the plan states.",
        ],
        "checklist": [
            "every source/destination buffer matches the plan",
            "indices, strides, and flattening rules match semantic_plan.access_map",
            "gather/scatter direction is correct for the target output",
        ],
    },
    "predicate_value_parity_fix": {
        "title": "Predicate Value Parity Fix",
        "objective": "Repair value-parity predicates so odd/even VALUE checks stay exact instead of collapsing into weaker nonzero tests.",
        "must": [
            "Implement true odd/even VALUE checks with parity logic such as bit-0 or modulo, not value != 0.",
            "Keep index predicates separate from value predicates.",
            "Preserve arithmetic and reduction unless the wrong value-parity predicate is what breaks semantics.",
        ],
        "avoid": [
            "Do not replace odd/even value logic with nonzero checks.",
            "Do not rewrite unrelated access or reduction structure.",
        ],
        "checklist": [
            "odd/even value checks use true parity logic",
            "index parity and value parity remain separate",
            "the repaired predicate still matches the task's value condition",
        ],
    },
    "predicate_index_parity_fix": {
        "title": "Predicate Index Parity Fix",
        "objective": "Repair index/lane parity predicates so code selects lanes by index position, not by value approximations.",
        "must": [
            "Implement index-driven parity using lane/global index expressions.",
            "Keep value comparisons separate from index comparisons.",
            "Preserve the original arithmetic and reduction unless the wrong index predicate directly corrupts them.",
        ],
        "avoid": [
            "Do not replace index parity with value != 0 or value parity checks.",
            "Do not widen this into a full algorithm rewrite.",
        ],
        "checklist": [
            "index parity is derived from lane/global indices",
            "value predicates are not reused as index predicates",
            "tail predicate and index predicate remain distinct when both are needed",
        ],
    },
    "predicate_filter_condition_fix": {
        "title": "Predicate Filter Condition Fix",
        "objective": "Repair data-dependent predicate logic while preserving the task's original value condition exactly.",
        "must": [
            "Use the real data condition from the task or semantic_plan.",
            "Preserve odd/even value checks, threshold checks, and comparison direction exactly.",
            "Keep index predicates separate when the task also has positional conditions.",
        ],
        "avoid": [
            "Do not replace odd/even value logic with value != 0.",
            "Do not collapse a threshold or equality check into a weaker nonzero predicate.",
        ],
        "checklist": [
            "data predicate matches the original value condition",
            "odd/even value checks use parity logic rather than nonzero checks",
            "data predicate is not confused with tail-only or index-only predicates",
        ],
    },
    "predicate_index_vs_data_fix": {
        "title": "Predicate Index Vs Data Fix",
        "objective": "Repair which predicate dimensions come from index position versus data values so semantic_plan.predicate_rule is respected.",
        "must": [
            "Align predicate sources with semantic_plan: index-driven rules use indices, data-driven rules use loaded values.",
            "Keep tail handling, index filtering, and data filtering as distinct concepts unless the plan explicitly combines them.",
            "Preserve arithmetic and access patterns unless the wrong predicate source directly corrupts them.",
        ],
        "avoid": [
            "Do not reuse a value predicate as an index predicate, or vice versa.",
            "Do not collapse mixed predicate logic into a tail-only mask.",
        ],
        "checklist": [
            "predicate source matches index/data dependency in the plan",
            "tail, index, and data predicates are composed intentionally",
            "lane selection for load/store/reduction uses the intended predicate",
        ],
    },
    "reduction_guard_fix": {
        "title": "Reduction Guard Fix",
        "objective": "Remove accidental reduction-shaped code from tasks that should not use horizontal reduction intrinsics.",
        "must": [
            "If the task is not a true reduction, remove svaddv/svminv/svmaxv/svcntp style horizontal finalize from the main result path.",
            "Preserve per-lane arithmetic and output shape semantics.",
            "Use scalar fallback for the irregular sub-step if a safe vectorized non-reduction form is unclear.",
        ],
        "avoid": [
            "Do not keep a reduction-shaped output in a non-reduction task.",
            "Do not force a vector reduction only because an accumulator already exists.",
        ],
        "checklist": [
            "non-reduction tasks do not end in horizontal reduction intrinsics",
            "output shape matches the task rather than a scalar reduction shape",
            "per-lane computation remains intact",
        ],
    },
    "reduction_finalize_fix": {
        "title": "Reduction Finalize Fix",
        "objective": "Repair the finalize stage of a real reduction so per-lane accumulation, horizontal finalize, and scalar writeback are explicit and correct.",
        "must": [
            "Separate per-lane accumulation from horizontal finalize.",
            "Write back a scalar result only after an explicit finalize step.",
            "For mean-style tasks, keep the final divide or normalization step explicit.",
        ],
        "avoid": [
            "Do not store a vector accumulator directly as the final scalar-semantic result.",
            "Do not change reduction kind while repairing finalize.",
        ],
        "checklist": [
            "per-lane accumulation is explicit",
            "horizontal finalize is explicit",
            "scalar/global writeback happens only after finalize",
        ],
    },
    "conservative_scalar_fallback_fix": {
        "title": "Conservative Scalar Fallback Fix",
        "objective": "For irregular or order-sensitive tasks, keep the repair conservative and use scalar fallback for the unsafe sub-step instead of forcing a brittle vector rewrite.",
        "must": [
            "Preserve the exact function interface and task semantics.",
            "Limit vectorization to the obviously safe sub-steps.",
            "Use scalar fallback for the irregular, order-sensitive, or control-heavy part.",
        ],
        "avoid": [
            "Do not force a full SVE rewrite of an irregular algorithm.",
            "Do not invent complex gather/scatter or reduction patterns without clear semantic support.",
        ],
        "checklist": [
            "unsafe irregular sub-step is kept scalar",
            "safe vectorized sub-steps remain simple and local",
            "the repair is conservative rather than expansive",
        ],
    },
    "semantic_fix": {
        "title": "Semantic Plan Consistency Fix",
        "objective": "Make the generated code conform to the semantic_plan with minimal edits.",
        "must": [
            "Use semantic_plan as the primary structure-of-truth for access, predicate, and reduction behavior.",
            "Prioritize must_fix_tags before remote-only symptoms.",
            "Keep the implementation minimal and conservative.",
        ],
        "avoid": [
            "Do not enlarge the rewrite beyond what is needed to restore plan/code consistency.",
        ],
        "checklist": [
            "code structure matches semantic_plan hotspots first",
            "function signature and external behavior stay unchanged",
        ],
    },
    "remote_escalation": {
        "title": "Remote Escalation",
        "objective": "Local verifier is inconclusive; rely on remote feedback while keeping edits minimal and well-scoped.",
        "must": [
            "Use remote failure evidence as the primary repair signal.",
            "Keep the code close to the previous completion unless the remote evidence clearly demands a local change.",
        ],
        "avoid": [
            "Do not invent extra local fixes unsupported by the current evidence.",
        ],
        "checklist": [
            "only observed remote failure modes are addressed",
            "repair remains minimal and interface-preserving",
        ],
    },
    "no_local_repair_needed": {
        "title": "No Local Repair Needed",
        "objective": "No local verifier hazards were found; only address the remote failure if it is explicitly reported.",
        "must": [
            "Prefer no-op over speculative rewrites.",
        ],
        "avoid": [
            "Do not change code without concrete evidence.",
        ],
        "checklist": [
            "no speculative local rewrite was introduced",
        ],
    },
}

PRE_REMOTE_LOCAL_REPAIR_ACTIONS = set()

PASS1_V22_SEMANTIC_ACTIONS = {
    "predicate_index_parity_fix",
    "predicate_value_parity_fix",
    "predicate_filter_condition_fix",
    "predicate_index_vs_data_fix",
    "reduction_guard_fix",
    "reduction_finalize_fix",
    "conservative_scalar_fallback_fix",
}


def _clean_str_list(value: Any, *, limit: int = 8) -> List[str]:
    out: List[str] = []
    if not isinstance(value, list):
        return out
    for item in value[: max(0, int(limit))]:
        txt = str(item).strip()
        if txt:
            out.append(txt)
    return out


def _clean_tag_name_list(value: Any, *, limit: int = 8) -> List[str]:
    out: List[str] = []
    if not isinstance(value, list):
        return out
    for item in value[: max(0, int(limit))]:
        if isinstance(item, dict):
            txt = str(item.get("tag") or "").strip()
        else:
            txt = str(item).strip()
        if txt:
            out.append(txt)
    return out


def _clean_named_diag_map(value: Any, *, per_key_limit: int = 12) -> Dict[str, List[str]]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for key, raw in value.items():
        items = _clean_str_list(raw, limit=per_key_limit)
        if items:
            out[str(key)] = items
    return out


def _strip_compile_error_prefix(text: str) -> str:
    """Remove volatile compile-mode/path/line prefixes from a diagnostic line."""
    msg = str(text or "").strip()
    if not msg:
        return ""
    msg = re.sub(r"\[compile_mode=[^\]]+\]\s*", "", msg)
    msg = re.sub(r"^compilation failed:\s*", "", msg, flags=re.IGNORECASE)
    # Keep the semantic error payload after the last clang-style "error:" prefix.
    m = re.search(r"(?:fatal\s+)?error:\s*(.+)$", msg, flags=re.IGNORECASE)
    if m:
        msg = m.group(1).strip()
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg


def _compact_cpp_type_name(text: str) -> str:
    t = str(text or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("(aka ", "aka ")
    return t


def _sanitize_syntax_signal(line: str) -> str:
    pre_sanitized = str(line or "").strip()
    if pre_sanitized.startswith("expected_token:"):
        return pre_sanitized
    if pre_sanitized in {
        "nested_function_definition",
        "expected_expression",
        "extraneous_closing_brace",
        "unexpected_end_of_input",
        "syntax_error",
    }:
        return pre_sanitized
    msg = _strip_compile_error_prefix(line).lower()
    if not msg:
        return ""
    if "function definition is not allowed here" in msg:
        return "nested_function_definition"
    if "expected expression" in msg:
        return "expected_expression"
    if "extraneous closing brace" in msg:
        return "extraneous_closing_brace"
    if "at end of input" in msg:
        return "unexpected_end_of_input"
    m = re.search(r"expected ['`]?([^'` ]+)['`]?", msg)
    if m:
        return f"expected_token:{m.group(1)}"
    return "syntax_error"


def _sanitize_invalid_cast_message(line: str) -> str:
    msg = _strip_compile_error_prefix(line)
    low = msg.lower()
    m = re.search(
        r"c-style cast from ['‘]([^'’]+)['’](?:\s*\([^)]*\))?\s+to ['‘]([^'’]+)['’](?:\s*\([^)]*\))?\s+is not allowed",
        msg,
        flags=re.IGNORECASE,
    )
    if m:
        src = _compact_cpp_type_name(m.group(1))
        dst = _compact_cpp_type_name(m.group(2))
        if "sv" in src.lower() or "__sv" in src.lower():
            return f"vector_to_scalar_cast:{src}->{dst}"
        return f"invalid_cast:{src}->{dst}"
    m = re.search(
        r"cannot cast from type ['‘]([^'’]+)['’].*?to (?:pointer )?type ['‘]([^'’]+)['’]",
        msg,
    )
    if m:
        src = _compact_cpp_type_name(m.group(1))
        dst = _compact_cpp_type_name(m.group(2))
        if ("sv" in src.lower() or "__sv" in src.lower()) and "*" in dst:
            return f"vector_to_pointer_cast:{src}->{dst}"
        return f"invalid_cast:{src}->{dst}"
    if "reinterpret_cast" in low:
        return "invalid_reinterpret_cast"
    if "static_cast from" in low:
        return "invalid_static_cast"
    if "casts away qualifiers" in low:
        return "casts_away_qualifiers"
    if "assignment to cast is illegal" in low:
        return "assignment_to_cast"
    if "cannot cast" in low:
        return "invalid_cast"
    return ""


def _sanitize_type_mismatch_message(line: str) -> str:
    msg = _strip_compile_error_prefix(line)
    low = msg.lower()
    m = re.search(
        r"cannot be narrowed from type ['‘]([^'’]+)['’].*?to ['‘]([^'’]+)['’]",
        msg,
    )
    if m:
        return f"narrowing_conversion:{_compact_cpp_type_name(m.group(1))}->{_compact_cpp_type_name(m.group(2))}"
    m = re.search(
        r"cannot be narrowed to type ['‘]([^'’]+)['’]",
        msg,
        flags=re.IGNORECASE,
    )
    if m:
        return f"narrowing_conversion:constant->{_compact_cpp_type_name(m.group(1))}"
    m = re.search(
        r"cannot initialize .*?type ['‘]([^'’]+)['’].*?type ['‘]([^'’]+)['’]",
        msg,
    )
    if m:
        return f"cannot_initialize:{_compact_cpp_type_name(m.group(1))}<-{_compact_cpp_type_name(m.group(2))}"
    if "subscripted value is not an array" in low:
        return "non_array_subscript"
    if "comparison between pointer and integer" in low:
        return "pointer_integer_comparison"
    if "vector condition type" in low:
        return "vector_used_as_scalar_condition"
    if "cannot convert between vector and non-scalar values" in low:
        return "vector_scalar_conversion"
    if "cannot convert between scalar type" in low and "vector type" in low:
        return "vector_scalar_conversion"
    if "invalid operands to binary expression" in low:
        return "invalid_binary_operands"
    if "invalid argument type" in low:
        return "invalid_argument_type"
    return "expression_type_mismatch" if msg else ""


def _sanitize_invalid_cpp_construct_message(line: str) -> str:
    msg = _strip_compile_error_prefix(line).lower()
    if not msg:
        return ""
    if "called object type" in msg and "is not a function" in msg:
        return "called_non_function_object"
    if "variable-sized object may not be initialized" in msg:
        return "vla_initialization"
    if "expression is not assignable" in msg:
        return "non_assignable_expression"
    if "does not allow incrementing expression of type bool" in msg:
        return "bool_increment"
    if "integer literal is too large" in msg:
        return "integer_literal_too_large"
    if "no matching literal operator" in msg:
        return "invalid_literal_suffix"
    if "invalid digit" in msg:
        return "invalid_numeric_literal"
    if "cannot jump from this goto statement" in msg:
        return "invalid_goto_scope"
    return "invalid_cpp_construct"


def _sanitize_sve_tuple_access_message(line: str) -> str:
    msg = _strip_compile_error_prefix(line).lower()
    if "member reference base type" in msg and (
        "__clang_sv" in msg or "__sv" in msg or re.search(r"\bsv\w*x\d+_t\b", msg)
    ):
        return "sve_tuple_access_misuse"
    return "sve_tuple_access_misuse" if msg else ""


def _sanitize_diagnostic_location(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if " | source: " not in text:
        return text
    prefix, source = text.split(" | source: ", 1)
    m = re.search(r"\bcolumn\s+(\d+)\b", prefix)
    if not m:
        return text
    source_detail = _source_focus_for_column(source, m.group(1))
    if not source_detail:
        return prefix
    return f"{prefix} | {source_detail}"


def _split_diagnostic_source_context(value: str) -> Tuple[str, str]:
    text = str(value or "").strip()
    marker = " | source_focus: "
    if marker not in text:
        return text, ""
    prefix, focus = text.split(marker, 1)
    m = re.match(r"((?:error|note|warning):(?: origin=\w+)? line \d+, column \d+)", prefix)
    loc = m.group(1) if m else "source_context"
    return prefix.strip(), f"{loc} | source_focus: {focus.strip()}"


def _diagnostic_location_is_related_note(value: str) -> bool:
    text = str(value or "").strip()
    low = text.lower()
    if not text:
        return False
    if low.startswith("note:"):
        return True
    if "origin=stl" in low or "origin=system" in low or "origin=acle" in low:
        return True
    system_source_markers = [
        "source: swap(pair<",
        "source: distance(_inputiterator",
        "source: operator()(_iterator",
        "source: if (__comp(",
        "source: *__last = _glibcxx_move",
        "source: { return bool(_m_comp",
        "__gnu_cxx::__ops::",
        "_glibcxx_move",
    ]
    return any(marker in low for marker in system_source_markers)


def _compact_related_note(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return _compact_source_context(text, limit=220)


def _sanitize_compile_diag_value(key: str, value: str) -> str:
    if key == "syntax_signals":
        return _sanitize_syntax_signal(value)
    if key == "diagnostic_locations":
        return _sanitize_diagnostic_location(value)
    if key == "invalid_cast_messages":
        return _sanitize_invalid_cast_message(value)
    if key == "type_mismatch_messages":
        return _sanitize_type_mismatch_message(value)
    if key == "sizeless_type_messages":
        return "sizeless_sve_type_misuse" if value else ""
    if key == "sve_tuple_access_messages":
        return _sanitize_sve_tuple_access_message(value)
    if key == "const_assignment_messages":
        return "const_assignment" if value else ""
    if key == "invalid_address_messages":
        return "invalid_address_or_rvalue_address" if value else ""
    if key == "invalid_cpp_construct_messages":
        return _sanitize_invalid_cpp_construct_message(value)
    if key == "error_messages":
        return ""
    if key == "error_locations":
        return ""
    return str(value or "").strip()


def _sanitize_compile_diagnostics_for_prompt(diagnostics: Dict[str, Any]) -> Dict[str, List[str]]:
    """Convert internal diagnostics to stable model-facing buckets with no raw clang paths."""
    diag = _clean_named_diag_map(diagnostics or {}, per_key_limit=12)
    if not diag:
        return {}

    suggestions: List[str] = []
    for key in ("compiler_suggestion", "compiler_suggestions", "did_you_mean_pairs"):
        suggestions.extend(diag.pop(key, []) or [])
    if suggestions:
        diag["compiler_suggestion"] = _dedup_text_items(suggestions, limit=12)

    sanitized: Dict[str, List[str]] = {}
    for key, values in diag.items():
        clean_values: List[str] = []
        for value in values or []:
            clean = _sanitize_compile_diag_value(key, value)
            if clean:
                if key == "diagnostic_locations" and _diagnostic_location_is_related_note(clean):
                    sanitized.setdefault("related_notes", []).append(_compact_related_note(clean))
                    continue
                if key == "diagnostic_locations":
                    clean, source_context = _split_diagnostic_source_context(clean)
                    if source_context:
                        sanitized.setdefault("source_contexts", []).append(source_context)
                clean_values.append(clean)
        if clean_values:
            sanitized[key] = _dedup_text_items(clean_values, limit=12)
    if sanitized.get("related_notes"):
        sanitized["related_notes"] = _dedup_text_items(sanitized["related_notes"], limit=6)
    if sanitized.get("source_contexts"):
        sanitized["source_contexts"] = _dedup_text_items(sanitized["source_contexts"], limit=6)
    return sanitized


def _call_shape_detail_label(name: str) -> str:
    n = str(name or "").strip()
    if n.startswith("svst"):
        return "store_pointer_or_value_shape"
    if n.startswith("svld"):
        return "load_pointer_or_index_shape"
    if n.startswith("svwhile"):
        return "predicate_loop_bound_shape"
    if n.startswith("svcntp"):
        return "predicate_count_shape"
    if n.startswith("svcmp") or n.startswith("svptest"):
        return "predicate_compare_shape"
    if n.startswith("sv"):
        return "intrinsic_arg_shape"
    return "call_arg_shape"


def _call_shape_family(name: str) -> str:
    n = str(name or "").strip()
    if n.startswith("sv"):
        return "acle"
    if n.startswith("__builtin_"):
        return "builtin"
    if n.startswith("std::") or n in {"distance", "reverse", "sort", "max", "min"}:
        return "std"
    return "cxx"


def _call_shape_reason_label(name: str, reasons: Sequence[str], diag: Dict[str, List[str]]) -> str:
    n = str(name or "").strip()
    family = _call_shape_family(n)
    reason_set = {str(r or "").strip() for r in reasons if str(r or "").strip()}
    location_blob = " ".join((diag.get("diagnostic_locations") or []) + (diag.get("related_notes") or [])).lower()
    if family == "std":
        if n.endswith("distance") and ("_inputiterator" in location_blob or "deduced conflicting types" in location_blob):
            return "iterator_type_conflict"
        if "no member named" in location_blob:
            return "member_or_api_shape"
        return "std_call_shape"
    if "no known conversion from 'bool' to 'svbool_t'" in location_blob:
        return "predicate_arg_shape"
    has_pointer_conversion = bool(re.search(r"no known conversion from '[^']*\*'[^,\n]* to '[^']*\*'", location_blob) or re.search(
        r"cannot initialize a parameter of type '[^']*\*'[^,\n]*with an .* type '[^']*\*'",
        location_blob,
    ))
    if has_pointer_conversion and n.startswith("svld"):
        return "load_pointer_element_type_shape"
    if has_pointer_conversion and n.startswith("svst"):
        return "store_pointer_element_type_shape"
    if has_pointer_conversion and family != "acle":
        return "pointer_element_type_shape"
    if re.search(r"requires\s+\d+\s+arguments?", location_blob) or "too few arguments" in location_blob or "too many arguments" in location_blob:
        return "arity_shape"
    if re.search(r"cannot initialize.*svu?int\d+_t.*svu?int\d+_t", location_blob):
        return "vector_signedness_or_width_shape"
    if "pointer_type" in reason_set:
        return _call_shape_detail_label(n)
    if "argument_type" in reason_set:
        return _call_shape_detail_label(n)
    if "no_matching_overload" in reason_set:
        return "overload_or_arity_shape" if family != "acle" else _call_shape_detail_label(n)
    return _call_shape_detail_label(n)


def _has_syntax_token_signal(diag: Dict[str, List[str]]) -> bool:
    return bool(diag.get("syntax_signals") or diag.get("expected_tokens"))


def _syntax_should_be_primary(diag: Dict[str, List[str]]) -> bool:
    if not _has_syntax_token_signal(diag):
        return False
    tokens = {str(tok or "").strip() for tok in (diag.get("expected_tokens") or [])}
    signals = {str(sig or "").strip() for sig in (diag.get("syntax_signals") or [])}
    if tokens.intersection({";", "{", "}", "expression"}):
        return True
    if signals.intersection({"expected_token:;", "expected_token:{", "expected_token:}", "expected_expression"}):
        return True
    if any(sig in {"expected_expression", "nested_function_definition", "extraneous_closing_brace", "unexpected_end_of_input"} for sig in signals):
        return True
    location_blob = " ".join(diag.get("diagnostic_locations") or [])
    if re.search(
        r"\b(?:auto|bool|char|float|double|size_t|u?int(?:8|16|32|64)_t|sv[a-z0-9_]*_t|std::[A-Za-z_][A-Za-z0-9_:<>]*)\s+[A-Za-z_][A-Za-z0-9_]*\s+[A-Za-z_][A-Za-z0-9_]*\b",
        location_blob,
    ):
        return True
    non_syntax_roots = [
        "unsupported_symbols",
        "missing_helper_symbols",
        "missing_index_symbols",
        "call_shape_mismatches",
        "ambiguous_calls",
        "compile_time_immediate_args",
        "missing_include_headers",
        "missing_members",
        "unknown_type_names",
        "redefined_symbols",
        "type_mismatch_messages",
        "sve_tuple_access_messages",
        "sizeless_type_messages",
        "const_assignment_messages",
        "invalid_cast_messages",
        "invalid_address_messages",
        "invalid_cpp_construct_messages",
    ]
    if not any(diag.get(key) for key in non_syntax_roots):
        return True
    return False


def _primary_compile_error_type(diag: Dict[str, List[str]]) -> str:
    if _syntax_should_be_primary(diag):
        return "syntax_token"
    if diag.get("sve_tuple_access_messages"):
        return "sve_tuple_access_misuse"
    if diag.get("sizeless_type_messages"):
        return "sve_sizeless_misuse"
    if (
        diag.get("call_shape_mismatches")
        or diag.get("ambiguous_calls")
        or diag.get("compile_time_immediate_args")
    ):
        return "call_shape_mismatch"
    if diag.get("unsupported_symbols"):
        return "unsupported_api"
    if (
        diag.get("missing_helper_symbols")
        or diag.get("missing_index_symbols")
        or diag.get("undeclared_identifiers")
        or diag.get("missing_include_headers")
        or diag.get("missing_members")
        or diag.get("unknown_type_names")
    ):
        return "symbol_resolution"
    if (
        diag.get("redefined_symbols")
        or diag.get("const_assignment_messages")
        or diag.get("invalid_cast_messages")
        or diag.get("invalid_address_messages")
        or diag.get("invalid_cpp_construct_messages")
    ):
        return "cxx_construct_invalid"
    if diag.get("type_mismatch_messages"):
        return "expression_type_mismatch"
    return "unparsed_compile_failure"


def _render_remote_evaluator_failure(
    parts: List[str],
    artifacts: Sequence[str],
    *,
    phase: str = "",
    section_name: str = "[REMOTE_EVALUATOR_FAILURE]",
) -> bool:
    clean = _dedup_text_items(artifacts or [], limit=12)
    if not clean:
        return False
    parts.append("\n" + section_name)
    parts.append("- error_type: remote_evaluator_failure")
    phase_clean = str(phase or "").strip()
    if phase_clean:
        parts.append(f"- phase: {phase_clean}")
    parts.append("- artifact: " + ", ".join(clean))
    parts.append("- action: rerun_or_ignore_for_code_repair")
    return True


def _remote_evaluator_artifacts_from_result(remote_result: Dict[str, Any], phase: str = "") -> List[str]:
    res = remote_result if isinstance(remote_result, dict) else {}
    reason = str(res.get("reason") or phase or "").strip().lower()
    blobs = "\n".join(
        str(res.get(key) or "")
        for key in [
            "ssh_blob",
            "compile_log_tail",
            "run_log_tail",
            "simdbench_raw_result",
            "sync_msg",
            "_sync_msg_tail",
        ]
    ).lower()
    artifacts: List[str] = []
    infra_reasons = {
        "scp_fail",
        "remote_cmd_fail",
        "bad_remote_json",
        "bad_remote_simdbench_eval",
        "remote_eval_json_parse_fail",
        "remote_transport_fail",
    }
    if reason in infra_reasons:
        artifacts.append(reason)
    if "tmp_src_missing" in blobs or "[tmp_src_missing]" in blobs:
        artifacts.append("tmp_src_missing")
    if "expecting value: line 1 column 1" in blobs or "json parse" in blobs and "empty" in blobs:
        artifacts.append("json_parse_empty_output")
    if "connection timed out" in blobs or "connection reset" in blobs or "ssh:" in blobs and "timed out" in blobs:
        artifacts.append("ssh_transport_failure")
    if "scp:" in blobs and ("no such file" in blobs or "failed" in blobs):
        artifacts.append("scp_failure")
    return _dedup_text_items(artifacts, limit=8)


def _collapse_call_shape_diagnostics(diag: Dict[str, List[str]]) -> None:
    """Merge overlapping call-shape buckets into one per-call summary.

    Clang often reports the same ACLE call as both "no matching function" and
    a pointer/argument mismatch.  Presenting those as separate top-level
    buckets makes the model chase multiple errors for one root cause.  Keep the
    per-call detail, but collapse it into a single model-facing field.
    """
    no_matching = list(diag.get("no_matching_calls") or [])
    ptr_mismatch = list(diag.get("ptr_type_mismatch_calls") or [])
    arg_mismatch = list(diag.get("arg_type_mismatch_calls") or [])
    if not (no_matching or ptr_mismatch or arg_mismatch):
        return

    specialized_symbols = set(diag.get("unsupported_symbols") or [])
    specialized_symbols.update(diag.get("missing_helper_symbols") or [])
    specialized_symbols.update(diag.get("missing_index_symbols") or [])
    names = [
        name for name in _dedup_text_items(no_matching + ptr_mismatch + arg_mismatch, limit=12)
        if name not in specialized_symbols
    ]
    if not names:
        diag.pop("no_matching_calls", None)
        diag.pop("ptr_type_mismatch_calls", None)
        diag.pop("arg_type_mismatch_calls", None)
        return
    ptr_set = set(ptr_mismatch)
    arg_set = set(arg_mismatch)
    summaries: List[str] = []
    for name in names:
        reasons: List[str] = []
        if name in ptr_set:
            reasons.append("pointer_type")
        if name in arg_set:
            reasons.append("argument_type")
        if not reasons:
            reasons.append("no_matching_overload")
        family = _call_shape_family(name)
        reason = _call_shape_reason_label(name, reasons, diag)
        suffix = "+".join(reasons)
        if family == "std" and reason != "std_call_shape":
            summaries.append(f"{name}:{family}:{reason}")
        else:
            summaries.append(f"{name}:{family}:{reason}:{suffix}")

    diag["call_shape_mismatches"] = _dedup_text_items(summaries, limit=12)
    diag.pop("no_matching_calls", None)
    diag.pop("ptr_type_mismatch_calls", None)
    diag.pop("arg_type_mismatch_calls", None)


def _normalize_unsupported_symbol_diagnostics(diag: Dict[str, List[str]]) -> None:
    """Do not label valid-but-wrong-shaped SVE overloads as unsupported APIs."""
    unsupported = list(diag.get("unsupported_symbols") or [])
    if not unsupported:
        return
    undeclared = set(diag.get("undeclared_identifiers") or [])
    call_related = set(diag.get("no_matching_calls") or [])
    call_related.update(diag.get("ptr_type_mismatch_calls") or [])
    call_related.update(diag.get("arg_type_mismatch_calls") or [])
    call_related.update(
        str(item).split(":", 1)[0]
        for item in (diag.get("call_shape_mismatches") or [])
        if str(item or "").strip()
    )
    kept: List[str] = []
    for name in unsupported:
        text = str(name or "").strip()
        if text.startswith("sv") and text not in undeclared and text in call_related:
            continue
        kept.append(text)
    if kept:
        diag["unsupported_symbols"] = _dedup_text_items(kept, limit=12)
    else:
        diag.pop("unsupported_symbols", None)


def _append_structured_compile_diagnostics(
    parts: List[str],
    diagnostics: Dict[str, Any],
    *,
    section_name: str = "[REMOTE_COMPILE_DIAGNOSTICS]",
    repair_rule: str = "",
) -> bool:
    """Render compile feedback as compact diagnostic buckets instead of raw clang text."""
    diag = _sanitize_compile_diagnostics_for_prompt(diagnostics or {})
    if not diag:
        return False
    if diag.get("non_constant_immediate_args"):
        immediate_args = list(diag.get("compile_time_immediate_args") or [])
        immediate_args.extend(diag.get("non_constant_immediate_args") or [])
        diag["compile_time_immediate_args"] = _dedup_text_items(immediate_args, limit=12)
        diag.pop("non_constant_immediate_args", None)

    if diag.get("undeclared_identifiers"):
        specialized_undeclared = set(diag.get("unsupported_symbols") or [])
        specialized_undeclared.update(diag.get("missing_helper_symbols") or [])
        specialized_undeclared.update(diag.get("missing_index_symbols") or [])
        plain_undeclared = [
            name for name in (diag.get("undeclared_identifiers") or [])
            if name not in specialized_undeclared
        ]
        if plain_undeclared:
            diag["undeclared_identifiers"] = _dedup_text_items(plain_undeclared, limit=12)
        else:
            diag.pop("undeclared_identifiers", None)

    _normalize_unsupported_symbol_diagnostics(diag)
    _collapse_call_shape_diagnostics(diag)

    evaluator_phase_values = (
        diag.pop("remote_failure_phase", None)
        or diag.pop("failure_phase", None)
        or diag.pop("phase", None)
        or []
    )
    evaluator_phase = str((evaluator_phase_values or [""])[0] if isinstance(evaluator_phase_values, list) else evaluator_phase_values).strip()
    evaluator_artifacts = diag.pop("remote_evaluator_artifacts", None) or []
    rendered_evaluator_failure = _render_remote_evaluator_failure(parts, evaluator_artifacts, phase=evaluator_phase)
    if not diag:
        return rendered_evaluator_failure

    primary_error_type = _primary_compile_error_type(diag)
    if primary_error_type == "unparsed_compile_failure":
        return rendered_evaluator_failure
    if primary_error_type == "syntax_token" and diag.get("undeclared_identifiers"):
        diag["cascaded_symbols"] = _dedup_text_items(diag.pop("undeclared_identifiers") or [], limit=12)

    parts.append("\n" + section_name)
    parts.append(f"- error_type: {primary_error_type}")
    ordered_keys = [
        "call_shape_mismatches",
        "unsupported_symbols",
        "missing_helper_symbols",
        "missing_index_symbols",
        "undeclared_identifiers",
        "cascaded_symbols",
        "missing_members",
        "unknown_type_names",
        "missing_include_headers",
        "ambiguous_calls",
        "compile_time_immediate_args",
        "type_mismatch_messages",
        "sve_tuple_access_messages",
        "sizeless_type_messages",
        "const_assignment_messages",
        "invalid_cast_messages",
        "invalid_address_messages",
        "invalid_cpp_construct_messages",
        "redefined_symbols",
        "compiler_suggestion",
        "expected_tokens",
        "syntax_signals",
        "diagnostic_locations",
        "source_contexts",
        "related_notes",
    ]
    for key in ordered_keys:
        values = diag.get(key) or []
        if values:
            parts.append(f"- {key}: " + ", ".join(values))

    if primary_error_type != "syntax_token" and _has_syntax_token_signal(diag):
        parts.append("- secondary_error_type: syntax_token")
    if diag.get("compiler_suggestion"):
        parts.append("- suggestion_trusted: false")

    return True


def _serial_reference_compile_failed(serial_ref_result: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(serial_ref_result, dict):
        return False
    compile_ok_raw = serial_ref_result.get("compile_ok", None)
    if compile_ok_raw is not None:
        try:
            if int(compile_ok_raw or 0) == 0:
                return True
        except Exception:
            if str(compile_ok_raw).strip().lower() in {"false", "no", "failed"}:
                return True
    reason = str(serial_ref_result.get("reason") or "").strip().lower()
    return "compile" in reason


def _append_symbol_closure_blocks(parts: List[str], static_verifier_report: Dict[str, Any]) -> None:
    report = static_verifier_report if isinstance(static_verifier_report, dict) else {}
    repair_payload = report.get("repair_prompt_payload")
    repair_payload = repair_payload if isinstance(repair_payload, dict) else {}
    symbol_targets = _clean_named_diag_map(
        repair_payload.get("symbol_closure_targets") or report.get("symbol_closure_targets") or {},
        per_key_limit=12,
    )
    remote_compile_diagnostics = _clean_named_diag_map(
        report.get("remote_compile_diagnostics") or {},
        per_key_limit=12,
    )
    remote_cpp17_compile_gate_diagnostics = _clean_named_diag_map(
        report.get("remote_cpp17_compile_gate_diagnostics") or {},
        per_key_limit=12,
    )

    if symbol_targets:
        parts.append("\n[LOCAL_SYMBOL_CLOSURE_TARGETS]")
        for key in [
            "defined_local_helpers",
            "missing_helper_symbols",
            "missing_index_symbols",
            "unsupported_builtin_symbols",
            "unsupported_sv_symbols",
            "syntax_signals",
        ]:
            values = symbol_targets.get(key) or []
            if values:
                parts.append(f"- {key}: " + ", ".join(values))
        parts.append(
            "- repair_rule: every helper-like symbol you keep calling must be defined in this same file before first use, or be inlined away."
        )
        parts.append(
            "- repair_rule: do not invent new *_local helper names unless you also emit their full definition in-file."
        )

    if remote_compile_diagnostics:
        _append_structured_compile_diagnostics(
            parts,
            remote_compile_diagnostics,
            section_name="[REMOTE_COMPILE_DIAGNOSTICS]",
            repair_rule="prioritize the exact unresolved symbols and syntax closure signals listed here before making any broader edits.",
        )

    if remote_cpp17_compile_gate_diagnostics:
        _append_structured_compile_diagnostics(
            parts,
            remote_cpp17_compile_gate_diagnostics,
            section_name="[REMOTE_CPP17_COMPILE_GATE_DIAGNOSTICS]",
            repair_rule="this candidate is blocked by the remote compile-only pre-gate before formal remote eval; clear the exact compile signals first.",
        )


def _append_signature_closure_blocks(parts: List[str], static_verifier_report: Dict[str, Any]) -> None:
    report = static_verifier_report if isinstance(static_verifier_report, dict) else {}
    repair_payload = report.get("repair_prompt_payload")
    repair_payload = repair_payload if isinstance(repair_payload, dict) else {}
    expected_signature_line = str(
        repair_payload.get("expected_signature_line")
        or report.get("expected_signature_line")
        or ""
    ).strip()
    actual_signature_line = str(
        repair_payload.get("actual_signature_line")
        or report.get("actual_signature_line")
        or ""
    ).strip()
    signature_check = repair_payload.get("signature_check")
    if not isinstance(signature_check, dict):
        signature_check = report.get("signature_check")
    signature_check = signature_check if isinstance(signature_check, dict) else {}
    mismatches = _clean_str_list(signature_check.get("mismatches"), limit=12)

    if not (expected_signature_line or actual_signature_line or mismatches):
        return

    parts.append("\n[TARGET_SIGNATURE_CLOSURE]")
    if expected_signature_line:
        parts.append(f"- expected_signature: {expected_signature_line}")
    if actual_signature_line:
        parts.append(f"- actual_signature: {actual_signature_line}")
    if mismatches:
        parts.append("- mismatches: " + ", ".join(mismatches))
    parts.append(
        "- repair_rule: the expected signature from the prompt prefix is the source of truth; do not preserve the current broken signature."
    )
    parts.append(
        "- repair_rule: repair the target function header first, then keep the body as intact as possible."
    )


def _build_controller_repair_template(
    *,
    static_verifier_report: Dict,
    repair_payload: Dict,
) -> List[str]:
    controller_action = str(repair_payload.get("controller_action", "")).strip() or "semantic_fix"
    spec = _CONTROLLER_REPAIR_TEMPLATE_SPECS.get(
        controller_action, _CONTROLLER_REPAIR_TEMPLATE_SPECS["semantic_fix"]
    )
    must_fix_tags = _clean_str_list(repair_payload.get("must_fix_tags"), limit=8)
    focus_areas = _clean_str_list(repair_payload.get("focus_areas"), limit=8)
    repair_steps = _clean_str_list(repair_payload.get("repair_steps"), limit=6)
    notes = _clean_str_list(repair_payload.get("notes"), limit=4)
    compile_tags = _clean_tag_name_list(static_verifier_report.get("compile_risk_tags"), limit=8)
    semantic_tags = _clean_tag_name_list(static_verifier_report.get("semantic_risk_tags"), limit=8)

    lines: List[str] = []
    lines.append("\n[LOCAL_REPAIR_TEMPLATE]")
    lines.append(f"- template_id: {controller_action}")
    lines.append(f"- template_title: {spec['title']}")
    lines.append(f"- objective: {spec['objective']}")

    if must_fix_tags:
        lines.append("- must_fix_tags: " + ", ".join(must_fix_tags))
    if focus_areas:
        lines.append("- focus_areas: " + ", ".join(focus_areas))
    if compile_tags:
        lines.append("- compile_tags: " + ", ".join(compile_tags))
    if semantic_tags:
        lines.append("- semantic_tags: " + ", ".join(semantic_tags))

    for item in spec.get("must", []):
        txt = str(item).strip()
        if txt:
            lines.append(f"- template_must: {txt}")
    for item in spec.get("avoid", []):
        txt = str(item).strip()
        if txt:
            lines.append(f"- template_avoid: {txt}")
    for item in spec.get("checklist", []):
        txt = str(item).strip()
        if txt:
            lines.append(f"- template_check: {txt}")
    for item in repair_steps:
        lines.append(f"- controller_step: {item}")
    for item in notes:
        lines.append(f"- controller_note: {item}")

    lines.append(
        "- Follow this template as the primary repair policy. Use the verifier report as evidence, not as a license to do unrelated rewrites."
    )
    lines.append("")
    return lines


def build_pre_remote_local_repair_prompt(
    *,
    spec_text: str,
    prompt_prefix: str,
    prev_completion: str,
    static_verifier_report: Dict,
) -> str:
    parts: List[str] = []

    if COMPLETION_MODE == COMPLETION_MODE_FULL:
        parts.append(
            "[LOCAL_PRE_REMOTE_REPAIR_TASK]\n"
            "You will be given:\n"
            "1) SPEC\n"
            "2) Previous completion (a standalone C/C++ source file)\n"
            "3) Local static verifier guidance\n\n"
            "Constraints:\n"
            "- Output ONLY the corrected full standalone C/C++ source file.\n"
            "- Make the minimal repair needed to fix the flagged local hazards.\n"
            "- Preserve the exact target function signature from the prompt prefix and the intended algorithm.\n"
            "- Do not introduce task-id-specific heuristics.\n"
            "- If you use Arm SVE intrinsics, every sv* call must be valid under <arm_sve.h>.\n"
        )
    else:
        parts.append(
            "[LOCAL_PRE_REMOTE_REPAIR_TASK]\n"
            "You will be given:\n"
            "1) SPEC\n"
            "2) Previous completion snippet\n"
            "3) Local static verifier guidance\n\n"
            "Constraints:\n"
            "- Output ONLY the corrected completion snippet.\n"
            "- Make the minimal repair needed to fix the flagged local hazards.\n"
            "- Preserve the exact target function signature from the prompt prefix and the intended algorithm.\n"
            "- Do not introduce task-id-specific heuristics.\n"
            "- If you use Arm SVE intrinsics, every sv* call must be valid under <arm_sve.h>.\n"
        )

    parts.append("\n[SPEC]\n" + spec_text.strip() + "\n")
    if "[PROMPT_PREFIX]" not in spec_text:
        parts.append("\n[PROMPT_PREFIX]\n" + prompt_prefix.rstrip() + "\n")
    parts.append("\n[PREV_COMPLETION]\n" + prev_completion.strip() + "\n")

    repair_payload = static_verifier_report.get("repair_prompt_payload") or {}
    if isinstance(repair_payload, dict):
        parts.extend(
            _build_controller_repair_template(
                static_verifier_report=static_verifier_report,
                repair_payload=repair_payload,
            )
        )

    compile_tags = _clean_tag_name_list(static_verifier_report.get("compile_risk_tags"), limit=8)
    semantic_tags = _clean_tag_name_list(static_verifier_report.get("semantic_risk_tags"), limit=8)
    controller_action = ""
    primary_focus_area = ""
    must_fix_tags: List[str] = []
    focus_areas: List[str] = []
    repair_steps: List[str] = []
    if isinstance(repair_payload, dict):
        controller_action = str(repair_payload.get("controller_action") or "").strip()
        primary_focus_area = str(repair_payload.get("primary_focus_area") or "").strip()
        must_fix_tags = _clean_str_list(repair_payload.get("must_fix_tags"), limit=8)
        focus_areas = _clean_str_list(repair_payload.get("focus_areas"), limit=8)
        repair_steps = _clean_str_list(repair_payload.get("repair_steps"), limit=6)
    _append_signature_closure_blocks(parts, static_verifier_report)
    _append_symbol_closure_blocks(parts, static_verifier_report)

    parts.append("\n[LOCAL_STATIC_VERIFIER_REPORT]")
    parts.append(
        f"- compile_risk_count: {len(compile_tags)}; semantic_risk_count: {len(semantic_tags)}"
    )
    if controller_action:
        parts.append(f"- controller_action: {controller_action}")
    if primary_focus_area:
        parts.append(f"- primary_focus_area: {primary_focus_area}")
    if must_fix_tags:
        parts.append("- must_fix_tags: " + ", ".join(must_fix_tags))
    if focus_areas:
        parts.append("- focus_areas: " + ", ".join(focus_areas))
    if compile_tags:
        parts.append("- compile_tags: " + ", ".join(compile_tags))
    if semantic_tags:
        parts.append("- semantic_tags: " + ", ".join(semantic_tags))
    for item in repair_steps[:4]:
        txt = str(item).strip()
        if txt:
            parts.append(f"- repair_step: {txt}")
    parts.append(
        "- Treat this as a pre-remote local repair pass: fix the flagged local families first, keep the edit scope minimal, and avoid speculative rewrites."
    )
    parts.append("\n[END]")
    return "\n".join(parts)


def bootstrap_has_explicit_vectorization_features(bootstrap_text: str) -> bool:
    text = str(bootstrap_text or "")
    if not text.strip():
        return False
    markers = (
        "vector_range(",
        "active_lanes<",
        "VL<",
        "prefix_scan(",
        "compact_write:",
        "arg_reduce:",
        "scan:",
        "reduce:",
        "any_lane(",
        "closed_form:",
    )
    return any(marker in text for marker in markers)


ROUTED_FEEDBACK_STYLE_ALIASES = {
    "routed",
    "auto",
    "route",
    "routing",
    "routed_by_type",
    "type_routed",
}


BOOTSTRAP_VECTORIZATION_GUARD = (
    "- Do not force full-function vectorization. Use SVE only for regions with "
    "explicit vector_range loops, C++-style active_lanes/VL vector loops, "
    "compact_write, or reduction structure in the bootstrap; "
    "keep the remaining loop-carried, "
    "dynamic-container, string, or scalar-control logic unchanged."
)


def requested_routed_feedback_style(style: str) -> bool:
    return str(style or "").strip().lower() in ROUTED_FEEDBACK_STYLE_ALIASES


def build_remote_feedback_prompt(
    *,
    spec_text: str,
    prompt_prefix: str,
    prev_completion: str,
    remote_result: Dict,
    compile_only: bool,
    static_verifier_report: Optional[Dict] = None,
    serial_ref_completion: Optional[str] = None,
    serial_bootstrap_completion: Optional[str] = None,
    serial_ref_result: Optional[Dict] = None,
    serial_diff: str = "",
    serial_mismatch_info: Optional[Dict] = None,
    serial_code_max_chars: int = 0,
    serial_feedback_style: str = "pseudocode",
    serial_feedback_style_requested: str = "",
    serial_pseudocode_max_chars: int = 6000,
    serial_diff_max_chars: int = 4000,
    serial_bootstrap_stage: str = "",
    serial_bootstrap_reason: str = "",
    serial_bootstrap_source: str = "",
    serial_ast_bootstrap_pseudocode: str = "",
    serial_ast_bootstrap_source: str = "",
    compile_feedback_style: str = "structured",
    disable_bootstrap_guard: bool = False,
) -> str:
    """
    Build a compact, high-signal prompt for the remote feedback repair loop.

    Key goals:
    - Keep SPEC + previous completion.
    - Remove noisy infra details (tmp dirs, ssh cmds, etc.) that distract the model.
    - Prefer structured compiler diagnostics; fall back to raw compiler excerpts when
      no known diagnostic bucket is recognized.
    """
    parts: List[str] = []

    reason = str(remote_result.get("reason", "")).strip()
    passed = remote_result.get("passed", None)
    simdbench_raw = str(remote_result.get("simdbench_raw_result", "")).strip()
    # Keep only the first line of simdbench_raw_result (often "compilation failed: ..." / "logical bug")
    simdbench_raw_first = simdbench_raw.splitlines()[0].strip() if simdbench_raw else ""
    simdbench_raw_lower = simdbench_raw_first.lower()
    compile_feedback_style_norm = str(compile_feedback_style or "structured").strip().lower()
    if compile_feedback_style_norm not in {"structured", "hybrid", "raw"}:
        compile_feedback_style_norm = "structured"
    runtime_like = (
        ("runtime failed" in simdbench_raw_lower)
        or ("segmentation fault" in simdbench_raw_lower)
        or ("sigsegv" in simdbench_raw_lower)
        or ("glibc" in simdbench_raw_lower)
        or ("double free" in simdbench_raw_lower)
        or ("invalid pointer" in simdbench_raw_lower)
    )
    if runtime_like:
        # Some runners label these under compile_error; override so the model focuses on bounds/memory safety.
        reason = "runtime_error"


    if COMPLETION_MODE == COMPLETION_MODE_FULL:
        parts.append(
            "[REMOTE_FEEDBACK_REPAIR_TASK]\n"
            "You will be given:\n"
            "1) SPEC (task description + reference prompt prefix)\n"
            "2) Previous completion (a standalone C/C++ source file that was compiled/run as-is)\n"
            "3) Remote compile/run feedback\n\n"
            "Constraints:\n"
            "- Output ONLY the corrected full standalone C/C++ source file.\n"
            "- Do NOT include any prompt metadata markers like [SPEC], [END], etc. Only code.\n"
            "- Keep the overall intent consistent with the SPEC.\n"
            "- For compile/API errors, fix locally. For logical errors with Bootstrap, follow Bootstrap semantics even if the affected loop must be rewritten. Avoid unrelated changes.\n- IMPORTANT C pointer rule: do NOT multiply indices by sizeof(T) when doing pointer arithmetic on a T* (use element indexing).\n- IMPORTANT C pointer rule: do NOT multiply indices by sizeof(T) when doing pointer arithmetic on a T* (use element indexing).\n"
            "- Preserve the exact target function signature from the prompt prefix; do not keep a broken current signature just because it already appears in the file.\n"
            "- If you use Arm SVE intrinsics, do NOT invent intrinsic names; every sv* call must be valid under <arm_sve.h>.\n"
            "- Common SVE pitfalls: svwhilelt_* may need explicit (uint64_t) casts; *_x intrinsics take a predicate argument; "
            "svcntp_* requires 2 args (pg, pred) and counts predicate bits (NOT popcount of integer vectors); svindex_* takes 2 scalar args (base, step); svptest_* expects predicates, not vectors.\n"
        )
    else:
        parts.append(
            "[REMOTE_FEEDBACK_REPAIR_TASK]\n"
            "You will be given:\n"
            "1) SPEC (task + prompt prefix)\n"
            "2) Previous completion snippet (what will be appended to the prompt prefix)\n"
            "3) Remote compile/run feedback\n\n"
            "Constraints:\n"
            "- Output ONLY the corrected completion snippet. Do NOT repeat the prompt prefix.\n"
            "- Keep the overall intent consistent with the SPEC.\n"
            "- For compile/API errors, fix locally. For logical errors with Bootstrap, follow Bootstrap semantics even if the affected loop must be rewritten. Avoid unrelated changes.\n- IMPORTANT C pointer rule: do NOT multiply indices by sizeof(T) when doing pointer arithmetic on a T* (use element indexing).\n"
            "- Preserve the exact target function signature from the prompt prefix; do not keep a broken current signature just because it already appears in the snippet.\n"
            "- If you use Arm SVE intrinsics, do NOT invent intrinsic names; every sv* call must be valid under <arm_sve.h>.\n"
            "- Common SVE pitfalls: svwhilelt_* may need explicit (uint64_t) casts; *_x intrinsics take a predicate argument; "
            "svcntp_* requires 2 args (pg, pred) and counts predicate bits (NOT popcount of integer vectors); svindex_* takes 2 scalar args (base, step); svptest_* expects predicates, not vectors.\n"
        )

    if compile_only:
        parts.append("- NOTE: This remote feedback is compile-only (no runtime test executed). Fix compile errors.\n")

    parts.append("\n[SPEC]\n" + spec_text.strip() + "\n")
    # `spec_text` already contains the prompt prefix in the common code paths; avoid duplicating it.
    if "[PROMPT_PREFIX]" not in spec_text:
        parts.append("\n[PROMPT_PREFIX]\n" + prompt_prefix.rstrip() + "\n")
    parts.append("\n[PREV_COMPLETION]\n" + prev_completion.strip() + "\n")

    # Do not surface the static-verifier repair controller/template in remote
    # repair prompts. Keep the verifier result available for logs and
    # structured compile diagnostics below, but let Bootstrap/remote feedback
    # drive the model-facing repair instruction.
    static_report_blocks_rendered = False

    # Optional: SERIAL reference (scalar) implementation for phased bootstrap / differential debugging
    if serial_ref_completion and not _serial_reference_compile_failed(serial_ref_result):
        feedback_style = str(serial_feedback_style or "pseudocode").strip().lower()
        pseudocode_styles = {
            "pseudocode",
            "constrained_pseudocode",
            "dataflow_pseudocode",
            "unified_dataflow_pseudocode",
            "both",
        }
        dataflow_styles = {"constrained_pseudocode", "dataflow_pseudocode"}
        if feedback_style not in (pseudocode_styles | {"code"}):
            feedback_style = "pseudocode"
        # The validated serial reference remains the oracle for correctness and
        # serial-vs-SIMD diffs.  The optional bootstrap completion is only for
        # the semantic hint shown to the model, so benchmark helper-heavy scalar
        # code does not leak into the repair prompt.
        serial_semantics_completion = str(serial_bootstrap_completion or serial_ref_completion or "").strip()
        serial_code = serial_semantics_completion
        if feedback_style in {"code", "both"}:
            if serial_code_max_chars and serial_code_max_chars > 0 and len(serial_code) > int(serial_code_max_chars):
                serial_code = _truncate_middle(serial_code, int(serial_code_max_chars))
        serial_pseudocode = ""
        if feedback_style in pseudocode_styles:
            serial_pseudocode = str(serial_ast_bootstrap_pseudocode or "").strip()
            if serial_pseudocode:
                if serial_pseudocode_max_chars and serial_pseudocode_max_chars > 0 and len(serial_pseudocode) > int(serial_pseudocode_max_chars):
                    serial_pseudocode = _truncate_middle(serial_pseudocode, int(serial_pseudocode_max_chars))
            else:
                pseudo_func_name = None
                try:
                    pseudo_func_name = extract_function_name_from_prompt_prefix(prompt_prefix)
                except Exception:
                    pseudo_func_name = None
                if feedback_style in dataflow_styles:
                    serial_pseudocode = build_dataflow_serial_reference_pseudocode(
                        serial_semantics_completion,
                        func_name=pseudo_func_name,
                        prompt_prefix=prompt_prefix,
                        max_chars=int(serial_pseudocode_max_chars or 0),
                    )
                else:
                    serial_pseudocode = build_serial_reference_pseudocode(
                        serial_semantics_completion,
                        func_name=pseudo_func_name,
                        prompt_prefix=prompt_prefix,
                        max_chars=int(serial_pseudocode_max_chars or 0),
                    )

        if serial_bootstrap_stage:
            parts.append("\n[BOOTSTRAP]")
            parts.append(
                f"- stage: {serial_bootstrap_stage}"
                + (f"; reason: {serial_bootstrap_reason}" if serial_bootstrap_reason else "")
            )
            parts.append(
                "- Use the semantic bootstrap below as a task-specific repair guide for dataflow, indexing, predicates, reductions, and vectorizable regions."
            )
            parts.append(
                "- Keep the exact target function signature from the prompt prefix."
            )
            if not bool(disable_bootstrap_guard) and serial_pseudocode:
                routed_request = requested_routed_feedback_style(
                    serial_feedback_style_requested or serial_feedback_style
                )
                conservative_nonvector_route = (
                    feedback_style != "pseudocode"
                    and not bootstrap_has_explicit_vectorization_features(serial_pseudocode)
                )
                if routed_request or conservative_nonvector_route:
                    parts.append(BOOTSTRAP_VECTORIZATION_GUARD)
            parts.append("[/BOOTSTRAP]\n")

        if feedback_style in pseudocode_styles and serial_pseudocode:
            parts.append("\n[SEMANTIC_BOOTSTRAP]")
            parts.append(serial_pseudocode)
            parts.append("[/SEMANTIC_BOOTSTRAP]\n")

        if feedback_style in {"code", "both"}:
            parts.append("\n[SERIAL_REFERENCE_COMPLETION]\n" + serial_code + "\n")

        if serial_ref_result:
            parts.append("\n[SERIAL_REFERENCE_RESULT_SUMMARY]")
            for k in ["compile_ok", "run_ok", "passed", "reason", "compile_rc", "run_rc"]:
                v = serial_ref_result.get(k, None)
                if v is not None:
                    parts.append(f"- {k}: {v}")
            sraw = str(serial_ref_result.get("simdbench_raw_result", "")).strip()
            sraw_first = sraw.splitlines()[0].strip() if sraw else ""
            if sraw_first:
                parts.append(f"- simdbench_raw_result: {sraw_first}")

            # Serial compile diagnostics are deliberately not surfaced to the repair model:
            # a serial-reference compile failure means the auxiliary oracle is unusable,
            # not that the target SVE function should chase serial compiler noise.
            s_rtail = str(serial_ref_result.get("run_log_tail", "")).strip()
            s_r_excerpt = _extract_interesting_log_lines(s_rtail, max_lines=40) if s_rtail else ""
            if s_r_excerpt:
                parts.append("\n[SERIAL_RUN_LOG_EXCERPT]\n" + s_r_excerpt + "\n")

        if serial_diff and feedback_style in {"code", "both"}:
            diff_txt = str(serial_diff).strip()
            if serial_diff_max_chars and serial_diff_max_chars > 0 and len(diff_txt) > int(serial_diff_max_chars):
                diff_txt = _truncate_middle(diff_txt, int(serial_diff_max_chars))
            parts.append("\n[SERIAL_VS_SVE_DIFF]\n" + diff_txt + "\n")
            parts.append(
                "\n[NOTE]\nThe SERIAL reference implementation passed the tests; treat it as the oracle. "
                "Compare SERIAL vs SVE carefully and apply minimal fixes (common bugs: indexing, tail handling, predicate usage, "
                "wrong lane width b32/b64, wrong load/store type).\n"
            )
        elif serial_diff and feedback_style in {
            "pseudocode",
            "constrained_pseudocode",
            "dataflow_pseudocode",
            "unified_dataflow_pseudocode",
        }:
            parts.append(
                "\n[NOTE]\nSerial-vs-SVE code diff is intentionally omitted in pseudocode feedback mode. "
                "Use the semantic pseudocode and mismatch facts instead of copying scalar code.\n"
            )

        if isinstance(serial_mismatch_info, dict) and serial_mismatch_info:
            parts.append("\n[SERIAL_MISMATCH_HARNESS_RESULT]")
            for k in ["mismatch", "index", "expected", "got", "note", "error"]:
                if k in serial_mismatch_info:
                    parts.append(f"- {k}: {serial_mismatch_info.get(k)}")
            mismatch_examples = serial_mismatch_info.get("mismatch_examples")
            if isinstance(mismatch_examples, list) and mismatch_examples:
                parts.append("- mismatch_examples:")
                for idx, item in enumerate(mismatch_examples[:16]):
                    if isinstance(item, dict):
                        parts.append(
                            "  - "
                            f"[{idx}] index={item.get('index')} "
                            f"expected={item.get('expected')} "
                            f"got={item.get('got')} "
                            f"note={item.get('note')}"
                        )
            parts.append("")

    parts.append("\n[REMOTE_RESULT_SUMMARY]")
    # Display values (remote_result can be inconsistent for runtime crashes).
    compile_ok_disp = remote_result.get("compile_ok", None)
    run_ok_disp = remote_result.get("run_ok", None)
    passed_disp = remote_result.get("passed", None)
    reason_disp = remote_result.get("reason", None)
    compile_rc_disp = remote_result.get("compile_rc", None)
    run_rc_disp = remote_result.get("run_rc", None)

    if runtime_like:
        # Runtime-like implies the binary actually ran; treat this as a runtime error even if the runner mislabels it.
        reason_disp = "runtime_error"
        compile_ok_disp = 1

    for k, v in [
        ("compile_ok", compile_ok_disp),
        ("run_ok", run_ok_disp),
        ("passed", passed_disp),
        ("reason", reason_disp),
        ("compile_rc", compile_rc_disp),
        ("run_rc", run_rc_disp),
    ]:
        if v is not None:
            parts.append(f"- {k}: {v}")
    if simdbench_raw_first:
        summary_compile_like = (
            "compilation failed" in simdbench_raw_lower
            or str(reason_disp or "").strip().lower() in {"compile_error", "compile_failed"}
            or (compile_ok_disp is not None and str(compile_ok_disp).strip() in {"0", "False", "false"})
        )
        if compile_feedback_style_norm == "structured" and summary_compile_like and not runtime_like:
            parts.append("- simdbench_result: compile_error")
        else:
            parts.append(f"- simdbench_raw_result: {simdbench_raw_first}")

    # Logs: extract only interesting lines. For compile logs, default to structured buckets
    # so repair models do not overfit to raw clang paths, candidate notes, or shell noise.
    ctail = str(remote_result.get("compile_log_tail", "")).strip()
    rtail = str(remote_result.get("run_log_tail", "")).strip()
    # Some runners may stuff runtime crash traces into compile_log_tail.
    # If we detected a runtime-like failure and run_log_tail is empty, surface it as RUN log instead.
    if runtime_like and (not rtail) and ctail:
        rtail, ctail = ctail, ""


    c_excerpt = _extract_interesting_log_lines(ctail, max_lines=60) if ctail else ""
    r_excerpt = _extract_interesting_log_lines(rtail, max_lines=60) if rtail else ""

    report_compile_diagnostics = {}
    if isinstance(static_verifier_report, dict):
        report_compile_diagnostics = _clean_named_diag_map(
            static_verifier_report.get("remote_compile_diagnostics") or {},
            per_key_limit=12,
        )
    inline_compile_diagnostics = extract_remote_compile_diagnostics(ctail) if ctail else {}
    remote_failure_phase = str(reason_disp or reason or "").strip()
    if remote_failure_phase:
        if report_compile_diagnostics:
            report_compile_diagnostics = dict(report_compile_diagnostics)
            report_compile_diagnostics.setdefault("remote_failure_phase", [remote_failure_phase])
        if inline_compile_diagnostics:
            inline_compile_diagnostics = dict(inline_compile_diagnostics)
            inline_compile_diagnostics.setdefault("remote_failure_phase", [remote_failure_phase])
    evaluator_artifacts = _remote_evaluator_artifacts_from_result(remote_result, remote_failure_phase)
    if evaluator_artifacts:
        artifact_diag = {
            "remote_evaluator_artifacts": evaluator_artifacts,
            "remote_failure_phase": [remote_failure_phase or "remote_evaluator_failure"],
        }
        if inline_compile_diagnostics:
            inline_compile_diagnostics = _merge_compile_diagnostic_maps(inline_compile_diagnostics, artifact_diag)
        elif report_compile_diagnostics:
            report_compile_diagnostics = _merge_compile_diagnostic_maps(report_compile_diagnostics, artifact_diag)
        else:
            inline_compile_diagnostics = artifact_diag
    render_structured_compile = compile_feedback_style_norm in {"structured", "hybrid"}
    merged_compile_diagnostics = _merge_compile_diagnostic_maps(
        report_compile_diagnostics,
        inline_compile_diagnostics,
    )
    structured_compile_rendered = False

    if render_structured_compile and merged_compile_diagnostics and not static_report_blocks_rendered:
        structured_compile_rendered = _append_structured_compile_diagnostics(
            parts,
            merged_compile_diagnostics,
            section_name="[REMOTE_COMPILE_DIAGNOSTICS]",
        )

    include_raw_compile_excerpt = False
    if c_excerpt:
        if compile_feedback_style_norm in {"hybrid", "raw"}:
            include_raw_compile_excerpt = True
        elif compile_feedback_style_norm == "structured" and not structured_compile_rendered:
            include_raw_compile_excerpt = True

    if include_raw_compile_excerpt:
        parts.append("\n[COMPILE_LOG_EXCERPT]\n" + c_excerpt + "\n")
    if r_excerpt:
        parts.append("\n[RUN_LOG_EXCERPT]\n" + r_excerpt + "\n")

    # If logical_bug gives no details, push the model to re-derive from spec rather than chase noise.
    if reason == "logical_bug" and (not r_excerpt) and simdbench_raw_first.lower() in {"logical bug", "logical_bug"}:
        parts.append(
            "\n[NOTE]\nThe remote feedback only says 'logical bug' without mismatch details. "
            "Re-derive the correct algorithm strictly from the SPEC, and fix the code accordingly. "
            "Do not guess from remote paths/commands.\n"
        )

    parts.append("\nYour goal is 根据反馈信息对代码进行修正，使其能够成功编译，并在有运行测试时通过测试。\n")
    parts.append("[END]")
    return "\n".join(parts)


def make_remote_feedback_text(res: Dict[str, Any], tail_chars: int = 8000, max_chars: int = 4000) -> str:
    """Render a compact text summary of a remote simdbench_eval result dict.

    This is used as context for the serial-reference model (and for debugging).
    We intentionally keep it short and tail-truncated because logs can be huge.
    """
    if not isinstance(res, dict):
        return ""

    def _clip(x: Any, n: int) -> str:
        if x is None:
            return ""
        s = str(x)
        if n is not None and n > 0 and len(s) > n:
            return s[-n:]
        return s

    compile_ok = res.get("compile_ok")
    run_ok = res.get("run_ok")
    passed = res.get("passed")
    reason = res.get("reason")
    raw = res.get("simdbench_raw_result") or res.get("raw_result")

    compile_rc = res.get("compile_rc")
    run_rc = res.get("run_rc")
    compile_cmd = res.get("compile_cmd")
    run_cmd = res.get("run_cmd")

    compile_log_tail = _clip(res.get("compile_log_tail"), tail_chars)
    run_log_tail = _clip(res.get("run_log_tail"), tail_chars)

    parts: List[str] = []
    parts.append("[REMOTE_RESULT_SUMMARY]")
    parts.append(f"passed={passed} compile_ok={compile_ok} run_ok={run_ok} reason={reason} raw={raw}")
    if compile_rc is not None or run_rc is not None:
        parts.append(f"compile_rc={compile_rc} run_rc={run_rc}")

    if compile_cmd:
        parts.append("[COMPILE_CMD]")
        parts.append(_clip(compile_cmd, 1000))
    if run_cmd:
        parts.append("[RUN_CMD]")
        parts.append(_clip(run_cmd, 1000))

    if compile_log_tail:
        compile_diag = extract_remote_compile_diagnostics(compile_log_tail)
        if compile_diag:
            diag_parts: List[str] = []
            rendered_diag = _append_structured_compile_diagnostics(
                diag_parts,
                compile_diag,
                section_name="[COMPILE_DIAGNOSTICS]",
                repair_rule="use these structured compile diagnostics instead of copying raw compiler output.",
            )
            if rendered_diag:
                parts.extend(diag_parts)
            else:
                parts.append("[COMPILE_LOG_TAIL]")
                parts.append(compile_log_tail)
        else:
            parts.append("[COMPILE_LOG_TAIL]")
            parts.append(compile_log_tail)
    if run_log_tail:
        parts.append("[RUN_LOG_TAIL]")
        parts.append(run_log_tail)

    out = "\n".join(parts).strip()
    if max_chars and len(out) > max_chars:
        out = out[-max_chars:]
        out = "[...snip...]\n" + out
    return out + "\n"
def remote_feedback_loop(
    *,
    model,
    tok,
    task_id: str,
    sample_idx: int,
    intrinsic: str,
    prompt_prefix: str,
    spec_text: str,
    completion_in: str,
    func_name: Optional[str] = None,
    whitelist_set: Optional[Set[str]],
    whitelist_list: Optional[List[str]],
    op_index: Optional[Dict[str, List[str]]],
    sigs: Dict[str, List[List[str]]],
    rets: Optional[Dict[str, List[str]]] = None,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    rounds: int,
    remote_early_stop_no_improve: bool = False,
    remote_user: str,
    remote_host: str,
    remote_port: int,
    remote_ssh_key: str,
    remote_tmp_root: str,
    remote_eval_mode: str,      # compile_only | cmd | simdbench_one
    remote_eval_cmd: str,       # used when remote_eval_mode=cmd
    remote_simdbench_eval: str, # used when remote_eval_mode=simdbench_one
    remote_simdbench_problem_file: str,
    remote_simdbench_scalar_problem_file: str,
    remote_simdbench_k: str,
    remote_simdbench_n_workers: int,
    remote_simdbench_output_path: str,
    remote_simdbench_tail_chars: int,
    remote_simdbench_compile_timeout: float,
    remote_compiler: str,
    remote_cflags: str,
    remote_timeout: int,
    remote_flock_path: str,
    remote_no_strict_hostkey: bool,
    reapply_name_shape: bool,
    reapply_name_max_iters: int,
    reapply_shape_max_iters: int,
    print_summary: bool,
    print_prompts: bool,
    dump_dir: Optional[Path],
    prompt_max_chars: int,
    # Optional SERIAL reference generation for differential debugging
    serial_model=None,
    serial_tok=None,
    serial_llm_backend: str = "none",
    serial_api_model: str = "",
    serial_fallback: bool = False,
    serial_do_sample: bool = False,
    serial_temperature: float = 0.2,
    serial_top_p: float = 0.9,
    serial_repetition_penalty: float = 1.0,
    serial_max_new_tokens: int = 1024,
    serial_prompt_max_chars: int = 0,
    serial_diff_max_chars: int = 4000,
    serial_diff_context_lines: int = 3,
    serial_cache: Any = None,
    serial_ref_max_attempts_per_task: int = 5,
    serial_use_solution_scalar: bool = False,
    serial_solution_scalar: str = "",
    serial_audited_nohelper_scalar: str = "",
    serial_entrypoint_scalar: str = "",
    serial_audited_nohelper_entrypoint: str = "",
    serial_audited_nohelper_source: str = "",
    serial_entrypoint_simd: str = "",
    serial_eval_intrinsic: str = "scalar",
    serial_mismatch_harness: bool = False,
    serial_mismatch_max_new_tokens: int = 512,
    serial_mismatch_prompt_max_chars: int = 20000,
    serial_bootstrap_mode: str = "phased",
    serial_feedback_style: str = "pseudocode",
    serial_pseudocode_max_chars: int = 6000,
    serial_ast_bootstrap_pseudocode: str = "",
    serial_ast_bootstrap_style: str = "",
    serial_ast_bootstrap_pattern: str = "",
    serial_ast_bootstrap_reason: str = "",
    serial_ast_bootstrap_source: str = "",
    static_verifier: bool = False,
    static_verifier_script: str = "",
    static_verifier_timeout: int = 20,
    remote_cpp17_compile_gate: bool = False,
    remote_cpp17_compile_gate_mode: str = "off",
    remote_cpp17_compile_gate_compiler: str = "",
    remote_cpp17_compile_gate_cflags: str = "",
    remote_cpp17_compile_gate_timeout: int = 20,
    remote_cpp17_compile_gate_max_repair_iters: int = 2,
    nl_description_ds_r1: str = "",
    serial_c_code: str = "",
    test_harness_code: str = "",
    source_type: str = "",
    problem_type: str = "",
    problem_subtype: str = "",
    source_name: str = "",
    remote_perf_precheck: bool = False,
    remote_perf_precheck_optimization: str = "-O1",
    remote_perf_precheck_n_reputation: int = 1,
    remote_perf_precheck_n_workers: int = 1,
    remote_perf_precheck_timeout: float = 30.0,
    remote_perf_precheck_python: str = "python3",
    remote_perf_precheck_repo_root: str = "",
    semantic_plan: Optional[Dict[str, Any]] = None,
    remote_score_mode: str = "legacy",
    remote_repair_cursor_mode: str = "latest",
    remote_semantic_no_improve_patience: int = 0,
    remote_compile_feedback_style: str = "structured",
    disable_bootstrap_guard: bool = False,
) -> Tuple[str, Dict]:
    serial_feedback_style_requested = str(serial_feedback_style or "pseudocode").strip().lower()
    serial_feedback_route_info = classify_serial_feedback_route(
        serial_feedback_style_requested,
        task_id=task_id,
        problem_type=problem_type,
        problem_subtype=problem_subtype,
        source_name=source_name,
        source_type=source_type,
        entrypoint_simd=serial_entrypoint_simd or func_name or "",
        prompt_prefix=prompt_prefix,
    )
    if str(serial_ast_bootstrap_pseudocode or "").strip() and serial_feedback_style_requested in {
        "routed",
        "auto",
        "route",
        "routing",
        "routed_by_type",
        "type_routed",
    }:
        ast_style = str(serial_ast_bootstrap_style or "").strip()
        if ast_style in {
            "pseudocode",
            "dataflow_pseudocode",
            "unified_dataflow_pseudocode",
            "constrained_pseudocode",
            "code",
            "both",
        }:
            serial_feedback_route_info = {
                "style": ast_style,
                "source": str(serial_ast_bootstrap_source or "serial_ast_bootstrap_jsonl"),
                "pattern": str(serial_ast_bootstrap_pattern or ""),
                "reason": str(serial_ast_bootstrap_reason or "AST-derived serial bootstrap route."),
            }
    serial_feedback_style_effective = str(serial_feedback_route_info.get("style") or "pseudocode")
    if serial_feedback_style_effective not in {
        "pseudocode",
        "dataflow_pseudocode",
        "unified_dataflow_pseudocode",
        "constrained_pseudocode",
        "code",
        "both",
    }:
        serial_feedback_style_effective = "pseudocode"

    info: Dict = {"enabled": True, "rounds": rounds, "history": []}
    info["early_stop_no_improve"] = {"enabled": bool(remote_early_stop_no_improve), "triggered": False}
    info["scoring"] = {
        "mode": str(remote_score_mode or "legacy"),
        "repair_cursor_mode_requested": str(remote_repair_cursor_mode or "latest"),
        "repair_cursor_mode": "latest",
        "best_fallback_enabled": False,
        "semantic_no_improve_patience": int(max(0, remote_semantic_no_improve_patience or 0)),
    }
    info["compile_feedback_style"] = str(remote_compile_feedback_style or "structured")
    info["static_verifier"] = {
        "enabled": bool(static_verifier and static_verifier_script),
        "script": static_verifier_script if static_verifier else "",
        "timeout_s": int(static_verifier_timeout),
        "history": [],
        "final": None,
    }
    info["remote_cpp17_compile_gate"] = {
        "enabled": bool(remote_cpp17_compile_gate and str(remote_cpp17_compile_gate_mode or "off") != "off"),
        "mode": str(remote_cpp17_compile_gate_mode or "off"),
        "compiler": str(remote_cpp17_compile_gate_compiler or remote_compiler or ""),
        "cflags": str(remote_cpp17_compile_gate_cflags or ""),
        "timeout_s": int(remote_cpp17_compile_gate_timeout or 20),
        "max_repair_iters": int(max(0, remote_cpp17_compile_gate_max_repair_iters or 0)),
        "history": [],
        "blocked": False,
        "blocked_rounds": [],
        "recovered_rounds": [],
        "final": None,
        "availability_errors": [],
    }
    info["remote_perf_precheck"] = {
        "enabled": bool(remote_perf_precheck),
        "optimization": str(remote_perf_precheck_optimization or ""),
        "n_reputation": int(max(1, remote_perf_precheck_n_reputation or 1)),
        "n_workers": int(max(1, remote_perf_precheck_n_workers or 1)),
        "timeout_s": float(remote_perf_precheck_timeout or 30.0),
        "repo_root": str(remote_perf_precheck_repo_root or ""),
        "python": str(remote_perf_precheck_python or "python3"),
        "history": [],
        "final": None,
        "availability_errors": [],
    }
    # SERIAL reference fallback (always log a summary into meta.jsonl for observability)

    info["serial_ref"] = {
        "enabled": bool(serial_fallback),
        "triggered": False,
        "attempted": False,
        "serial_model_available": serial_model is not None,
        "bootstrap_mode": str(serial_bootstrap_mode or "phased"),
        "serial_bootstrap_stage": "none",
        "serial_bootstrap_reason": "",
        "serial_bootstrap_source": "",
        "serial_bootstrap_attempt_source": "",
        "serial_ref_validated": False,
        "serial_ref_failure_kind": "none",
        "serial_ref_budget_consumed": False,
        "serial_ref_budget_consumed_count": 0,
        "backend": serial_llm_backend if serial_fallback else None,
        "model": (
            getattr(serial_model, "model", None)
            or getattr(serial_model, "model_id", None)
            or getattr(serial_model, "name", None)
            or (serial_api_model if serial_fallback else None)
        ),
        "cache_enabled": serial_cache is not None,
        "cache_path": str(getattr(serial_cache, "path", "")) if serial_cache is not None else "",
        "cache_hit": False,
        "reused": False,
        "saved_to_cache": False,
        "attempts_used": 0,
        "attempts_max": int(serial_ref_max_attempts_per_task),
        "attempt_history": [],
        "source": None,
        "used_dataset_solution": False,
        "used_audited_nohelper_bootstrap": False,
        "audited_nohelper_bootstrap_available": bool(str(serial_audited_nohelper_scalar or "").strip()),
        "audited_nohelper_bootstrap_source": str(serial_audited_nohelper_source or ""),
        "ast_bootstrap_available": bool(str(serial_ast_bootstrap_pseudocode or "").strip()),
        "ast_bootstrap_source": str(serial_ast_bootstrap_source or ""),
        "ast_bootstrap_pattern": str(serial_ast_bootstrap_pattern or ""),
        "ast_bootstrap_reason": str(serial_ast_bootstrap_reason or ""),
        "bootstrap_semantics_source": "",
        "bootstrap_semantics_completion_len": 0,
        "eval_intrinsic": str(serial_eval_intrinsic or "scalar"),
        "feedback_style": serial_feedback_style_effective,
        "feedback_style_requested": serial_feedback_style_requested,
        "feedback_style_effective": serial_feedback_style_effective,
        "routing": {
            "problem_type": str(problem_type or ""),
            "problem_subtype": str(problem_subtype or ""),
            "source_name": str(source_name or ""),
            "source_type": str(source_type or ""),
            "source": str(serial_feedback_route_info.get("source") or ""),
            "pattern": str(serial_feedback_route_info.get("pattern") or ""),
            "reason": str(serial_feedback_route_info.get("reason") or ""),
            "has_scalar_reference": False,
        },
        "note": "",
        "mismatch": None,
    }

    def _reroute_serial_feedback_from_reference(serial_source: str, source_label: str = "") -> None:
        """Update routed feedback style after a validated scalar reference exists."""
        nonlocal serial_feedback_style_effective, serial_feedback_route_info
        if str(serial_feedback_style_requested or "").strip().lower() not in {
            "routed",
            "auto",
            "route",
            "routing",
            "routed_by_type",
            "type_routed",
        }:
            return
        if not str(serial_source or "").strip():
            return
        if str(serial_ast_bootstrap_pseudocode or "").strip():
            ast_style = str(serial_ast_bootstrap_style or "").strip()
            if ast_style in {
                "pseudocode",
                "dataflow_pseudocode",
                "unified_dataflow_pseudocode",
                "constrained_pseudocode",
                "code",
                "both",
            }:
                route = {
                    "style": ast_style,
                    "source": str(serial_ast_bootstrap_source or "serial_ast_bootstrap_jsonl"),
                    "pattern": str(serial_ast_bootstrap_pattern or ""),
                    "reason": str(serial_ast_bootstrap_reason or "AST-derived serial bootstrap route."),
                }
                serial_feedback_route_info = route
                serial_feedback_style_effective = ast_style
                try:
                    info["serial_ref"]["feedback_style"] = serial_feedback_style_effective
                    info["serial_ref"]["feedback_style_effective"] = serial_feedback_style_effective
                    info["serial_ref"]["feedback_style_requested"] = serial_feedback_style_requested
                    routing = info["serial_ref"].setdefault("routing", {})
                    routing.update(
                        {
                            "source": str(route.get("source") or ""),
                            "pattern": str(route.get("pattern") or ""),
                            "reason": str(route.get("reason") or ""),
                            "has_scalar_reference": True,
                            "scalar_reference_source": str(source_label or ""),
                        }
                    )
                except Exception:
                    pass
                return
        route = classify_serial_feedback_route(
            serial_feedback_style_requested,
            task_id=task_id,
            problem_type=problem_type,
            problem_subtype=problem_subtype,
            source_name=source_name,
            source_type=source_type,
            entrypoint_simd=serial_entrypoint_simd or func_name or "",
            prompt_prefix=prompt_prefix,
            serial_ref_source=serial_source,
        )
        routed_style = str(route.get("style") or "").strip()
        if routed_style not in {
            "pseudocode",
            "dataflow_pseudocode",
            "unified_dataflow_pseudocode",
            "constrained_pseudocode",
            "code",
            "both",
        }:
            return
        serial_feedback_route_info = route
        serial_feedback_style_effective = routed_style
        try:
            info["serial_ref"]["feedback_style"] = serial_feedback_style_effective
            info["serial_ref"]["feedback_style_effective"] = serial_feedback_style_effective
            info["serial_ref"]["feedback_style_requested"] = serial_feedback_style_requested
            routing = info["serial_ref"].setdefault("routing", {})
            routing.update(
                {
                    "source": str(route.get("source") or ""),
                    "pattern": str(route.get("pattern") or ""),
                    "reason": str(route.get("reason") or ""),
                    "has_scalar_reference": True,
                    "scalar_reference_source": str(source_label or ""),
                }
            )
        except Exception:
            pass

    # How many chars of remote feedback we inject into LLM prompts (after the remote side already
    # truncated logs to `remote_simdbench_tail_chars`). Keeping this bounded avoids blowing up prompts.
    remote_feedback_max_chars = int(min(remote_simdbench_tail_chars, 4000))
    # System prompt for the serial reference model (keep it explicit: scalar only).
    # NOTE: The user prompt already contains detailed rules; this just reinforces "no SIMD".
    serial_system_prompt = (
        "You are generating a SERIAL (scalar-only) C/C++ reference implementation.\n"
        "Do NOT use any SIMD/vector intrinsics or vector types (SVE/NEON/AVX/etc).\n"
        "Return ONLY a single complete C/C++ source file. No markdown. No explanations."
    )
    serial_mismatch_system_prompt = (
        "You are generating a C++17 mismatch harness for differential testing.\n"
        "Return ONLY a single complete C++ source file. No markdown. No explanations."
    )



    if serial_fallback and (serial_model is None) and (not serial_use_solution_scalar):
        info["serial_ref"]["note"] = "serial_fallback enabled but serial_model is None (check --serial_llm_backend / API key / env)."
    completion = completion_in

    # -------------------------
    # Optional SERIAL reference (scalar) generation state (for compile_ok=1 & run_ok=0)
    # -------------------------
    serial_ref_completion: Optional[str] = None
    serial_bootstrap_completion: Optional[str] = None
    serial_bootstrap_completion_source: str = ""
    serial_ref_result: Optional[Dict] = None
    serial_ref_ok: bool = False
    serial_ref_attempted: bool = False
    serial_ref_round: Optional[int] = None

    if str(serial_audited_nohelper_scalar or "").strip():
        serial_bootstrap_completion = build_serial_reference_from_solution(
            str(serial_audited_nohelper_scalar or ""),
            str(serial_audited_nohelper_entrypoint or serial_entrypoint_simd or func_name or ""),
            serial_entrypoint_simd or (func_name or ""),
        )
        if serial_bootstrap_completion:
            serial_bootstrap_completion_source = str(
                serial_audited_nohelper_source or "audited_nohelper_scalar"
            ).strip() or "audited_nohelper_scalar"
            info["serial_ref"].update(
                {
                    "used_audited_nohelper_bootstrap": True,
                    "audited_nohelper_bootstrap_available": True,
                    "audited_nohelper_bootstrap_source": serial_bootstrap_completion_source,
                    "bootstrap_semantics_source": serial_bootstrap_completion_source,
                    "bootstrap_semantics_completion_len": len(serial_bootstrap_completion),
                }
            )


    # -------------------------
    # FIX: define timeouts (the previous script version had undefined vars here)
    # -------------------------
    _run_timeout = int(remote_timeout) if remote_timeout and remote_timeout > 0 else 15
    _compile_timeout = float(remote_simdbench_compile_timeout) if (remote_simdbench_compile_timeout and remote_simdbench_compile_timeout > 0) else float(_run_timeout)
    # SSH wrapper timeout should cover compile+run plus slack
    _ssh_timeout_s = int(max(60.0, _compile_timeout + float(_run_timeout) + 60.0))

    def _maybe_dump(tag: str, content: str, round_no: int) -> None:
        if dump_dir is None:
            return
        dump_dir.mkdir(parents=True, exist_ok=True)
        (dump_dir / f"{tag}_round{round_no}.txt").write_text(content, encoding="utf-8")

    # Backward-compat alias: some call sites use `_dump_text`
    _dump_text = _maybe_dump
    # Derive the per-sample `case_dir` from `dump_dir` (if provided).
    # We use this for saving cached serial reference files in the same directory layout as the main run.
    case_dir: Optional[Path] = None
    if dump_dir is not None:
        try:
            case_dir = dump_dir.parent
        except Exception:
            case_dir = None



    def _dump_json(tag: str, obj: Dict, round_no: int) -> None:
        if dump_dir is None:
            return
        dump_dir.mkdir(parents=True, exist_ok=True)
        (dump_dir / f"{tag}_round{round_no}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    def _derive_remote_case_namespace() -> str:
        seed_parts: List[str] = [task_id, str(sample_idx), str(os.getpid())]
        rank_hint = ""
        if case_dir is not None:
            try:
                seed_parts.append(str(case_dir.resolve()))
            except Exception:
                seed_parts.append(str(case_dir))
            try:
                rank_name = str(case_dir.name or "").strip()
                if rank_name.startswith("rank"):
                    rank_hint = rank_name
            except Exception:
                rank_hint = ""
        elif dump_dir is not None:
            try:
                seed_parts.append(str(dump_dir.resolve()))
            except Exception:
                seed_parts.append(str(dump_dir))
        else:
            seed_parts.append(str(time.time_ns()))

        digest = hashlib.sha1("||".join(seed_parts).encode("utf-8", errors="replace")).hexdigest()[:12]
        return f"{rank_hint}_{digest}" if rank_hint else digest

    remote_case_namespace = _derive_remote_case_namespace()
    remote_case_dir = f"{remote_tmp_root.rstrip('/')}/cases/{remote_case_namespace}/{task_id}/sample{sample_idx}"
    remote_src = f"{remote_case_dir}/case.cpp"
    remote_obj = f"{remote_case_dir}/case.o"
    remote_completion_path = f"{remote_case_dir}/completion.txt"
    remote_sample_file = f"{remote_case_dir}/sample.jsonl"
    info["remote_case_namespace"] = remote_case_namespace
    info["remote_case_dir"] = remote_case_dir

    def _strip_intrinsic_suffix(tid: str) -> str:
        """Map SIMD task_id (e.g., SimdBench_32_SVE) to scalar task_id (SimdBench_32)."""
        if not tid or "_" not in tid:
            return tid
        base, suffix = tid.rsplit("_", 1)
        if suffix.upper() in {"SVE", "NEON", "AVX", "AVX2", "SSE", "SSE2", "SSE4", "RVV"}:
            return base
        return tid

    def _sync_to_remote(
        round_no: int,
        full_source: str,
        completion_text: str,
        *,
        task_id_override: Optional[str] = None,
    ) -> Tuple[bool, str]:
        mk_log = (dump_dir / f"_remote_mkdir_round{round_no}.log") if dump_dir else None
        cp_log = (dump_dir / f"_remote_scp_round{round_no}.log") if dump_dir else None

        ok_mk, msg_mk = remote_mkdir(
            user=remote_user,
            host=remote_host,
            port=remote_port,
            key_path=remote_ssh_key,
            remote_dir=remote_case_dir,
            no_strict_hostkey=remote_no_strict_hostkey,
            log_path=mk_log,
            print_cmd=print_summary,
        )
        if not ok_mk:
            return False, msg_mk

        if dump_dir is None:
            local_src = Path(f".__tmp_{task_id}_sample{sample_idx}_round{round_no}.cpp")
            local_comp = Path(f".__tmp_{task_id}_sample{sample_idx}_round{round_no}.txt")
            local_sample = Path(f".__tmp_{task_id}_sample{sample_idx}_round{round_no}.jsonl")
        else:
            local_src = dump_dir / f"case_round{round_no}.cpp"
            local_comp = dump_dir / f"completion_round{round_no}.txt"
            local_sample = dump_dir / f"sample_round{round_no}.jsonl"

        local_src.write_text(full_source, encoding="utf-8", errors="replace")
        local_comp.write_text(completion_text, encoding="utf-8", errors="replace")
        tid = task_id_override or task_id
        local_sample.write_text(
            json.dumps({"task_id": tid, "completion": completion_text}, ensure_ascii=False) + "\n",
            encoding="utf-8",
            errors="replace",
        )

        # 1) upload sample jsonl (required for simdbench_one/cmd mode)
        ok_cp0, msg_cp0 = remote_scp_file(
            user=remote_user,
            host=remote_host,
            port=remote_port,
            key_path=remote_ssh_key,
            local_path=local_sample,
            remote_path=remote_sample_file,
            no_strict_hostkey=remote_no_strict_hostkey,
            log_path=cp_log,
            print_cmd=print_summary,
        )
        if not ok_cp0:
            return False, msg_mk + "\n" + msg_cp0

        # 2) upload source file (useful for compile_only or debugging)
        ok_cp1, msg_cp1 = remote_scp_file(
            user=remote_user,
            host=remote_host,
            port=remote_port,
            key_path=remote_ssh_key,
            local_path=local_src,
            remote_path=remote_src,
            no_strict_hostkey=remote_no_strict_hostkey,
            log_path=None,
            print_cmd=False,
        )
        if not ok_cp1:
            return False, msg_mk + "\n" + msg_cp0 + "\n" + msg_cp1

        # 3) upload completion snippet (debug)
        ok_cp2, msg_cp2 = remote_scp_file(
            user=remote_user,
            host=remote_host,
            port=remote_port,
            key_path=remote_ssh_key,
            local_path=local_comp,
            remote_path=remote_completion_path,
            no_strict_hostkey=remote_no_strict_hostkey,
            log_path=None,
            print_cmd=False,
        )
        if not ok_cp2:
            return False, msg_mk + "\n" + msg_cp0 + "\n" + msg_cp1 + "\n" + msg_cp2

        if dump_dir is None:
            for p in (local_src, local_comp, local_sample):
                try:
                    p.unlink()
                except Exception:
                    pass

        return True, msg_mk + "\n" + msg_cp0 + "\n" + msg_cp1 + "\n" + msg_cp2

    def _remote_perf_precheck_enabled() -> bool:
        return bool(
            remote_perf_precheck
            and remote_eval_mode == "simdbench_one"
            and build_nonbenchmark_perf_problem_row is not None
        )

    def _render_remote_perf_precheck_feedback(precheck: Optional[Dict[str, Any]]) -> str:
        if not isinstance(precheck, dict):
            return ""
        parts: List[str] = ["[PERF_PRECHECK_SUMMARY]"]
        for key in [
            "correctness_ok",
            "perf_ok",
            "candidate_failed",
            "availability_error",
            "reason",
            "detail_reason",
        ]:
            value = precheck.get(key)
            if value is not None:
                parts.append(f"- {key}: {value}")

        raw_excerpt = str(precheck.get("raw_result_excerpt") or "").strip()
        if raw_excerpt:
            parts.append("[PERF_RESULT_EXCERPT]")
            parts.append(_tail_chars(raw_excerpt, 2000))

        stdout_tail = str(precheck.get("stdout_tail") or "").strip()
        if stdout_tail:
            parts.append("[PERF_STDOUT_TAIL]")
            parts.append(_extract_interesting_log_lines(stdout_tail, max_lines=40) or _tail_chars(stdout_tail, 2000))

        stderr_tail = str(precheck.get("stderr_tail") or "").strip()
        if stderr_tail:
            parts.append("[PERF_STDERR_TAIL]")
            parts.append(_extract_interesting_log_lines(stderr_tail, max_lines=40) or _tail_chars(stderr_tail, 2000))

        return "\n".join(parts).strip()

    def _run_remote_perf_precheck(round_no: int, completion_text: str) -> Dict[str, Any]:
        base_summary: Dict[str, Any] = {
            "correctness_ok": False,
            "perf_ok": False,
            "candidate_failed": False,
            "availability_error": False,
            "reason": "perf_precheck_disabled",
            "detail_reason": None,
            "speedup": None,
            "speedup_median": None,
            "speedup_mean": None,
            "speedup_samples": 0,
            "raw_result_excerpt": "",
            "stdout_tail": "",
            "stderr_tail": "",
        }

        if not _remote_perf_precheck_enabled():
            if build_nonbenchmark_perf_problem_row is None:
                base_summary["reason"] = "perf_precheck_builder_unavailable"
                base_summary["availability_error"] = True
            return base_summary

        if not str(nl_description_ds_r1 or "").strip():
            base_summary["reason"] = "perf_precheck_missing_nl_description"
            base_summary["availability_error"] = True
            return base_summary
        if not str(serial_c_code or "").strip():
            base_summary["reason"] = "perf_precheck_missing_serial_c_code"
            base_summary["availability_error"] = True
            return base_summary
        if not str(test_harness_code or "").strip():
            base_summary["reason"] = "perf_precheck_missing_test_harness_code"
            base_summary["availability_error"] = True
            return base_summary

        source_row = {
            "task_id": task_id,
            "sample_id": f"{task_id}__sample{sample_idx}",
            "nl_description_ds_r1": nl_description_ds_r1,
            "serial_c_code": serial_c_code,
            "test_harness_code": test_harness_code,
            "source_type": source_type,
        }
        try:
            perf_problem_row = build_nonbenchmark_perf_problem_row(source_row, intrinsic=intrinsic)
        except Exception as exc:
            base_summary["reason"] = "perf_precheck_problem_build_failed"
            base_summary["availability_error"] = True
            base_summary["stderr_tail"] = _tail_chars(f"{type(exc).__name__}: {exc}", 2000)
            return base_summary

        if dump_dir is None:
            local_perf_problem = Path(f".__tmp_{task_id}_sample{sample_idx}_perf_problem_round{round_no}.jsonl")
            local_perf_sample = Path(f".__tmp_{task_id}_sample{sample_idx}_perf_sample_round{round_no}.jsonl")
        else:
            local_perf_problem = dump_dir / f"perf_precheck_problem_round{round_no}.jsonl"
            local_perf_sample = dump_dir / f"perf_precheck_sample_round{round_no}.jsonl"

        local_perf_problem.write_text(
            json.dumps(perf_problem_row, ensure_ascii=False) + "\n",
            encoding="utf-8",
            errors="replace",
        )
        local_perf_sample.write_text(
            json.dumps({"task_id": task_id, "completion": completion_text}, ensure_ascii=False) + "\n",
            encoding="utf-8",
            errors="replace",
        )

        remote_perf_problem = f"{remote_case_dir}/perf_precheck_round{round_no}.problem.jsonl"
        remote_perf_sample = f"{remote_case_dir}/perf_precheck_round{round_no}.sample.jsonl"
        remote_perf_output_dir = f"{remote_case_dir}/perf_precheck_round{round_no}_out"

        try:
            mk_log = (dump_dir / f"_remote_perf_precheck_mkdir_round{round_no}.log") if dump_dir else None
            ok_mk, msg_mk = remote_mkdir(
                user=remote_user,
                host=remote_host,
                port=remote_port,
                key_path=remote_ssh_key,
                remote_dir=remote_perf_output_dir,
                no_strict_hostkey=remote_no_strict_hostkey,
                log_path=mk_log,
                print_cmd=print_summary,
            )
            if not ok_mk:
                base_summary["reason"] = "perf_precheck_remote_mkdir_failed"
                base_summary["availability_error"] = True
                base_summary["stderr_tail"] = _tail_chars(msg_mk, 2000)
                return base_summary

            cp_log = (dump_dir / f"_remote_perf_precheck_scp_round{round_no}.log") if dump_dir else None
            ok_cp0, msg_cp0 = remote_scp_file(
                user=remote_user,
                host=remote_host,
                port=remote_port,
                key_path=remote_ssh_key,
                local_path=local_perf_problem,
                remote_path=remote_perf_problem,
                no_strict_hostkey=remote_no_strict_hostkey,
                log_path=cp_log,
                print_cmd=print_summary,
            )
            if not ok_cp0:
                base_summary["reason"] = "perf_precheck_problem_scp_failed"
                base_summary["availability_error"] = True
                base_summary["stderr_tail"] = _tail_chars(msg_cp0, 2000)
                return base_summary

            ok_cp1, msg_cp1 = remote_scp_file(
                user=remote_user,
                host=remote_host,
                port=remote_port,
                key_path=remote_ssh_key,
                local_path=local_perf_sample,
                remote_path=remote_perf_sample,
                no_strict_hostkey=remote_no_strict_hostkey,
                log_path=None,
                print_cmd=False,
            )
            if not ok_cp1:
                base_summary["reason"] = "perf_precheck_sample_scp_failed"
                base_summary["availability_error"] = True
                base_summary["stderr_tail"] = _tail_chars(msg_cp1, 2000)
                return base_summary

            repo_root = str(remote_perf_precheck_repo_root or "").strip()
            if not repo_root:
                base_summary["reason"] = "perf_precheck_missing_repo_root"
                base_summary["availability_error"] = True
                return base_summary

            remote_python = str(remote_perf_precheck_python or "python3").strip() or "python3"
            optimization = str(remote_perf_precheck_optimization or "-O1").strip() or "-O1"
            timeout_value = float(remote_perf_precheck_timeout or 30.0)

            remote_python_code = (
                "import json, traceback\n"
                "from simdbench.execution import exec_cpp, clean_generated_code, get_header, swallow_io, time_limit, TimeoutException\n"
                f"sample_file = {remote_perf_sample!r}\n"
                f"problem_file = {remote_perf_problem!r}\n"
                f"timeout_value = {timeout_value!r}\n"
                f"optimization = {optimization!r}\n"
                "summary = {\n"
                "    'correctness_ok': False,\n"
                "    'perf_ok': False,\n"
                "    'candidate_failed': False,\n"
                "    'availability_error': False,\n"
                "    'reason': 'perf_correctness_unchecked',\n"
                "    'detail_reason': None,\n"
                "    'speedup': None,\n"
                "    'speedup_median': None,\n"
                "    'speedup_mean': None,\n"
                "    'speedup_samples': 0,\n"
                "    'raw_result_excerpt': '',\n"
                "}\n"
                "try:\n"
                "    with open(problem_file, 'r', encoding='utf-8') as pf:\n"
                "        problem = json.loads(next(line for line in pf if line.strip()))\n"
                "    with open(sample_file, 'r', encoding='utf-8') as sf:\n"
                "        sample = json.loads(next(line for line in sf if line.strip()))\n"
                "    intrinsic = problem['intrinsic']\n"
                "    compile_modes = problem.get('compile_modes')\n"
                "    completion = sample['completion']\n"
                "    if completion.find(problem['entrypoint_simd']) == -1 and intrinsic == 'scalar':\n"
                "        completion_eval = completion.replace(problem['entrypoint_scalar'], problem['entrypoint_simd'])\n"
                "    else:\n"
                "        completion_eval = completion\n"
                "    correctness_cpp_no_simd_header = (\n"
                "        get_header() + '\\n' +\n"
                "        problem['solution_scalar'] + '\\n' +\n"
                "        clean_generated_code(completion_eval, problem['entrypoint_simd']) + '\\n' +\n"
                "        problem['test_correctness']\n"
                "    )\n"
                "    for header in get_header(intrinsic).split('\\n'):\n"
                "        if header.startswith('#include'):\n"
                "            correctness_cpp_no_simd_header = correctness_cpp_no_simd_header.replace(header, '')\n"
                "    correctness_cpp = (\n"
                "        get_header() + '\\n' +\n"
                "        get_header(intrinsic) + '\\n' +\n"
                "        problem['solution_scalar'] + '\\n' +\n"
                "        clean_generated_code(completion_eval, problem['entrypoint_simd']) + '\\n' +\n"
                "        problem['test_correctness']\n"
                "    )\n"
                "    try:\n"
                "        with swallow_io():\n"
                "            with time_limit(timeout_value):\n"
                "                no_intrinsic_res, no_intrinsic_stdout = exec_cpp(\n"
                "                    correctness_cpp_no_simd_header,\n"
                "                    'scalar',\n"
                "                    timeout_value,\n"
                "                    compile_modes=compile_modes,\n"
                "                )\n"
                "        if no_intrinsic_res == 1:\n"
                "            summary['candidate_failed'] = True\n"
                "            summary['reason'] = 'no_intrinsic_in_code_under_perf_check'\n"
                "            summary['raw_result_excerpt'] = str(no_intrinsic_stdout)[:2000]\n"
                "            print(json.dumps(summary, ensure_ascii=False))\n"
                "            raise SystemExit(0)\n"
                "    except TimeoutException:\n"
                "        pass\n"
                "    except Exception:\n"
                "        pass\n"
                "    try:\n"
                "        with swallow_io():\n"
                "            with time_limit(timeout_value):\n"
                "                correctness_res, correctness_stdout = exec_cpp(\n"
                "                    correctness_cpp,\n"
                "                    intrinsic,\n"
                "                    timeout_value,\n"
                "                    optimization=optimization,\n"
                "                    compile_modes=compile_modes,\n"
                "                )\n"
                "        if correctness_res == 0:\n"
                "            summary['candidate_failed'] = True\n"
                "            summary['reason'] = 'logical_bug'\n"
                "            summary['detail_reason'] = 'correctness_regressed_during_perf'\n"
                "            summary['raw_result_excerpt'] = str(correctness_stdout)[:2000]\n"
                "        else:\n"
                "            data = json.loads(correctness_stdout)\n"
                "            if int(data.get('correctness', 0)) == 1:\n"
                "                summary['correctness_ok'] = True\n"
                "                summary['perf_ok'] = True\n"
                "                summary['candidate_failed'] = False\n"
                "                summary['reason'] = 'perf_correctness_ok'\n"
                "                summary['raw_result_excerpt'] = str(correctness_stdout)[:2000]\n"
                "            else:\n"
                "                summary['candidate_failed'] = True\n"
                "                summary['reason'] = 'logical_bug'\n"
                "                summary['detail_reason'] = 'correctness_regressed_during_perf'\n"
                "                summary['raw_result_excerpt'] = str(correctness_stdout)[:2000]\n"
                "    except TimeoutException:\n"
                "        summary['candidate_failed'] = True\n"
                "        summary['reason'] = 'perf_correctness_timeout'\n"
                "    except BaseException as exc:\n"
                "        summary['candidate_failed'] = True\n"
                "        summary['reason'] = 'logical_bug'\n"
                "        summary['detail_reason'] = 'correctness_regressed_during_perf'\n"
                "        summary['raw_result_excerpt'] = (type(exc).__name__ + ': ' + str(exc) + '\\n' + traceback.format_exc())[:2000]\n"
                "except Exception as exc:\n"
                "    summary['availability_error'] = True\n"
                "    summary['candidate_failed'] = False\n"
                "    summary['reason'] = 'perf_precheck_exception'\n"
                "    summary['raw_result_excerpt'] = (type(exc).__name__ + ': ' + str(exc) + '\\n' + traceback.format_exc())[:2000]\n"
                "print(json.dumps(summary, ensure_ascii=False))\n"
            )

            remote_shell = (
                "set -euo pipefail\n"
                f"mkdir -p {shlex.quote(remote_perf_output_dir)}\n"
                f"cd {shlex.quote(repo_root)}\n"
                f"PYTHONPATH={shlex.quote(repo_root)} {shlex.quote(remote_python)} - <<'PY'\n"
                f"{remote_python_code}"
                "PY\n"
            )
            remote_cmd = f"bash -lc {shlex.quote(remote_shell)}"
            if remote_flock_path:
                remote_cmd = f"flock -x {_quote_remote_token(remote_flock_path)} -c {shlex.quote(remote_cmd)}"

            ssh_log = (dump_dir / f"_remote_perf_precheck_round{round_no}.log") if dump_dir else None
            remote = f"{remote_user}@{remote_host}"
            cmd = ["ssh", "-i", remote_ssh_key, "-p", str(remote_port)] + _ssh_scp_common_opts(remote_no_strict_hostkey) + [remote, remote_cmd]
            timeout_s = max(60, int(timeout_value) + 90)
            rc, out, err, cmd_str = _run_cmd_capture(
                cmd,
                timeout_s=timeout_s,
                log_path=ssh_log,
                print_cmd=print_summary,
                tail_chars=8000,
            )
            if rc != 0:
                base_summary["reason"] = "perf_precheck_remote_cmd_failed"
                base_summary["availability_error"] = True
                base_summary["stdout_tail"] = _tail_chars(out, 4000)
                base_summary["stderr_tail"] = _tail_chars(err, 4000)
                base_summary["_ssh_cmd"] = cmd_str
                base_summary["_remote_cmd"] = remote_cmd
                return base_summary

            js = _parse_last_json_line(out)
            if js is None:
                base_summary["reason"] = "perf_precheck_json_parse_failed"
                base_summary["availability_error"] = True
                base_summary["stdout_tail"] = _tail_chars(out, 4000)
                base_summary["stderr_tail"] = _tail_chars(err, 4000)
                base_summary["_ssh_cmd"] = cmd_str
                base_summary["_remote_cmd"] = remote_cmd
                return base_summary

            if isinstance(js, dict):
                js["stdout_tail"] = _tail_chars(out, 4000)
                js["stderr_tail"] = _tail_chars(err, 4000)
                js["_ssh_cmd"] = cmd_str
                js["_remote_cmd"] = remote_cmd
                return js

            base_summary["reason"] = "perf_precheck_invalid_json_object"
            base_summary["availability_error"] = True
            base_summary["stdout_tail"] = _tail_chars(out, 4000)
            base_summary["stderr_tail"] = _tail_chars(err, 4000)
            return base_summary
        finally:
            if dump_dir is None:
                for path in (local_perf_problem, local_perf_sample):
                    try:
                        path.unlink()
                    except Exception:
                        pass

    def _apply_remote_perf_precheck(round_no: int, completion_text: str, res: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(res, dict):
            return res
        if not _remote_perf_precheck_enabled():
            return res

        try:
            compile_ok = int(res.get("compile_ok", 0) or 0)
            run_ok = int(res.get("run_ok", 0) or 0)
        except Exception:
            compile_ok, run_ok = 0, 0
        if compile_ok != 1 or run_ok != 1:
            return res

        precheck = _run_remote_perf_precheck(round_no, completion_text)
        info["remote_perf_precheck"]["history"].append({"round": round_no, "result": precheck})
        info["remote_perf_precheck"]["final"] = precheck

        merged = dict(res)
        merged["perf_precheck"] = precheck
        merged["perf_precheck_ok"] = 1 if bool(precheck.get("perf_ok")) else 0
        merged["perf_precheck_reason"] = precheck.get("detail_reason") or precheck.get("reason")

        if bool(precheck.get("availability_error")):
            info["remote_perf_precheck"]["availability_errors"].append(str(precheck.get("reason") or "availability_error"))
            return merged

        if not bool(precheck.get("candidate_failed")):
            return merged

        merged["run_ok"] = 0
        merged["passed"] = 0
        merged["reason"] = str(precheck.get("reason") or "perf_eval_failed")
        feedback_text = _render_remote_perf_precheck_feedback(precheck)
        existing_run_log = str(merged.get("run_log_tail") or "").strip()
        if feedback_text:
            merged["run_log_tail"] = _tail_chars(
                "\n\n".join(x for x in [existing_run_log, feedback_text] if x),
                8000,
            )
        simdbench_raw = str(merged.get("simdbench_raw_result") or "").strip()
        precheck_reason = str(precheck.get("detail_reason") or precheck.get("reason") or "perf_eval_failed").strip()
        merged["simdbench_raw_result"] = (
            f"{simdbench_raw}\nperf_precheck: {precheck_reason}".strip()
            if simdbench_raw
            else f"perf_precheck: {precheck_reason}"
        )
        return merged

    def _static_verifier_fallback_report(
        *,
        full_source: str,
        error_msg: str,
    ) -> Dict[str, Any]:
        return {
            "schema_version": "sve_static_verifier_pass1_v22",
            "verifier_name": "sve_static_verifier_pass1_v22",
            "task_id": task_id,
            "ok": False,
            "compile_risk_tags": [],
            "semantic_risk_tags": [],
            "repair_action": "escalate_remote",
            "repair_prompt_payload": {
                "controller_version": "repair_controller_pass1_v22",
                "controller_action": "remote_escalation",
                "primary_focus_area": None,
                "ordered_tags": [],
                "tag_groups": {"compile": {}, "semantic": {}},
                "summary_lines": [],
                "must_fix_tags": [],
                "focus_areas": [],
                "repair_steps": [
                    "treat local verifier as unavailable and rely on remote feedback only"
                ],
                "notes": [
                    "static verifier failed open; continue with remote evaluation and do not trust local preflight coverage"
                ],
            },
            "stats": {
                "compile_risk_count": 0,
                "semantic_risk_count": 0,
                "semantic_access_risk_count": 0,
                "semantic_predicate_risk_count": 0,
                "semantic_reduction_risk_count": 0,
            },
            "input_summary": {
                "code_len": len(full_source),
                "has_semantic_plan": isinstance(semantic_plan, dict),
                "plan_access_kind": str((((semantic_plan or {}).get("access_map") or {}) if isinstance((semantic_plan or {}).get("access_map"), dict) else {}).get("access_kind") or "") or None,
                "plan_predicate_kind": str((((semantic_plan or {}).get("predicate_rule") or {}) if isinstance((semantic_plan or {}).get("predicate_rule"), dict) else {}).get("kind") or "") or None,
                "plan_reduction_type": str((((semantic_plan or {}).get("reduction_rule") or {}) if isinstance((semantic_plan or {}).get("reduction_rule"), dict) else {}).get("type") or "") or None,
                "plan_uncertain_fields": sorted(
                    [
                        str(x).strip()
                        for x in ((semantic_plan or {}).get("uncertain_fields") or [])
                        if str(x).strip()
                    ]
                ) if isinstance(semantic_plan, dict) else [],
            },
            "internal_errors": [error_msg],
        }

    def _build_signature_drift_report(
        *,
        full_source: str,
        expected_signature_line: str,
        actual_signature_line: str,
        signature_check: Dict[str, Any],
    ) -> Dict[str, Any]:
        mismatches = [
            str(x).strip()
            for x in (signature_check.get("mismatches") or [])
            if str(x).strip()
        ]
        evidence_parts: List[str] = []
        if expected_signature_line:
            evidence_parts.append(f"expected={expected_signature_line}")
        if actual_signature_line:
            evidence_parts.append(f"actual={actual_signature_line}")
        if mismatches:
            evidence_parts.append("mismatches=" + ",".join(mismatches))
        evidence_text = "; ".join(evidence_parts) or "target function signature differs from prompt prefix"
        compile_tag = {
            "tag": "target_signature_drift",
            "severity": "error",
            "summary": "the generated target function signature differs from the exact prompt-prefix signature and must be restored before other repair",
            "line": None,
            "evidence": evidence_text,
        }
        repair_steps = [
            "restore the target function header to the exact signature from the prompt prefix before addressing downstream compile noise",
            "keep the body as intact as possible; if flat indexing is needed, do it inside the function body instead of changing the header",
        ]
        summary_lines = [
            "controller_action=compile_signature_shape_fix",
            f"expected_signature={expected_signature_line}" if expected_signature_line else "expected_signature=<missing>",
            f"actual_signature={actual_signature_line}" if actual_signature_line else "actual_signature=<missing>",
        ]
        if mismatches:
            summary_lines.append("signature_mismatches=" + ",".join(mismatches))
        report: Dict[str, Any] = {
            "schema_version": "sve_static_verifier_pass1_v22",
            "verifier_name": "sve_static_verifier_pass1_v22",
            "task_id": task_id,
            "ok": False,
            "compile_risk_tags": [compile_tag],
            "semantic_risk_tags": [],
            "repair_action": "compile_signature_shape_fix",
            "expected_signature_line": expected_signature_line,
            "actual_signature_line": actual_signature_line,
            "signature_check": deepcopy(signature_check),
            "repair_prompt_payload": {
                "controller_version": "repair_controller_pass1_v22",
                "controller_action": "compile_signature_shape_fix",
                "primary_focus_area": "target_function_signature",
                "ordered_tags": ["target_signature_drift"],
                "tag_groups": {"compile": {"signature_closure": 1}, "semantic": {}},
                "summary_lines": summary_lines,
                "must_fix_tags": ["target_signature_drift"],
                "focus_areas": ["target_function_signature"],
                "repair_steps": repair_steps,
                "notes": [
                    "use the prompt-prefix signature as ground truth instead of preserving the current completion header"
                ],
                "expected_signature_line": expected_signature_line,
                "actual_signature_line": actual_signature_line,
                "signature_check": deepcopy(signature_check),
            },
            "stats": {
                "compile_risk_count": 1,
                "semantic_risk_count": 0,
                "semantic_access_risk_count": 0,
                "semantic_predicate_risk_count": 0,
                "semantic_reduction_risk_count": 0,
            },
            "input_summary": {
                "code_len": len(full_source),
                "has_semantic_plan": isinstance(semantic_plan, dict),
                "expected_signature_line": expected_signature_line,
                "actual_signature_line": actual_signature_line,
            },
            "internal_errors": [],
        }
        return report

    def _run_signature_closure_check(
        *,
        round_no: int,
        full_source: str,
    ) -> Optional[Dict[str, Any]]:
        expected_signature_line = target_signature_line_from_decl(CURRENT_TARGET_FUNC_DECL)
        if not expected_signature_line:
            return None
        if not parse_shared_signature:
            return None
        try:
            expected_signature = parse_shared_signature(expected_signature_line)
            if not isinstance(expected_signature, dict):
                return None
            signature_check = build_target_signature_closure_check(
                expected_signature_line,
                expected_signature,
                full_source,
            )
        except Exception as exc:
            report = _static_verifier_fallback_report(
                full_source=full_source,
                error_msg=f"signature_check_error:{type(exc).__name__}:{exc}",
            )
            info["static_verifier"]["history"].append({"round": round_no, "report": report})
            if dump_dir is not None:
                _dump_json("static_verifier", report, round_no)
            return report

        if bool(signature_check.get("match", True)):
            return None

        actual_signature_line = str(signature_check.get("actual_signature_line") or "").strip()
        report = _build_signature_drift_report(
            full_source=full_source,
            expected_signature_line=expected_signature_line,
            actual_signature_line=actual_signature_line,
            signature_check=signature_check if isinstance(signature_check, dict) else {},
        )
        info["static_verifier"]["history"].append({"round": round_no, "report": report})
        if dump_dir is not None:
            _dump_json("static_verifier", report, round_no)
        return report

    def _run_static_verifier(round_no: int, completion_text: str) -> Optional[Dict[str, Any]]:
        full_source = completion_text if COMPLETION_MODE == COMPLETION_MODE_FULL else (prompt_prefix + completion_text)
        signature_report = _run_signature_closure_check(round_no=round_no, full_source=full_source)
        if signature_report is not None:
            return signature_report

        if not (static_verifier and static_verifier_script):
            return None

        script_path = Path(static_verifier_script).expanduser()
        if not script_path.is_file():
            report = _static_verifier_fallback_report(
                full_source=full_source,
                error_msg=f"static_verifier_script_missing:{script_path}",
            )
            info["static_verifier"]["history"].append({"round": round_no, "report": report})
            return report

        payload = {
            "task_id": task_id,
            "sample_idx": sample_idx,
            "round_no": round_no,
            "intrinsic": intrinsic,
            "spec_text": spec_text,
            "prompt_prefix": prompt_prefix,
            "semantic_plan": semantic_plan if isinstance(semantic_plan, dict) else None,
            "generated_code": full_source,
        }
        verifier_log = (dump_dir / f"_static_verifier_round{round_no}.log") if dump_dir else None
        cmd = [sys.executable, str(script_path)]
        rc, out, err, _cmd_str = _run_cmd_capture(
            cmd,
            timeout_s=int(static_verifier_timeout or 0),
            input_text=json.dumps(payload, ensure_ascii=False),
            log_path=verifier_log,
            print_cmd=False,
        )
        if rc != 0:
            report = _static_verifier_fallback_report(
                full_source=full_source,
                error_msg=f"static_verifier_rc:{rc}:{_tail_chars(err or out, 2000)}",
            )
            info["static_verifier"]["history"].append({"round": round_no, "report": report})
            if dump_dir is not None:
                _dump_json("static_verifier", report, round_no)
            return report

        try:
            report = json.loads(out)
            if not isinstance(report, dict):
                raise ValueError("static verifier output is not a JSON object")
        except Exception as exc:
            report = _static_verifier_fallback_report(
                full_source=full_source,
                error_msg=f"static_verifier_json_error:{type(exc).__name__}:{exc}",
            )

        info["static_verifier"]["history"].append({"round": round_no, "report": report})
        if dump_dir is not None:
            _dump_json("static_verifier", report, round_no)
        return report

    def _remote_cpp17_compile_gate_fallback_report(
        *,
        full_source: str,
        error_msg: str,
        reason: str,
    ) -> Dict[str, Any]:
        compiler_name = str(remote_cpp17_compile_gate_compiler or remote_compiler or "").strip()
        cflags = str(remote_cpp17_compile_gate_cflags or "").strip()
        return {
            "schema_version": "remote_cpp17_compile_gate_pass1_v22",
            "gate_name": "remote_cpp17_compile_gate_pass1_v22",
            "task_id": task_id,
            "ok": False,
            "compile_ok": False,
            "blocked": True,
            "reason": str(reason or "internal_error"),
            "compile_risk_tags": [],
            "semantic_risk_tags": [],
            "repair_action": "escalate_remote",
            "repair_prompt_payload": {
                "controller_version": "repair_controller_pass1_v22",
                "controller_action": "remote_escalation",
                "primary_focus_area": None,
                "ordered_tags": [],
                "tag_groups": {"compile": {}, "semantic": {}},
                "summary_lines": [],
                "must_fix_tags": [],
                "focus_areas": [],
                "repair_steps": [],
                "notes": [
                    "remote_cpp17_compile_gate unavailable; rely on mode-specific fail-open/fail-closed behavior"
                ],
            },
            "diagnostics": {},
            "symbol_closure_targets": {},
            "compiler": {
                "path": compiler_name,
                "host": str(remote_host or ""),
                "cflags": cflags,
            },
            "stderr_tail": "",
            "stdout_tail": "",
            "cmd": "",
            "input_summary": {"code_len": len(full_source)},
            "internal_errors": [error_msg],
        }

    def _classify_remote_cpp17_compile_gate_failure(res: Optional[Dict[str, Any]]) -> str:
        res = res if isinstance(res, dict) else {}
        compile_log = str(res.get("compile_log_tail") or "")
        low = compile_log.lower()
        compiler_name = str(res.get("compiler") or remote_cpp17_compile_gate_compiler or remote_compiler or "").strip().lower()
        if not low.strip():
            return "internal_error"
        if (
            "could not resolve hostname" in low
            or "connection timed out" in low
            or "connection refused" in low
            or "permission denied" in low
            or "no route to host" in low
            or "connection closed" in low
            or "kex_exchange_identification" in low
            or "ssh:" in low
        ):
            return "ssh_failure"
        if (
            "command not found" in low
            or (compiler_name and f"{compiler_name}: not found" in low)
            or (compiler_name and compiler_name in low and "no such file or directory" in low)
        ):
            return "compiler_missing"
        return "compile_error"

    def _run_remote_cpp17_compile_gate(round_no: int, completion_text: str) -> Optional[Dict[str, Any]]:
        gate_enabled = bool(remote_cpp17_compile_gate and str(remote_cpp17_compile_gate_mode or "off") != "off")
        if not gate_enabled:
            return None

        full_source = completion_text if COMPLETION_MODE == COMPLETION_MODE_FULL else (prompt_prefix + completion_text)
        ok_sync, sync_msg = _sync_to_remote(round_no, full_source, completion_text)
        if not ok_sync:
            report = _remote_cpp17_compile_gate_fallback_report(
                full_source=full_source,
                error_msg=f"remote_cpp17_compile_gate_sync_fail:{_tail_chars(sync_msg, 2000)}",
                reason="sync_fail",
            )
            info["remote_cpp17_compile_gate"]["availability_errors"].append(report["internal_errors"][0])
            if dump_dir is not None:
                _dump_json("remote_cpp17_compile_gate", report, round_no)
            return report

        gate_log = (dump_dir / f"_remote_cpp17_compile_gate_round{round_no}.log") if dump_dir else None
        compiler_name = str(remote_cpp17_compile_gate_compiler or remote_compiler or "clang++").strip() or "clang++"
        cflags = str(remote_cpp17_compile_gate_cflags or "").strip()
        if not cflags:
            cflags = "-std=c++17 -fsyntax-only -march=armv8.2-a+sve -ferror-limit=20"
        res = remote_eval_compile_only(
            user=remote_user,
            host=remote_host,
            port=remote_port,
            key_path=remote_ssh_key,
            remote_src_path=remote_src,
            remote_obj_path=remote_obj,
            compiler=compiler_name,
            cflags=cflags,
            timeout_s=int(max(1, remote_cpp17_compile_gate_timeout or 20)),
            flock_path=remote_flock_path,
            no_strict_hostkey=remote_no_strict_hostkey,
            log_path=gate_log,
            print_cmd=print_summary,
        )

        promoted_report = _promote_remote_compile_diagnostics_report(None, res)
        diagnostics = extract_remote_compile_diagnostics(str(res.get("compile_log_tail", "") or ""))
        symbol_targets = {}
        if isinstance(promoted_report, dict):
            if isinstance(promoted_report.get("symbol_closure_targets"), dict):
                symbol_targets = deepcopy(promoted_report.get("symbol_closure_targets") or {})
            else:
                payload = promoted_report.get("repair_prompt_payload")
                if isinstance(payload, dict) and isinstance(payload.get("symbol_closure_targets"), dict):
                    symbol_targets = deepcopy(payload.get("symbol_closure_targets") or {})

        compile_ok = bool(int(res.get("compile_ok", 0) or 0) == 1)
        reason = "ok" if compile_ok else _classify_remote_cpp17_compile_gate_failure(res)
        report = {
            "schema_version": "remote_cpp17_compile_gate_pass1_v22",
            "gate_name": "remote_cpp17_compile_gate_pass1_v22",
            "task_id": task_id,
            "ok": bool(compile_ok),
            "compile_ok": bool(compile_ok),
            "blocked": not bool(compile_ok),
            "reason": reason,
            "compile_risk_tags": list((promoted_report or {}).get("compile_risk_tags") or []),
            "semantic_risk_tags": list((promoted_report or {}).get("semantic_risk_tags") or []),
            "repair_action": (promoted_report or {}).get("repair_action") or "escalate_remote",
            "repair_prompt_payload": deepcopy((promoted_report or {}).get("repair_prompt_payload") or {}),
            "diagnostics": diagnostics,
            "symbol_closure_targets": symbol_targets,
            "compiler": {
                "path": compiler_name,
                "host": str(remote_host or ""),
                "cflags": cflags,
            },
            "stderr_tail": _tail_chars(str(res.get("compile_log_tail") or ""), 4000),
            "stdout_tail": "",
            "cmd": str(res.get("_ssh_cmd") or ""),
            "input_summary": {"code_len": len(full_source)},
            "internal_errors": [],
            "remote_compile_result": res,
        }
        if isinstance(report["repair_prompt_payload"], dict) and symbol_targets:
            report["repair_prompt_payload"]["symbol_closure_targets"] = deepcopy(symbol_targets)
        if reason in {"sync_fail", "ssh_failure", "compiler_missing", "internal_error"}:
            info["remote_cpp17_compile_gate"]["availability_errors"].append(reason)
        if dump_dir is not None:
            _dump_json("remote_cpp17_compile_gate", report, round_no)
        return report

    def _merge_remote_cpp17_compile_gate_report(
        verifier_report: Optional[Dict[str, Any]],
        gate_report: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not isinstance(gate_report, dict):
            return deepcopy(verifier_report) if isinstance(verifier_report, dict) else {}

        if isinstance(verifier_report, dict):
            merged = deepcopy(verifier_report)
        else:
            merged = {
                "schema_version": "remote_cpp17_compile_gate_merged_v22",
                "verifier_name": "remote_cpp17_compile_gate_merged_v22",
                "task_id": task_id,
                "ok": True,
                "compile_risk_tags": [],
                "semantic_risk_tags": [],
                "repair_action": "none",
                "repair_prompt_payload": {
                    "controller_version": "repair_controller_pass1_v22",
                    "controller_action": "no_local_repair_needed",
                    "primary_focus_area": None,
                    "ordered_tags": [],
                    "tag_groups": {"compile": {}, "semantic": {}},
                    "summary_lines": [],
                    "must_fix_tags": [],
                    "focus_areas": [],
                    "repair_steps": [],
                    "notes": [],
                },
            }

        merged["ok"] = bool(merged.get("ok", True)) and bool(gate_report.get("compile_ok", False))
        compile_tags = list(merged.get("compile_risk_tags") or [])
        seen_compile = {
            (
                str(item.get("tag") or "").strip(),
                str(item.get("evidence") or "").strip(),
            )
            for item in compile_tags
            if isinstance(item, dict)
        }
        for item in gate_report.get("compile_risk_tags") or []:
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("tag") or "").strip(),
                str(item.get("evidence") or "").strip(),
            )
            if not key[0] or key in seen_compile:
                continue
            seen_compile.add(key)
            compile_tags.append(dict(item))
        merged["compile_risk_tags"] = compile_tags
        merged["repair_action"] = gate_report.get("repair_action") or merged.get("repair_action")
        gate_payload = gate_report.get("repair_prompt_payload")
        if isinstance(gate_payload, dict):
            merged["repair_prompt_payload"] = deepcopy(gate_payload)

        gate_diagnostics = gate_report.get("diagnostics")
        if isinstance(gate_diagnostics, dict):
            merged["remote_cpp17_compile_gate_diagnostics"] = deepcopy(gate_diagnostics)
        merged["remote_cpp17_compile_gate_report"] = {
            "reason": gate_report.get("reason"),
            "compile_ok": gate_report.get("compile_ok"),
            "blocked": gate_report.get("blocked"),
            "compiler": gate_report.get("compiler"),
        }

        symbol_targets = gate_report.get("symbol_closure_targets")
        if isinstance(symbol_targets, dict) and symbol_targets:
            merged["symbol_closure_targets"] = deepcopy(symbol_targets)
            payload = merged.get("repair_prompt_payload")
            if isinstance(payload, dict):
                payload["symbol_closure_targets"] = deepcopy(symbol_targets)
                merged["repair_prompt_payload"] = payload
        return merged

    def _remote_cpp17_compile_gate_should_fail_open(gate_report: Optional[Dict[str, Any]]) -> bool:
        if str(remote_cpp17_compile_gate_mode or "off") != "best_effort":
            return False
        if not isinstance(gate_report, dict):
            return False
        return str(gate_report.get("reason") or "").strip() in {
            "sync_fail",
            "ssh_failure",
            "compiler_missing",
            "internal_error",
        }

    def _apply_remote_cpp17_compile_gate(
        *,
        round_no: int,
        completion_text: str,
        verifier_report: Optional[Dict[str, Any]],
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
        gate_enabled = bool(remote_cpp17_compile_gate and str(remote_cpp17_compile_gate_mode or "off") != "off")
        if not gate_enabled:
            return completion_text, verifier_report, None, False

        current_completion = completion_text
        current_verifier = verifier_report
        max_iters = int(max(0, remote_cpp17_compile_gate_max_repair_iters or 0))
        last_gate_report: Optional[Dict[str, Any]] = None

        for attempt_idx in range(0, max_iters + 1):
            gate_report = _run_remote_cpp17_compile_gate(round_no, current_completion)
            last_gate_report = gate_report
            merged_report = _merge_remote_cpp17_compile_gate_report(current_verifier, gate_report)
            gate_compile_ok = bool(isinstance(gate_report, dict) and gate_report.get("compile_ok") is True)
            info["remote_cpp17_compile_gate"]["history"].append(
                {
                    "round": round_no,
                        "attempt": attempt_idx,
                        "compile_ok": gate_compile_ok,
                        "reason": str((gate_report or {}).get("reason") or ""),
                        "report": gate_report,
                    }
                )

            if gate_compile_ok or _remote_cpp17_compile_gate_should_fail_open(gate_report):
                if gate_compile_ok and attempt_idx > 0:
                    info["remote_cpp17_compile_gate"]["recovered_rounds"].append(round_no)
                info["remote_cpp17_compile_gate"]["final"] = gate_report
                return current_completion, merged_report, gate_report, False

            if attempt_idx >= max_iters:
                info["remote_cpp17_compile_gate"]["blocked"] = True
                info["remote_cpp17_compile_gate"]["blocked_rounds"].append(round_no)
                info["remote_cpp17_compile_gate"]["final"] = gate_report
                return current_completion, merged_report, gate_report, True

            info["remote_cpp17_compile_gate"]["blocked"] = True
            info["remote_cpp17_compile_gate"]["blocked_rounds"].append(round_no)
            info["remote_cpp17_compile_gate"]["final"] = gate_report
            return current_completion, merged_report, gate_report, True

        return current_completion, verifier_report, last_gate_report, False

    def _remote_eval(round_no: int) -> Dict:
        full_source = completion if COMPLETION_MODE == COMPLETION_MODE_FULL else (prompt_prefix + completion)
        ok_sync, sync_msg = _sync_to_remote(round_no, full_source, completion)
        if not ok_sync:
            return {
                "id": task_id,
                "eval_round": round_no,
                "compile_ok": 0,
                "run_ok": 0,
                "reason": "scp_fail",
                "remote_case_namespace": remote_case_namespace,
                "sync_msg": _tail_chars(sync_msg, 4000),
            }

        ssh_log = (dump_dir / f"_remote_eval_round{round_no}.log") if dump_dir else None

        if remote_eval_mode == "compile_only":
            res = remote_eval_compile_only(
                user=remote_user,
                host=remote_host,
                port=remote_port,
                key_path=remote_ssh_key,
                remote_src_path=remote_src,
                remote_obj_path=remote_obj,
                compiler=remote_compiler,
                cflags=remote_cflags,
                timeout_s=int(max(float(_run_timeout), float(_compile_timeout))),
                flock_path=remote_flock_path,
                no_strict_hostkey=remote_no_strict_hostkey,
                log_path=ssh_log,
                print_cmd=print_summary,
            )
            res["id"] = task_id
            res["eval_round"] = round_no
            res["remote_dir"] = remote_case_dir
            res["remote_src"] = remote_src
            res["remote_sample_file"] = remote_sample_file
            res["remote_case_namespace"] = remote_case_namespace
            res["_sync_msg_tail"] = _tail_chars(sync_msg, 2000)
            return res

        format_vars = {
            "task_id": task_id,
            "sample_idx": str(sample_idx),
            "intrinsic": intrinsic,
            "remote_dir": remote_case_dir,
            "remote_src": remote_src,
            "remote_obj": remote_obj,
            "remote_completion": remote_completion_path,
            "remote_sample_file": remote_sample_file,
            "cflags": remote_cflags,
            "timeout": str(_run_timeout),
            "compile_timeout": str(_compile_timeout),
        }

        # Built-in simdbench_one mode: call remote helper script
        if remote_eval_mode == "simdbench_one":
            if not remote_simdbench_eval.strip():
                return {
                    "id": task_id,
                    "compile_ok": 0,
                    "run_ok": 0,
                    "reason": "bad_remote_simdbench_eval",
                    "_sync_msg_tail": _tail_chars(sync_msg, 2000),
                }

            remote_cmd_tmpl = (
                f"{remote_simdbench_eval.strip()} "
                f"--sample_file {{remote_sample_file}} "
                f"--intrinsic {{intrinsic}} "
                f"--timeout {{timeout}} "
                f"--compile_timeout {{compile_timeout}} "
                f"--k {remote_simdbench_k} "
                f"--n_workers {int(remote_simdbench_n_workers)} "
                f"--tail_chars {int(remote_simdbench_tail_chars)} "
                f"--json"
            )
            if remote_simdbench_problem_file:
                remote_cmd_tmpl += f" --problem_file {remote_simdbench_problem_file}"
            if remote_simdbench_output_path:
                # you can set this to "{remote_dir}" to isolate per-sample artifacts
                remote_cmd_tmpl += f" --output_path {remote_simdbench_output_path}"

            js, blob = remote_eval_cmd_json(
                user=remote_user,
                host=remote_host,
                port=remote_port,
                key_path=remote_ssh_key,
                remote_cmd_template=remote_cmd_tmpl,
                format_vars=format_vars,
                timeout_s=_ssh_timeout_s,
                flock_path=remote_flock_path,
                no_strict_hostkey=remote_no_strict_hostkey,
                log_path=ssh_log,
                print_cmd=print_summary,
            )
            if js is None:
                return {
                    "id": task_id,
                    "compile_ok": 0,
                    "run_ok": 0,
                    "reason": "remote_cmd_fail",
                    "ssh_blob": _tail_chars(blob, 8000),
                    "_sync_msg_tail": _tail_chars(sync_msg, 2000),
                }
            js["id"] = js.get("id", task_id)
            js["eval_round"] = round_no
            js["remote_dir"] = remote_case_dir
            js["remote_sample_file"] = remote_sample_file
            js["remote_case_namespace"] = remote_case_namespace
            js["_sync_msg_tail"] = _tail_chars(sync_msg, 2000)
            js = _apply_remote_perf_precheck(round_no, completion, js)
            return js

        # Generic cmd mode: user-provided command template
        if remote_eval_mode == "cmd":
            js, blob = remote_eval_cmd_json(
                user=remote_user,
                host=remote_host,
                port=remote_port,
                key_path=remote_ssh_key,
                remote_cmd_template=remote_eval_cmd,
                format_vars=format_vars,
                timeout_s=_ssh_timeout_s,
                flock_path=remote_flock_path,
                no_strict_hostkey=remote_no_strict_hostkey,
                log_path=ssh_log,
                print_cmd=print_summary,
            )
            if js is None:
                return {
                    "id": task_id,
                    "compile_ok": 0,
                    "run_ok": 0,
                    "reason": "remote_cmd_fail",
                    "ssh_blob": _tail_chars(blob, 8000),
                    "_sync_msg_tail": _tail_chars(sync_msg, 2000),
                }
            js["id"] = js.get("id", task_id)
            js["eval_round"] = round_no
            js["remote_dir"] = remote_case_dir
            js["remote_sample_file"] = remote_sample_file
            js["remote_case_namespace"] = remote_case_namespace
            js["_sync_msg_tail"] = _tail_chars(sync_msg, 2000)
            return js

        return {
            "id": task_id,
            "eval_round": round_no,
            "compile_ok": 0,
            "run_ok": 0,
            "reason": f"bad_remote_eval_mode:{remote_eval_mode}",
            "remote_case_namespace": remote_case_namespace,
            "_sync_msg_tail": _tail_chars(sync_msg, 2000),
        }

    
    def _remote_eval_any(round_no: int, completion_text: str, *, tag: str = "", intrinsic_override: Optional[str] = None) -> Dict:
        """Evaluate an arbitrary completion (used for SERIAL reference debugging)."""
        intrinsic_eval = str(intrinsic_override or intrinsic)
        intrinsic_eval_lower = intrinsic_eval.strip().lower()
        full_source = completion_text if COMPLETION_MODE == COMPLETION_MODE_FULL else (prompt_prefix + completion_text)
        eval_task_id = task_id
        remote_problem_file_for_eval = remote_simdbench_problem_file
        if intrinsic_eval_lower == "scalar":
            scalar_problem_file = str(remote_simdbench_scalar_problem_file or "").strip()
            if scalar_problem_file:
                # A dedicated scalar problem file normally uses scalar task ids
                # such as SimdBench_32.  When scalar eval reuses the same SVE
                # problem file, keep the benchmark task id unchanged so rows
                # like VecIntrinBench_h2v1_merged_upsample_SVE remain findable.
                eval_task_id = _strip_intrinsic_suffix(task_id)
                remote_problem_file_for_eval = scalar_problem_file
        ok_sync, sync_msg = _sync_to_remote(
            round_no,
            full_source,
            completion_text,
            task_id_override=eval_task_id,
        )
        if not ok_sync:
            res = {
                "id": task_id,
                "eval_round": round_no,
                "compile_ok": 0,
                "run_ok": 0,
                "reason": "scp_fail",
                "remote_case_namespace": remote_case_namespace,
                "sync_msg": _tail_chars(sync_msg, 4000),
            }
            if tag:
                res["_tag"] = tag
            return res

        ssh_log = (dump_dir / f"_remote_eval_round{round_no}.log") if dump_dir else None

        if remote_eval_mode == "compile_only":
            res = remote_eval_compile_only(
                user=remote_user,
                host=remote_host,
                port=remote_port,
                key_path=remote_ssh_key,
                remote_src_path=remote_src,
                remote_obj_path=remote_obj,
                compiler=remote_compiler,
                cflags=remote_cflags,
                timeout_s=int(max(float(_run_timeout), float(_compile_timeout))),
                flock_path=remote_flock_path,
                no_strict_hostkey=remote_no_strict_hostkey,
                log_path=ssh_log,
                print_cmd=print_summary,
            )
            res["id"] = task_id
            res["eval_round"] = round_no
            res["remote_dir"] = remote_case_dir
            res["remote_src"] = remote_src
            res["remote_sample_file"] = remote_sample_file
            res["remote_case_namespace"] = remote_case_namespace
            res["_sync_msg_tail"] = _tail_chars(sync_msg, 2000)
            if tag:
                res["_tag"] = tag
            return res

        format_vars = {
            "task_id": task_id,
            "sample_idx": str(sample_idx),
            "intrinsic": intrinsic_eval,
            "remote_dir": remote_case_dir,
            "remote_src": remote_src,
            "remote_obj": remote_obj,
            "remote_completion": remote_completion_path,
            "remote_sample_file": remote_sample_file,
            "cflags": remote_cflags,
            "timeout": str(_run_timeout),
            "compile_timeout": str(_compile_timeout),
        }

        if remote_eval_mode == "simdbench_one":
            if not remote_simdbench_eval.strip():
                res = {
                    "id": task_id,
                    "eval_round": round_no,
                    "compile_ok": 0,
                    "run_ok": 0,
                    "reason": "bad_remote_simdbench_eval",
                    "remote_case_namespace": remote_case_namespace,
                    "_sync_msg_tail": _tail_chars(sync_msg, 2000),
                }
                if tag:
                    res["_tag"] = tag
                return res

            remote_cmd_tmpl = (
                f"{remote_simdbench_eval.strip()} "
                f"--sample_file {{remote_sample_file}} "
                f"--intrinsic {{intrinsic}} "
                f"--timeout {{timeout}} "
                f"--compile_timeout {{compile_timeout}} "
                f"--k {remote_simdbench_k} "
                f"--n_workers {int(remote_simdbench_n_workers)} "
                f"--tail_chars {int(remote_simdbench_tail_chars)} "
                f"--json"
            )
            if remote_problem_file_for_eval:
                remote_cmd_tmpl += f" --problem_file {remote_problem_file_for_eval}"
            if remote_simdbench_output_path:
                remote_cmd_tmpl += f" --output_path {remote_simdbench_output_path}"

            js, blob = remote_eval_cmd_json(
                user=remote_user,
                host=remote_host,
                port=remote_port,
                key_path=remote_ssh_key,
                remote_cmd_template=remote_cmd_tmpl,
                format_vars=format_vars,
                timeout_s=_ssh_timeout_s,
                flock_path=remote_flock_path,
                no_strict_hostkey=remote_no_strict_hostkey,
                log_path=ssh_log,
                print_cmd=print_summary,
            )
            if js is None:
                res = {
                    "id": task_id,
                    "eval_round": round_no,
                    "compile_ok": 0,
                    "run_ok": 0,
                    "reason": "remote_cmd_fail",
                    "remote_case_namespace": remote_case_namespace,
                    "ssh_blob": _tail_chars(blob, 8000),
                    "_sync_msg_tail": _tail_chars(sync_msg, 2000),
                }
                if tag:
                    res["_tag"] = tag
                return res

            js["id"] = js.get("id", task_id)
            js["eval_round"] = round_no
            js["remote_dir"] = remote_case_dir
            js["remote_sample_file"] = remote_sample_file
            js["remote_case_namespace"] = remote_case_namespace
            js["_sync_msg_tail"] = _tail_chars(sync_msg, 2000)
            if tag:
                js["_tag"] = tag
            return js

        if remote_eval_mode == "cmd":
            js, blob = remote_eval_cmd_json(
                user=remote_user,
                host=remote_host,
                port=remote_port,
                key_path=remote_ssh_key,
                remote_cmd_template=remote_eval_cmd,
                format_vars=format_vars,
                timeout_s=_ssh_timeout_s,
                flock_path=remote_flock_path,
                no_strict_hostkey=remote_no_strict_hostkey,
                log_path=ssh_log,
                print_cmd=print_summary,
            )
            if js is None:
                res = {
                    "id": task_id,
                    "eval_round": round_no,
                    "compile_ok": 0,
                    "run_ok": 0,
                    "reason": "remote_cmd_fail",
                    "remote_case_namespace": remote_case_namespace,
                    "ssh_blob": _tail_chars(blob, 8000),
                    "_sync_msg_tail": _tail_chars(sync_msg, 2000),
                }
                if tag:
                    res["_tag"] = tag
                return res

            js["id"] = js.get("id", task_id)
            js["eval_round"] = round_no
            js["remote_dir"] = remote_case_dir
            js["remote_sample_file"] = remote_sample_file
            js["remote_case_namespace"] = remote_case_namespace
            js["_sync_msg_tail"] = _tail_chars(sync_msg, 2000)
            if tag:
                js["_tag"] = tag
            return js

        res = {
            "id": task_id,
            "eval_round": round_no,
            "compile_ok": 0,
            "run_ok": 0,
            "reason": f"bad_remote_eval_mode:{remote_eval_mode}",
            "remote_case_namespace": remote_case_namespace,
            "_sync_msg_tail": _tail_chars(sync_msg, 2000),
        }
        if tag:
            res["_tag"] = tag
        return res

    def _remote_eval_harness(round_no: int, harness_source: str, *, tag: str = "serial_mismatch") -> Dict:
        """Compile + run a standalone harness on remote; expects JSON as last stdout line."""
        full_source = ensure_serial_mismatch_harness_preamble(harness_source)
        ok_sync, sync_msg = _sync_to_remote(round_no, full_source, harness_source)
        if not ok_sync:
            res = {
                "id": task_id,
                "eval_round": round_no,
                "compile_ok": 0,
                "run_ok": 0,
                "reason": "scp_fail",
                "remote_case_namespace": remote_case_namespace,
                "sync_msg": _tail_chars(sync_msg, 4000),
            }
            if tag:
                res["_tag"] = tag
            return res

        ssh_log = (dump_dir / f"_remote_eval_round{round_no}.log") if dump_dir else None
        format_vars = {
            "task_id": task_id,
            "sample_idx": str(sample_idx),
            "intrinsic": intrinsic,
            "remote_dir": remote_case_dir,
            "remote_src": remote_src,
            "remote_obj": remote_obj,
            "remote_completion": remote_completion_path,
            "remote_sample_file": remote_sample_file,
            "cflags": remote_cflags,
            "timeout": str(_run_timeout),
            "compile_timeout": str(_compile_timeout),
        }

        remote_cmd_tmpl = (
            f"bash -lc "
            f"\"{remote_compiler} {{remote_src}} {{cflags}} -std=c++17 -o {{remote_obj}} "
            f"&& timeout {{timeout}} {{remote_obj}}\""
        )

        js, blob = remote_eval_cmd_json(
            user=remote_user,
            host=remote_host,
            port=remote_port,
            key_path=remote_ssh_key,
            remote_cmd_template=remote_cmd_tmpl,
            format_vars=format_vars,
            timeout_s=_ssh_timeout_s,
            flock_path=remote_flock_path,
            no_strict_hostkey=remote_no_strict_hostkey,
            log_path=ssh_log,
            print_cmd=print_summary,
        )
        if js is None:
            res = {
                "id": task_id,
                "eval_round": round_no,
                "compile_ok": 0,
                "run_ok": 0,
                "reason": "remote_cmd_fail",
                "remote_case_namespace": remote_case_namespace,
                "ssh_blob": _tail_chars(blob, 8000),
                "_sync_msg_tail": _tail_chars(sync_msg, 2000),
            }
            if tag:
                res["_tag"] = tag
            return res

        js["id"] = js.get("id", task_id)
        js["eval_round"] = round_no
        js["remote_dir"] = remote_case_dir
        js["remote_src"] = remote_src
        js["remote_sample_file"] = remote_sample_file
        js["remote_case_namespace"] = remote_case_namespace
        js["_sync_msg_tail"] = _tail_chars(sync_msg, 2000)
        if tag:
            js["_tag"] = tag
        return js

    def _extract_controller_action(verifier_report: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(verifier_report, dict):
            return None
        payload = verifier_report.get("repair_prompt_payload")
        if isinstance(payload, dict):
            action = str(payload.get("controller_action") or "").strip()
            if action:
                return action
        action = str(verifier_report.get("repair_action") or "").strip()
        return action or None

    def _is_semantic_controller_action(action: Optional[str]) -> bool:
        return str(action or "").strip() in PASS1_V22_SEMANTIC_ACTIONS

    def _is_compile_controller_action(action: Optional[str]) -> bool:
        action = str(action or "").strip()
        return action.startswith("compile_") or action == "local_compile_fix"

    def _remote_result_is_success(res: Optional[Dict[str, Any]]) -> bool:
        res = res if isinstance(res, dict) else {}
        reason = str(res.get("reason") or "").strip().lower()
        try:
            compile_ok = int(res.get("compile_ok", 0) or 0)
            run_ok = int(res.get("run_ok", 0) or 0)
        except Exception:
            compile_ok, run_ok = 0, 0
        if reason == "ok":
            return True
        if remote_eval_mode == "compile_only":
            return compile_ok == 1
        return run_ok == 1

    def _remote_result_text(res: Optional[Dict[str, Any]]) -> str:
        res = res if isinstance(res, dict) else {}
        fields = [
            "reason",
            "simdbench_raw_result",
            "compile_log_tail",
            "run_log_tail",
            "ssh_blob",
            "sync_msg",
            "_sync_msg_tail",
        ]
        return "\n".join(str(res.get(k) or "") for k in fields).lower()

    def _is_logical_bug_result(res: Optional[Dict[str, Any]]) -> bool:
        text = _remote_result_text(res)
        return "logical_bug" in text or "logical bug" in text

    def _is_runtime_error_or_timeout_result(res: Optional[Dict[str, Any]]) -> bool:
        res = res if isinstance(res, dict) else {}
        text = _remote_result_text(res)
        reason = str(res.get("reason") or "").strip().lower()
        try:
            compile_ok = int(res.get("compile_ok", 0) or 0)
        except Exception:
            compile_ok = 0
        if reason in {"runtime_error", "run_timeout", "runtime_timeout", "timeout"}:
            return True
        runtime_markers = (
            "runtime failed",
            "runtime_error",
            "segmentation fault",
            "sigsegv",
            "double free",
            "invalid pointer",
            "glibc",
        )
        if any(marker in text for marker in runtime_markers):
            return True
        timeout_markers = ("run timeout", "runtime timeout", "timed out", "time limit", "timeout")
        return compile_ok == 1 and any(marker in text for marker in timeout_markers)

    def _no_improve_early_stop_suppression_reason(
        best_res: Optional[Dict[str, Any]],
        cur_res: Optional[Dict[str, Any]],
    ) -> str:
        if _is_runtime_error_or_timeout_result(cur_res):
            return "runtime_error_or_timeout"
        if _is_logical_bug_result(best_res) and _is_logical_bug_result(cur_res):
            return "consecutive_logical_bug"
        return ""

    def _tag_names_from_report(verifier_report: Optional[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        verifier_report = verifier_report if isinstance(verifier_report, dict) else {}
        compile_names: List[str] = []
        semantic_names: List[str] = []
        for item in (verifier_report.get("compile_risk_tags") or []):
            if isinstance(item, dict):
                name = str(item.get("tag") or "").strip()
                if name:
                    compile_names.append(name)
        for item in (verifier_report.get("semantic_risk_tags") or []):
            if isinstance(item, dict):
                name = str(item.get("tag") or "").strip()
                if name:
                    semantic_names.append(name)
        return compile_names, semantic_names

    phased_serial_bootstrap = str(serial_bootstrap_mode or "phased").strip() == "phased"

    def _serial_budget_count() -> int:
        if serial_cache is not None:
            try:
                return int(serial_cache.attempts(task_id))
            except Exception:
                pass
        return int(info["serial_ref"].get("serial_ref_budget_consumed_count") or 0)

    def _consume_serial_budget() -> int:
        new_count: Optional[int] = None
        if serial_cache is not None:
            try:
                new_count = int(serial_cache.bump_attempt(task_id))
            except Exception:
                new_count = None
        if new_count is None:
            new_count = int(info["serial_ref"].get("serial_ref_budget_consumed_count") or 0) + 1
        info["serial_ref"]["serial_ref_budget_consumed"] = True
        info["serial_ref"]["serial_ref_budget_consumed_count"] = int(
            max(new_count, int(info["serial_ref"].get("serial_ref_budget_consumed_count") or 0))
        )
        return int(info["serial_ref"]["serial_ref_budget_consumed_count"])

    def _serial_failure_kind_from_result(res: Optional[Dict[str, Any]]) -> str:
        if not isinstance(res, dict):
            return "infra"
        reason = str(res.get("reason") or "").strip()
        raw_parts = [
            str(res.get("simdbench_raw_result") or ""),
            str(res.get("compile_log_tail") or ""),
            str(res.get("run_log_tail") or ""),
            str(res.get("ssh_blob") or ""),
            str(res.get("sync_msg") or ""),
            str(res.get("_sync_msg_tail") or ""),
        ]
        haystack = "\n".join(raw_parts).lower()
        if "task_id_not_found_in_problem_file" in haystack:
            return "infra"
        if reason in {"scp_fail", "remote_cmd_fail", "bad_remote_simdbench_eval"}:
            return "infra"
        if reason.startswith("bad_remote_eval_mode"):
            return "infra"
        if "connection timed out" in haystack or "connection reset" in haystack:
            return "infra"
        return "serial_code"

    def _set_serial_bootstrap_metadata(stage: str, reason: str, source: str = "") -> None:
        if stage and str(info["serial_ref"].get("serial_bootstrap_stage") or "none") in {"", "none"}:
            info["serial_ref"]["serial_bootstrap_stage"] = stage
        if reason and not str(info["serial_ref"].get("serial_bootstrap_reason") or "").strip():
            info["serial_ref"]["serial_bootstrap_reason"] = reason
        if source:
            info["serial_ref"]["serial_bootstrap_source"] = source

    def _set_serial_bootstrap_attempt_source(source: str) -> None:
        source = str(source or "").strip()
        if source and not str(info["serial_ref"].get("serial_bootstrap_attempt_source") or "").strip():
            info["serial_ref"]["serial_bootstrap_attempt_source"] = source

    def _completion_bootstrap_hazard_reason(completion_text: str) -> str:
        normalized = strip_noncode(completion_text or "")
        issues = detect_forbidden_structures(normalized, func_name)
        for issue in issues:
            if issue == "missing_target_function":
                return "missing_target_function"
            if issue == "ellipsis_placeholder":
                return "ellipsis_placeholder"
            if issue.startswith("brace_delta_"):
                return "brace_closure_mismatch"
        body = try_extract_function_body(normalized, func_name) or normalized
        body = strip_comments(str(body or ""))
        body = re.sub(r"\s+", " ", body).strip()
        if not body or body == "}":
            return "empty_function_body"
        lowered = body.lower()
        for token in ["todo", "your code here", "not implemented", "placeholder", "stub"]:
            if token in lowered:
                return "placeholder_function_body"
        return ""

    def _infer_pre_remote_serial_reason(
        verifier_report: Optional[Dict[str, Any]],
        gate_report: Optional[Dict[str, Any]],
        completion_text: str,
    ) -> str:
        return ""

    def _classify_serial_trigger(
        res: Optional[Dict[str, Any]],
        verifier_report: Optional[Dict[str, Any]],
    ) -> Tuple[str, str]:
        if not isinstance(res, dict):
            return "", ""
        try:
            compile_ok = int(res.get("compile_ok", 0) or 0)
            run_ok = int(res.get("run_ok", 0) or 0)
        except Exception:
            compile_ok, run_ok = 0, 0
        reason = str(res.get("reason") or "").strip() or "unknown"
        if _serial_failure_kind_from_result(res) == "infra":
            return "", ""
        if compile_ok == 1 and run_ok == 0:
            return "logic_fail", reason
        if phased_serial_bootstrap and compile_ok != 1:
            return "compile_fail", reason
        return "", ""

    def _synthesize_gate_blocked_result(round_no: int, gate_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        gate_report = gate_report if isinstance(gate_report, dict) else {}
        diagnostics = gate_report.get("diagnostics")
        diag_blob = ""
        if diagnostics:
            try:
                diag_blob = json.dumps(diagnostics, ensure_ascii=False)
            except Exception:
                diag_blob = str(diagnostics)
        return {
            "id": task_id,
            "eval_round": round_no,
            "compile_ok": 0,
            "run_ok": 0,
            "passed": False,
            "reason": str(gate_report.get("reason") or "remote_cpp17_compile_gate_blocked"),
            "compile_log_tail": diag_blob,
            "run_log_tail": "",
            "simdbench_raw_result": str(gate_report.get("reason") or "remote_cpp17_compile_gate_blocked"),
            "remote_case_namespace": remote_case_namespace,
            "_synthetic_pre_remote_blocked": True,
        }

    def _ensure_serial_reference(
        *,
        stage: str,
        reason: str,
        round_hint: int,
        trigger_res: Optional[Dict[str, Any]],
    ) -> None:
        nonlocal serial_ref_completion, serial_ref_result, serial_ref_ok, serial_ref_attempted, serial_ref_round
        if serial_ref_ok:
            return
        if not serial_fallback:
            return
        if serial_model is None and not serial_use_solution_scalar:
            return

        info["serial_ref"]["triggered"] = True
        _set_serial_bootstrap_metadata(stage, reason)

        cache_rec: Optional[Dict[str, Any]] = None
        if serial_cache is not None:
            try:
                cache_rec = serial_cache.get(task_id)
            except Exception:
                cache_rec = None
            if isinstance(cache_rec, dict) and cache_rec.get("completion"):
                cache_source = str(cache_rec.get("backend") or "cache")
                if serial_use_solution_scalar and serial_solution_scalar and cache_source != "dataset_solution":
                    info["serial_ref"]["cache_skipped_reason"] = (
                        "solution_scalar_available; skip non-dataset_solution serial reference cache"
                    )
                    cache_rec = None
                else:
                    serial_ref_attempted = True
                    serial_ref_completion = str(cache_rec.get("completion") or "")
                    serial_ref_result = (
                        cache_rec.get("result")
                        if isinstance(cache_rec.get("result"), dict)
                        else {"passed": True, "compile_ok": 1, "run_ok": 1, "reason": "ok"}
                    )
                    serial_ref_ok = True
                    serial_ref_round = None
                    route_source = serial_bootstrap_completion or serial_ref_completion
                    route_label = serial_bootstrap_completion_source or cache_source
                    _reroute_serial_feedback_from_reference(route_source, route_label)
                    cache_round_label = f"cache_hit_r{max(0, int(round_hint))}"
                    info["serial_ref"].update(
                        {
                            "cache_hit": True,
                            "reused": True,
                            "attempted": False,
                            "ok": True,
                            "source": cache_source,
                            "round": None,
                            "round_label": cache_round_label,
                            "completion_len": len(serial_ref_completion),
                            "prompt_len": 0,
                            "serial_ref_validated": True,
                            "serial_ref_failure_kind": "none",
                            "bootstrap_semantics_source": route_label,
                            "bootstrap_semantics_completion_len": len(route_source or ""),
                        }
                    )
                    _set_serial_bootstrap_metadata(stage, reason, route_label)
                    if case_dir is not None:
                        try:
                            local_remote_dir = case_dir / "_remote"
                            local_remote_dir.mkdir(parents=True, exist_ok=True)
                            cpp = render_cpp_case(prompt_prefix, serial_ref_completion)
                            cache_case_file = local_remote_dir / "case_serial_ref_cache.cpp"
                            cache_case_file.write_text(cpp, encoding="utf-8")
                            info["serial_ref"]["code_file"] = str(cache_case_file)
                        except Exception:
                            pass
                    return

        if serial_use_solution_scalar and serial_solution_scalar and not serial_ref_ok:
            serial_ref_attempted = True
            _set_serial_bootstrap_attempt_source("dataset_solution")
            dataset_completion = build_serial_reference_from_solution(
                serial_solution_scalar,
                serial_entrypoint_scalar,
                serial_entrypoint_simd or (func_name or ""),
            )
            if dataset_completion:
                serial_eval_round = 9000 + max(0, int(round_hint))
                dataset_eval = _remote_eval_any(
                    serial_eval_round,
                    dataset_completion,
                    tag="serial_ref",
                    intrinsic_override=serial_eval_intrinsic,
                )
                _dump_json("serial_result", dataset_eval, serial_eval_round)
                serial_ref_result = dataset_eval
                passed = bool(dataset_eval) and dataset_eval.get("passed") is True
                failure_kind = "none" if passed else _serial_failure_kind_from_result(dataset_eval)
                budget_consumed = False
                if (not passed) and failure_kind != "infra":
                    _consume_serial_budget()
                    budget_consumed = True
                info["serial_ref"]["attempt_history"].append(
                    {
                        "attempt": 0,
                        "round": serial_eval_round,
                        "passed": bool(passed),
                        "compile_ok": (dataset_eval or {}).get("compile_ok"),
                        "run_ok": (dataset_eval or {}).get("run_ok"),
                        "reason": (dataset_eval or {}).get("reason"),
                        "source": "dataset_solution",
                        "failure_kind": failure_kind,
                        "budget_consumed": budget_consumed,
                    }
                )
                if passed:
                    serial_ref_completion = dataset_completion
                    serial_ref_ok = True
                    serial_ref_round = serial_eval_round
                    route_source = serial_bootstrap_completion or serial_ref_completion
                    route_label = serial_bootstrap_completion_source or "dataset_solution"
                    _reroute_serial_feedback_from_reference(route_source, route_label)
                    saved = False
                    if serial_cache is not None:
                        try:
                            saved = serial_cache.save_passed(
                                task_id=task_id,
                                func_name=func_name,
                                completion=serial_ref_completion,
                                backend="dataset_solution",
                                model="dataset_solution",
                                result=dataset_eval,
                            )
                        except Exception:
                            saved = False
                    info["serial_ref"].update(
                        {
                            "used_dataset_solution": True,
                            "source": "dataset_solution",
                            "ok": True,
                            "round": int(serial_ref_round),
                            "completion_len": len(serial_ref_completion),
                            "prompt_len": 0,
                            "saved_to_cache": bool(saved),
                            "serial_ref_validated": True,
                            "serial_ref_failure_kind": "none",
                            "bootstrap_semantics_source": route_label,
                            "bootstrap_semantics_completion_len": len(route_source or ""),
                        }
                    )
                    _set_serial_bootstrap_metadata(stage, reason, route_label)
                    return
                info["serial_ref"]["used_dataset_solution"] = True
                info["serial_ref"]["source"] = "dataset_solution"
                info["serial_ref"]["serial_ref_failure_kind"] = failure_kind

        if serial_model is None or serial_ref_ok:
            return

        max_budget = max(1, int(serial_ref_max_attempts_per_task))
        if _serial_budget_count() >= max_budget:
            info["serial_ref"].update(
                {
                    "attempted": False,
                    "ok": False,
                    "note": f"serial_ref max attempts reached for task_id (max={max_budget}); skip generation",
                }
            )
            return

        simd_fb = make_remote_feedback_text(
            trigger_res,
            tail_chars=remote_simdbench_tail_chars,
            max_chars=remote_feedback_max_chars,
        )
        attempts_used = int(info["serial_ref"].get("attempts_used") or 0)
        attempt_history = list(info["serial_ref"].get("attempt_history") or [])
        prev_code: Optional[str] = None
        prev_fb: Optional[str] = None
        last_serial_prompt = ""
        last_serial_eval_round = -1
        last_failure_kind = str(info["serial_ref"].get("serial_ref_failure_kind") or "none")
        infra_failures = 0

        info["serial_ref"]["attempted"] = True
        info["serial_ref"]["cache_hit"] = False
        info["serial_ref"]["reused"] = False
        _set_serial_bootstrap_attempt_source(serial_llm_backend or "serial_llm")

        while _serial_budget_count() < max_budget:
            attempts_used += 1
            budget_slot = _serial_budget_count() + 1
            serial_eval_round = 10000 + max(0, int(round_hint)) + (attempts_used - 1) * 10
            last_serial_eval_round = serial_eval_round

            serial_prompt = build_serial_reference_prompt(
                spec_text=spec_text,
                prompt_prefix=prompt_prefix,
                func_name=func_name,
                attempt_idx=budget_slot,
                prev_serial_completion=prev_code,
                prev_serial_feedback=prev_fb,
                simd_failure_feedback=simd_fb,
            )
            if serial_prompt_max_chars > 0:
                serial_prompt = _truncate_middle(serial_prompt, serial_prompt_max_chars)

            last_serial_prompt = serial_prompt
            _dump_text("serial_prompt", serial_prompt, serial_eval_round)
            _serial_gen_backend = (
                "api" if serial_llm_backend in ("openai", "deepseek", "api") else "hf"
            )
            serial_completion = generate_text(
                serial_model,
                serial_tok,
                user_text=serial_prompt,
                system_text=serial_system_prompt,
                max_new_tokens=serial_max_new_tokens,
                do_sample=serial_do_sample,
                temperature=serial_temperature,
                top_p=serial_top_p,
                repetition_penalty=serial_repetition_penalty,
                backend=_serial_gen_backend,
                api_model=serial_api_model,
                is_t0=(serial_eval_round == 0),
            )
            serial_completion = strip_noncode(serial_completion)
            serial_completion = normalize_completion_snippet(
                serial_completion, func_name=func_name, skip_required_includes=True
            )
            _dump_text("serial_completion", serial_completion, serial_eval_round)

            if looks_like_simd_output(serial_completion, func_name=func_name):
                _consume_serial_budget()
                last_failure_kind = "serial_code"
                prev_code = serial_completion
                prev_fb = (
                    "Rejected: your output contains SIMD/SVE/NEON constructs. "
                    "You MUST output scalar loops only."
                )
                attempt_history.append(
                    {
                        "attempt": budget_slot,
                        "round": serial_eval_round,
                        "passed": False,
                        "reason": "simd_detected",
                        "source": "serial_llm",
                        "failure_kind": "serial_code",
                        "budget_consumed": True,
                    }
                )
                continue

            serial_eval = _remote_eval_any(
                serial_eval_round,
                serial_completion,
                tag="serial_ref",
                intrinsic_override=serial_eval_intrinsic,
            )
            _dump_json("serial_result", serial_eval, serial_eval_round)
            serial_ref_result = serial_eval
            passed = bool(serial_eval) and serial_eval.get("passed") is True
            failure_kind = "none" if passed else _serial_failure_kind_from_result(serial_eval)
            budget_consumed = False
            if (not passed) and failure_kind != "infra":
                _consume_serial_budget()
                budget_consumed = True
            attempt_history.append(
                {
                    "attempt": budget_slot,
                    "round": serial_eval_round,
                    "passed": bool(passed),
                    "compile_ok": (serial_eval or {}).get("compile_ok"),
                    "run_ok": (serial_eval or {}).get("run_ok"),
                    "reason": (serial_eval or {}).get("reason"),
                    "source": "serial_llm",
                    "failure_kind": failure_kind,
                    "budget_consumed": budget_consumed,
                }
            )
            if passed:
                serial_ref_completion = serial_completion
                serial_ref_ok = True
                serial_ref_round = serial_eval_round
                route_source = serial_bootstrap_completion or serial_ref_completion
                route_label = serial_bootstrap_completion_source or serial_llm_backend or "serial_llm"
                _reroute_serial_feedback_from_reference(route_source, route_label)
                saved = False
                if serial_cache is not None:
                    try:
                        saved = serial_cache.save_passed(
                            task_id=task_id,
                            func_name=func_name,
                            completion=serial_ref_completion,
                            backend=serial_llm_backend,
                            model=serial_model,
                            result=serial_eval,
                        )
                    except Exception:
                        saved = False
                info["serial_ref"].update(
                    {
                        "source": serial_llm_backend or "serial_llm",
                        "ok": True,
                        "round": int(serial_ref_round),
                        "completion_len": len(serial_ref_completion),
                        "prompt_len": len(last_serial_prompt),
                        "saved_to_cache": bool(saved),
                        "serial_ref_validated": True,
                        "serial_ref_failure_kind": "none",
                        "bootstrap_semantics_source": route_label,
                        "bootstrap_semantics_completion_len": len(route_source or ""),
                    }
                )
                _set_serial_bootstrap_metadata(stage, reason, route_label)
                break

            last_failure_kind = failure_kind
            prev_code = serial_completion
            prev_fb = make_remote_feedback_text(
                serial_eval,
                tail_chars=remote_simdbench_tail_chars,
                max_chars=remote_feedback_max_chars,
            )
            if failure_kind == "infra":
                infra_failures += 1
                if infra_failures >= 2:
                    break
            else:
                infra_failures = 0

        info["serial_ref"]["attempts_used"] = int(attempts_used)
        info["serial_ref"]["attempt_history"] = attempt_history
        if not serial_ref_ok:
            info["serial_ref"].update(
                {
                    "ok": False,
                    "round": int(last_serial_eval_round) if last_serial_eval_round >= 0 else None,
                    "prompt_len": len(last_serial_prompt),
                    "serial_ref_validated": False,
                    "serial_ref_failure_kind": last_failure_kind or "none",
                }
            )

    def _promote_failed_untagged_report(
        verifier_report: Optional[Dict[str, Any]],
        res: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        verifier_report = deepcopy(verifier_report) if isinstance(verifier_report, dict) else {}
        if isinstance(res, dict) and not _remote_result_is_success(res):
            verifier_report["remote_failure_reason"] = str(res.get("reason") or "").strip() or "unknown_failure"
        return verifier_report

    def _promote_remote_compile_diagnostics_report(
        verifier_report: Optional[Dict[str, Any]],
        res: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        verifier_report = deepcopy(verifier_report) if isinstance(verifier_report, dict) else {}
        res = res if isinstance(res, dict) else {}
        try:
            compile_ok = int(res.get("compile_ok", 0) or 0)
        except Exception:
            compile_ok = 0
        diagnostics = extract_remote_compile_diagnostics(str(res.get("compile_log_tail", "") or ""))
        if compile_ok == 1 or not diagnostics:
            return verifier_report

        verifier_report["ok"] = False
        verifier_report["remote_compile_diagnostics"] = diagnostics
        compile_tags = list(verifier_report.get("compile_risk_tags") or [])
        seen_compile = {
            (
                str(item.get("tag") or "").strip(),
                str(item.get("evidence") or "").strip(),
            )
            for item in compile_tags
            if isinstance(item, dict)
        }

        def _append_compile_tag(tag: str, summary: str, evidence: str, *, severity: str = "error") -> None:
            key = (str(tag).strip(), str(evidence).strip())
            if key in seen_compile:
                return
            seen_compile.add(key)
            compile_tags.append(
                {
                    "tag": str(tag).strip(),
                    "severity": severity,
                    "summary": summary,
                    "line": None,
                    "evidence": str(evidence).strip(),
                }
            )

        for name in diagnostics.get("missing_index_symbols") or []:
            _append_compile_tag(
                "missing_index_helper_definition",
                "remote compile log confirms an index helper is still called without an in-file definition",
                name,
            )
        for name in diagnostics.get("missing_helper_symbols") or []:
            _append_compile_tag(
                "missing_local_helper_definition",
                "remote compile log confirms a helper-like local symbol is still called without an in-file definition",
                name,
            )
        for name in diagnostics.get("unsupported_symbols") or []:
            _append_compile_tag(
                "unsupported_builtin_symbol" if str(name).startswith("__builtin_") else "unsupported_sv_symbol",
                "remote compile log confirms an unsupported builtin or suspicious sv* symbol spelling",
                name,
            )
        for signal in diagnostics.get("syntax_signals") or []:
            _append_compile_tag(
                "probable_unbalanced_delimiters",
                "remote compile log reports a syntax-closure problem near this line",
                signal,
                severity="warning",
            )

        verifier_report["compile_risk_tags"] = compile_tags
        symbol_targets = _clean_named_diag_map(
            verifier_report.get("symbol_closure_targets") or {},
            per_key_limit=12,
        )
        if diagnostics.get("missing_helper_symbols"):
            symbol_targets["missing_helper_symbols"] = _dedup_text_items(
                list(symbol_targets.get("missing_helper_symbols") or []) + list(diagnostics.get("missing_helper_symbols") or []),
                limit=12,
            )
        if diagnostics.get("missing_index_symbols"):
            symbol_targets["missing_index_symbols"] = _dedup_text_items(
                list(symbol_targets.get("missing_index_symbols") or []) + list(diagnostics.get("missing_index_symbols") or []),
                limit=12,
            )
        unsupported_builtin_symbols = [
            x for x in (diagnostics.get("unsupported_symbols") or []) if str(x).startswith("__builtin_")
        ]
        unsupported_sv_symbols = [
            x for x in (diagnostics.get("unsupported_symbols") or []) if str(x).startswith("sv")
        ]
        if unsupported_builtin_symbols:
            symbol_targets["unsupported_builtin_symbols"] = _dedup_text_items(
                list(symbol_targets.get("unsupported_builtin_symbols") or []) + unsupported_builtin_symbols,
                limit=12,
            )
        if unsupported_sv_symbols:
            symbol_targets["unsupported_sv_symbols"] = _dedup_text_items(
                list(symbol_targets.get("unsupported_sv_symbols") or []) + unsupported_sv_symbols,
                limit=12,
            )
        if diagnostics.get("syntax_signals"):
            symbol_targets["syntax_signals"] = _dedup_text_items(
                list(symbol_targets.get("syntax_signals") or []) + list(diagnostics.get("syntax_signals") or []),
                limit=8,
            )

        if symbol_targets:
            verifier_report["symbol_closure_targets"] = symbol_targets
        return verifier_report

    def _score_parts(
        res: Optional[Dict[str, Any]],
        verifier_report: Optional[Dict[str, Any]],
        completion_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        res = res if isinstance(res, dict) else {}
        verifier_report = verifier_report if isinstance(verifier_report, dict) else {}
        payload = verifier_report.get("repair_prompt_payload")
        payload = payload if isinstance(payload, dict) else {}
        compile_risk_tags = verifier_report.get("compile_risk_tags") or []
        semantic_risk_tags = verifier_report.get("semantic_risk_tags") or []
        must_fix_tags: List[Any] = []
        diff_len: Optional[int] = None
        try:
            if serial_ref_ok and serial_ref_completion and completion_text:
                a_code = serial_ref_completion
                b_code = completion_text
                if func_name:
                    a_body = try_extract_function_body(a_code, func_name)
                    b_body = try_extract_function_body(b_code, func_name)
                    if a_body:
                        a_code = a_body
                    if b_body:
                        b_code = b_body
                diff_len = int(
                    len(
                        make_unified_diff(
                            a_code,
                            b_code,
                            fromfile="serial_ref",
                            tofile="simd_current",
                            context_lines=int(serial_diff_context_lines or 3),
                            max_chars=int(serial_diff_max_chars or 4000),
                        )
                        or ""
                    )
                )
        except Exception:
            diff_len = None
        try:
            compile_ok = int(res.get("compile_ok", 0) or 0)
            run_ok = int(res.get("run_ok", 0) or 0)
        except Exception:
            compile_ok, run_ok = 0, 0
        parts = {
            "run_ok": run_ok,
            "compile_ok": compile_ok,
            "semantic_risk_count": len(semantic_risk_tags),
            "compile_risk_count": len(compile_risk_tags),
            "must_fix_count": len(must_fix_tags),
            "serial_ref_diff_len": diff_len,
            "remote_reason": str(res.get("reason") or ""),
        }
        parts["composite_score"] = [
            int(parts["run_ok"]),
            int(parts["compile_ok"]),
            (-int(diff_len) if diff_len is not None else None),
        ]
        return parts

    def _score_vector(parts: Dict[str, Any]) -> Tuple[int, ...]:
        if remote_score_mode != "composite":
            if remote_eval_mode == "compile_only":
                return (int(parts.get("compile_ok", 0) or 0),)
            return (
                int(parts.get("run_ok", 0) or 0),
                int(parts.get("compile_ok", 0) or 0),
            )
        return (
            int(parts.get("run_ok", 0) or 0),
            int(parts.get("compile_ok", 0) or 0),
        )

    def _is_score_better(cur_parts: Dict[str, Any], best_parts: Dict[str, Any]) -> bool:
        cur_vec = _score_vector(cur_parts)
        best_vec = _score_vector(best_parts)
        if cur_vec != best_vec:
            return cur_vec > best_vec
        cur_diff = cur_parts.get("serial_ref_diff_len", None)
        best_diff = best_parts.get("serial_ref_diff_len", None)
        if cur_diff is not None and best_diff is not None and cur_diff != best_diff:
            return int(cur_diff) < int(best_diff)
        return False

    def _has_semantic_progress(prev_parts: Dict[str, Any], cur_parts: Dict[str, Any]) -> bool:
        if int(cur_parts.get("run_ok", 0) or 0) != int(prev_parts.get("run_ok", 0) or 0):
            return False
        if int(cur_parts.get("compile_ok", 0) or 0) != int(prev_parts.get("compile_ok", 0) or 0):
            return False
        if int(cur_parts.get("semantic_risk_count", 0) or 0) < int(prev_parts.get("semantic_risk_count", 0) or 0):
            return True
        if (
            int(cur_parts.get("semantic_risk_count", 0) or 0) == int(prev_parts.get("semantic_risk_count", 0) or 0)
            and int(cur_parts.get("must_fix_count", 0) or 0) < int(prev_parts.get("must_fix_count", 0) or 0)
        ):
            return True
        return False

    def _pre_remote_score_parts(
        verifier_report: Optional[Dict[str, Any]],
        completion_text: str,
    ) -> Dict[str, Any]:
        verifier_report = verifier_report if isinstance(verifier_report, dict) else {}
        compile_risk_tags = verifier_report.get("compile_risk_tags") or []
        semantic_risk_tags = verifier_report.get("semantic_risk_tags") or []
        must_fix_tags: List[Any] = []
        parts = {
            "compile_risk_count": len(compile_risk_tags),
            "semantic_risk_count": len(semantic_risk_tags),
            "must_fix_count": len(must_fix_tags),
            "completion_len": len(completion_text or ""),
        }
        parts["score_vector"] = [
            int(parts["compile_risk_count"]),
            int(parts["semantic_risk_count"]),
            int(parts["must_fix_count"]),
            int(parts["completion_len"]),
        ]
        return parts

    def _is_pre_remote_better(cur_parts: Dict[str, Any], best_parts: Dict[str, Any]) -> bool:
        return tuple(int(x) for x in (cur_parts.get("score_vector") or [])) < tuple(
            int(x) for x in (best_parts.get("score_vector") or [])
        )

    def _apply_local_repair_hygiene(completion_text: str) -> Tuple[str, Dict[str, Any]]:
        code = normalize_completion_snippet(completion_text, func_name)
        info_local: Dict[str, Any] = {"postprocess": None, "name_reapply": None, "shape_reapply": None}
        if intrinsic == "SVE":
            code, pp_info = postprocess_sve_common_fixes(code)
            info_local["postprocess"] = pp_info

        if reapply_name_shape and whitelist_set is not None and whitelist_list is not None and op_index is not None:
            invalid = sorted(
                [c for c in extract_calls(code, whitelist_set, context_text=spec_text) if c not in whitelist_set]
            )
            if invalid and reapply_name_max_iters > 0:
                code, name_info = rag_fix_names(
                    model=model,
                    tok=tok,
                    completion_in=code,
                    whitelist_set=whitelist_set,
                    whitelist_list=whitelist_list,
                    op_index=op_index,
                    spec_text=spec_text,
                    max_iters=1,
                    max_new_tokens=max_new_tokens,
                    top_k=8,
                    cutoff=0.55,
                    do_sample=False,
                    temperature=_repair_temperature_for_backend(model, temperature),
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    sid=task_id,
                    rank=0,
                    func_name=func_name,
                    attempts_per_iter=1,
                    print_prompts=False,
                    dump_prompts_dir=None,
                )
                info_local["name_reapply"] = name_info

            if sigs and reapply_shape_max_iters > 0:
                code, shape_info = rag_fix_shapes(
                    model=model,
                    tok=tok,
                    completion_in=code,
                    whitelist_set=whitelist_set,
                    whitelist_list=whitelist_list,
                    sigs=sigs,
                    rets=rets,
                    max_iters=1,
                    spec_text=spec_text,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=_repair_temperature_for_backend(model, temperature),
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    check_types=True,
                    sid=task_id,
                    rank=0,
                    func_name=func_name,
                    attempts_per_iter=1,
                    print_summary=False,
                    print_prompts=False,
                    dump_prompts_dir=None,
                )
                info_local["shape_reapply"] = shape_info
        return code, info_local

    _STRUCTURED_PATCH_BLOCK_RE = re.compile(
        r"\[PATCH\]\s*SEARCH\s*\n(.*?)\nREPLACE\s*\n(.*?)\n\[/PATCH\]",
        re.DOTALL,
    )

    def _should_try_structured_patch(controller_action: Optional[str]) -> bool:
        return False

    def _strip_optional_code_fence(text: str) -> str:
        raw = str(text or "").strip()
        if not raw.startswith("```"):
            return raw
        lines = raw.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _build_structured_patch_prompt(base_prompt: str) -> str:
        return (
            "[STRUCTURED_PATCH_REPAIR_TASK]\n"
            "Ignore any instruction inside REFERENCE_GUIDANCE that asks for a full rewritten file.\n"
            "In this mode, output ONLY one or more exact-match patch blocks.\n\n"
            "Patch format:\n"
            "[PATCH]\n"
            "SEARCH\n"
            "<exact text copied from PREV_COMPLETION>\n"
            "REPLACE\n"
            "<replacement text>\n"
            "[/PATCH]\n\n"
            "Rules:\n"
            "- SEARCH must match PREV_COMPLETION exactly, including spaces and punctuation.\n"
            "- Keep patches minimal and local.\n"
            "- Do not output the full file.\n"
            "- Do not add commentary.\n"
            "- If you cannot express the repair as exact local patches, output exactly [PATCH_FALLBACK].\n\n"
            "[REFERENCE_GUIDANCE]\n"
            + str(base_prompt or "").strip()
            + "\n"
        )

    def _parse_structured_patch_response(raw_text: str) -> Tuple[List[Dict[str, str]], str]:
        cleaned = _strip_optional_code_fence(raw_text)
        if not cleaned:
            return [], "empty_response"
        if "[PATCH_FALLBACK]" in cleaned:
            return [], "fallback_requested"
        patches: List[Dict[str, str]] = []
        for match in _STRUCTURED_PATCH_BLOCK_RE.finditer(cleaned):
            search = str(match.group(1) or "")
            replace = str(match.group(2) or "")
            if not search:
                return [], "empty_search_block"
            patches.append({"search": search, "replace": replace})
        if not patches:
            return [], "no_patch_blocks"
        return patches, ""

    def _apply_structured_patch_response(prev_completion: str, raw_text: str) -> Tuple[Optional[str], Dict[str, Any]]:
        patches, parse_error = _parse_structured_patch_response(raw_text)
        info_local: Dict[str, Any] = {
            "attempted": True,
            "applied": False,
            "parse_error": parse_error,
            "patch_count": len(patches),
            "failure_reason": parse_error,
        }
        if parse_error:
            return None, info_local

        updated = str(prev_completion or "")
        applied_patches: List[Dict[str, Any]] = []
        for idx, patch in enumerate(patches):
            search = str(patch.get("search") or "")
            replace = str(patch.get("replace") or "")
            occurrences = updated.count(search)
            if occurrences != 1:
                info_local["failure_reason"] = (
                    "search_not_found" if occurrences == 0 else "search_ambiguous"
                )
                info_local["failed_patch_index"] = idx
                info_local["failed_search_excerpt"] = _tail_chars(search, 400)
                return None, info_local
            updated = updated.replace(search, replace, 1)
            applied_patches.append(
                {
                    "index": idx,
                    "search_len": len(search),
                    "replace_len": len(replace),
                }
            )

        info_local["applied"] = True
        info_local["failure_reason"] = ""
        info_local["applied_patches"] = applied_patches
        return updated, info_local

    def _run_patch_first_repair(
        *,
        base_prompt: str,
        prev_completion: str,
        dump_tag_prefix: str,
        dump_round: int,
        controller_action: Optional[str],
    ) -> Tuple[str, Dict[str, Any]]:
        info_local: Dict[str, Any] = {
            "mode": "full_rewrite",
            "controller_action": str(controller_action or ""),
            "patch": {
                "attempted": False,
                "applied": False,
                "parse_error": "",
                "patch_count": 0,
                "failure_reason": "",
            },
            "fallback_used": False,
        }

        if _should_try_structured_patch(controller_action):
            info_local["patch"]["attempted"] = True
            patch_prompt = _build_structured_patch_prompt(base_prompt)
            if prompt_max_chars and prompt_max_chars > 0 and len(patch_prompt) > int(prompt_max_chars):
                patch_prompt = _truncate_middle(patch_prompt, int(prompt_max_chars))
            _maybe_dump(f"{dump_tag_prefix}_patch_prompt", patch_prompt, dump_round)
            patch_raw = generate_text(
                model,
                tok,
                user_text=patch_prompt,
                system_text="",
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            _maybe_dump(f"{dump_tag_prefix}_patch_response", patch_raw, dump_round)
            patched_code, patch_info = _apply_structured_patch_response(prev_completion, patch_raw)
            info_local["patch"] = patch_info
            if patched_code is not None:
                info_local["mode"] = "structured_patch"
                return patched_code, info_local
            info_local["fallback_used"] = True

        full_prompt = base_prompt
        if prompt_max_chars and prompt_max_chars > 0 and len(full_prompt) > int(prompt_max_chars):
            full_prompt = _truncate_middle(full_prompt, int(prompt_max_chars))
        _maybe_dump(f"{dump_tag_prefix}_full_prompt", full_prompt, dump_round)
        full_out = generate_text(
            model,
            tok,
            user_text=full_prompt,
            system_text="",
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        _maybe_dump(f"{dump_tag_prefix}_full_completion", full_out, dump_round)
        return full_out, info_local

    info["pre_remote_selection"] = {
        "enabled": False,
        "repair_generated": False,
        "selected_source": "original",
        "selected_score": None,
        "skipped_reason": "",
        "original": {},
        "repair": {},
    }

    selected_source = "original"
    verifier_res0: Optional[Dict[str, Any]]
    original_completion = completion
    original_verifier = _run_static_verifier(round_no=-2, completion_text=original_completion)
    if original_verifier is not None:
        info["pre_remote_selection"]["enabled"] = True
        original_score = _pre_remote_score_parts(original_verifier, original_completion)
        info["pre_remote_selection"]["original"] = {
            "controller_action": _extract_controller_action(original_verifier),
            "score": original_score,
        }

        selected_completion = original_completion
        verifier_res0 = original_verifier
        selected_score = original_score
        skipped_reason = "verifier_repair_prompt_disabled"
        completion = original_completion

        info["pre_remote_selection"]["selected_source"] = selected_source
        info["pre_remote_selection"]["selected_score"] = selected_score
        info["pre_remote_selection"]["skipped_reason"] = skipped_reason
        info["static_verifier"]["pre_remote_selected"] = verifier_res0
    else:
        verifier_res0 = None
        info["pre_remote_selection"]["enabled"] = False
        info["pre_remote_selection"]["skipped_reason"] = "static_verifier_disabled"
        info["static_verifier"]["pre_remote_selected"] = verifier_res0

    completion, verifier_res0, _gate_res0, gate_blocked0 = _apply_remote_cpp17_compile_gate(
        round_no=0,
        completion_text=completion,
        verifier_report=verifier_res0,
    )
    pre_remote_serial_reason = _infer_pre_remote_serial_reason(verifier_res0, _gate_res0, completion)
    info["pre_remote_selection"]["serial_bootstrap"] = {
        "triggered": bool(pre_remote_serial_reason),
        "reason": pre_remote_serial_reason,
        "serial_ref_ok": False,
    }
    if pre_remote_serial_reason:
        bootstrap_seed_res = (
            _synthesize_gate_blocked_result(-3, _gate_res0)
            if gate_blocked0
            else {
                "id": task_id,
                "eval_round": -3,
                "compile_ok": 0,
                "run_ok": 0,
                "passed": False,
                "reason": pre_remote_serial_reason,
                "compile_log_tail": "",
                "run_log_tail": "",
                "simdbench_raw_result": pre_remote_serial_reason,
                "remote_case_namespace": remote_case_namespace,
                "_synthetic_pre_remote_blocked": True,
            }
        )
        _ensure_serial_reference(
            stage="pre_remote",
            reason=pre_remote_serial_reason,
            round_hint=0,
            trigger_res=bootstrap_seed_res,
        )
        info["pre_remote_selection"]["serial_bootstrap"]["serial_ref_ok"] = bool(serial_ref_ok)
        if serial_ref_ok:
            bootstrap_prompt = build_remote_feedback_prompt(
                spec_text=spec_text,
                prompt_prefix=prompt_prefix,
                prev_completion=completion,
                remote_result=bootstrap_seed_res,
                compile_only=True,
                static_verifier_report=verifier_res0,
                serial_ref_completion=(serial_ref_completion if serial_ref_ok else None),
                serial_bootstrap_completion=(serial_bootstrap_completion if serial_ref_ok else None),
                serial_ref_result=(serial_ref_result if serial_ref_ok else None),
                serial_diff="",
                serial_mismatch_info=None,
                serial_code_max_chars=int(serial_prompt_max_chars or 0),
                serial_feedback_style=serial_feedback_style_effective,
                serial_feedback_style_requested=serial_feedback_style_requested,
                serial_pseudocode_max_chars=int(serial_pseudocode_max_chars or 0),
                serial_diff_max_chars=int(serial_diff_max_chars or 4000),
                serial_bootstrap_stage="pre_remote",
                serial_bootstrap_reason=pre_remote_serial_reason,
                serial_bootstrap_source=str(
                    info["serial_ref"].get("serial_bootstrap_source")
                    or info["serial_ref"].get("source")
                    or ""
                ),
                serial_ast_bootstrap_pseudocode=str(serial_ast_bootstrap_pseudocode or ""),
                serial_ast_bootstrap_source=str(serial_ast_bootstrap_source or ""),
                compile_feedback_style=str(remote_compile_feedback_style or "structured"),
                disable_bootstrap_guard=bool(disable_bootstrap_guard),
            )
            _maybe_dump("remote_prompt_pre_remote_serial_bootstrap", bootstrap_prompt, -3)
            out, repair_generation = _run_patch_first_repair(
                base_prompt=bootstrap_prompt,
                prev_completion=completion,
                dump_tag_prefix="pre_remote_serial_bootstrap",
                dump_round=-3,
                controller_action=None,
            )
            completion = normalize_completion_snippet(out, func_name)
            if intrinsic == "SVE":
                completion, _pp_info = postprocess_sve_common_fixes(completion)
            verifier_res0 = _run_static_verifier(round_no=-3, completion_text=completion)
            completion, verifier_res0, _gate_res0, gate_blocked0 = _apply_remote_cpp17_compile_gate(
                round_no=0,
                completion_text=completion,
                verifier_report=verifier_res0,
            )
            info["pre_remote_selection"]["serial_bootstrap"]["repair_generation"] = repair_generation

    if gate_blocked0:
        res0 = _synthesize_gate_blocked_result(0, _gate_res0)
    else:
        res0 = _remote_eval(round_no=0)
    verifier_res0 = _promote_failed_untagged_report(verifier_res0, res0)
    verifier_res0 = _promote_remote_compile_diagnostics_report(verifier_res0, res0)
    score_parts0 = _score_parts(res0, verifier_res0, completion)
    info["history"].append(
        {
            "round": 0,
            "selected_source": selected_source,
            "static_verifier": verifier_res0,
            "result": res0,
            "composite_score": list(score_parts0.get("composite_score") or []),
            "composite_score_parts": score_parts0,
            "best_score_parts": score_parts0,
        }
    )

    if print_summary:
        print("=" * 120)
        print(f"[REMOTE_EVAL] id={task_id} sample={sample_idx} round=0 compile_ok={res0.get('compile_ok')} run_ok={res0.get('run_ok')} reason={res0.get('reason')}")
        print("=" * 120)

    if _remote_result_is_success(res0):
        info["last"] = res0
        info["best"] = res0
        info["best_round"] = 0
        info["final"] = res0
        info["last_score_parts"] = score_parts0
        info["best_score_parts"] = score_parts0
        info["static_verifier"]["final"] = verifier_res0
        return completion, info


    last_res = res0
    last_completion = completion
    last_score_parts = score_parts0
    best_completion = completion
    best_res = res0
    best_score_parts = score_parts0
    best_round = 0
    last_verifier_report = verifier_res0
    best_verifier_report = verifier_res0
    repair_cursor_completion = completion
    repair_cursor_res = res0
    repair_cursor_verifier_report = verifier_res0
    repair_cursor_score_parts = score_parts0


    last_serial_mismatch_info: Optional[Dict[str, Any]] = None
    best_serial_mismatch_info: Optional[Dict[str, Any]] = None
    repair_cursor_serial_mismatch_info: Optional[Dict[str, Any]] = None
    semantic_no_improve_streak = 0
    for r in range(1, rounds + 1):
        repair_source_action = None
        use_repair_cursor = False
        # Repair sequentially from the immediately previous completion.  The
        # historical best is still logged, but it no longer rewinds the repair
        # trajectory because the score can be misleading for logical/runtime bugs.
        completion = last_completion
        source_score_parts = last_score_parts
        source_res_for_repair = last_res
        # 0) Cheap deterministic patch from clang "did you mean": fix simple identifier typos
        #    without spending an LLM call.
        if last_res and (str(last_res.get("reason", "")).strip() == "compile_error"):
            patched_code, renames = apply_did_you_mean_renames(
                completion, str(last_res.get("compile_log_tail", ""))
            )
            if renames:
                patched_code = normalize_completion_snippet(patched_code, func_name)
                patched_code, _pp_info = postprocess_sve_common_fixes(patched_code)
                completion = patched_code

                verifier_res = _run_static_verifier(round_no=r, completion_text=completion)
                completion, verifier_res, _gate_res, gate_blocked = _apply_remote_cpp17_compile_gate(
                    round_no=r,
                    completion_text=completion,
                    verifier_report=verifier_res,
                )
                if gate_blocked:
                    res = _synthesize_gate_blocked_result(r, _gate_res)
                else:
                    res = _remote_eval(round_no=r)
                verifier_res = _promote_failed_untagged_report(verifier_res, res)
                verifier_res = _promote_remote_compile_diagnostics_report(verifier_res, res)
                last_res = res
                last_completion = completion
                last_verifier_report = verifier_res
                cur_score_parts = _score_parts(res, verifier_res, completion)
                last_score_parts = cur_score_parts
                best_updated = False
                if _is_score_better(cur_score_parts, best_score_parts):
                    best_res = res
                    best_completion = completion
                    best_round = r
                    best_score_parts = cur_score_parts
                    best_verifier_report = verifier_res
                    best_updated = True
                    semantic_no_improve_streak = 0

                current_action = _extract_controller_action(verifier_res)
                if best_updated:
                    repair_cursor_completion = last_completion
                    repair_cursor_res = last_res
                    repair_cursor_verifier_report = last_verifier_report
                    repair_cursor_score_parts = last_score_parts
                    repair_cursor_serial_mismatch_info = last_serial_mismatch_info

                info["history"].append(
                    {
                        "round": r,
                        "static_verifier": verifier_res,
                        "result": res,
                        "auto_patch": {"did_you_mean": renames, **(_pp_info or {})},
                        "composite_score": list(cur_score_parts.get("composite_score") or []),
                        "composite_score_parts": cur_score_parts,
                        "best_score_parts": best_score_parts,
                    }
                )

                if _remote_result_is_success(res):
                    info["final"] = res
                    info["last"] = last_res
                    info["best"] = best_res
                    info["best_round"] = best_round
                    info["last_score_parts"] = cur_score_parts
                    info["best_score_parts"] = best_score_parts
                    info["static_verifier"]["final"] = best_verifier_report
                    return completion, info
                continue

        serial_diff = ""
        serial_stage = ""
        serial_reason = ""
        serial_bootstrap_stage = ""
        serial_bootstrap_source = ""

        if serial_fallback and last_res and (serial_model is not None or serial_use_solution_scalar):
            serial_stage, serial_reason = _classify_serial_trigger(last_res, last_verifier_report)
            if serial_stage:
                serial_ref_triggered = True
                info["serial_ref"]["triggered"] = True
                if not serial_ref_ok:
                    _ensure_serial_reference(
                        stage=serial_stage,
                        reason=serial_reason,
                        round_hint=r,
                        trigger_res=last_res,
                    )
                serial_bootstrap_source = str(
                    info["serial_ref"].get("serial_bootstrap_source")
                    or info["serial_ref"].get("source")
                    or ""
                )
                if serial_stage:
                    serial_bootstrap_stage = serial_stage

        if (
            serial_ref_completion
            and serial_ref_ok
            and isinstance(last_res, dict)
            and int(last_res.get("compile_ok", 0) or 0) == 1
        ):
            a_code = serial_ref_completion
            b_code = completion

            try:
                if func_name:
                    a_body = try_extract_function_body(a_code, func_name)
                    b_body = try_extract_function_body(b_code, func_name)
                    if a_body:
                        a_code = a_body
                    if b_body:
                        b_code = b_body
            except Exception:
                pass

            serial_diff = make_unified_diff(
                a_code,
                b_code,
                fromfile="serial_ref",
                tofile="simd_current",
                context_lines=int(serial_diff_context_lines or 3),
                max_chars=int(serial_diff_max_chars or 4000),
            )

            if info.get("serial_ref") is not None:
                _serial_feedback_style_norm = serial_feedback_style_effective
                _diff_is_injected = bool(serial_diff and _serial_feedback_style_norm in {"code", "both"})
                info["serial_ref"]["diff_len"] = len(serial_diff or "")
                info["serial_ref"]["diff_available"] = bool(serial_diff)
                info["serial_ref"]["diff_injected"] = _diff_is_injected
                info["serial_ref"]["diff_excerpt"] = serial_diff if _diff_is_injected else ""
                info["serial_ref"]["feedback_style"] = _serial_feedback_style_norm
                info["serial_ref"]["feedback_style_requested"] = serial_feedback_style_requested
                info["serial_ref"]["feedback_style_effective"] = serial_feedback_style_effective

        serial_mismatch_info = last_serial_mismatch_info if serial_stage == "logic_fail" else None
        prompt_fb = build_remote_feedback_prompt(
            spec_text=spec_text,
            prompt_prefix=prompt_prefix,
            prev_completion=completion,
            remote_result=last_res,
            compile_only=(remote_eval_mode == "compile_only"),
            static_verifier_report=last_verifier_report,
            serial_ref_completion=(serial_ref_completion if serial_ref_ok else None),
            serial_bootstrap_completion=(serial_bootstrap_completion if serial_ref_ok else None),
            serial_ref_result=(serial_ref_result if serial_ref_ok else None),
            serial_diff=serial_diff,
            serial_mismatch_info=serial_mismatch_info,
            serial_code_max_chars=int(serial_prompt_max_chars or 0),
            serial_feedback_style=serial_feedback_style_effective,
            serial_feedback_style_requested=serial_feedback_style_requested,
            serial_pseudocode_max_chars=int(serial_pseudocode_max_chars or 0),
            serial_diff_max_chars=int(serial_diff_max_chars or 4000),
            serial_bootstrap_stage=serial_bootstrap_stage,
            serial_bootstrap_reason=serial_reason,
            serial_bootstrap_source=serial_bootstrap_source,
            serial_ast_bootstrap_pseudocode=str(serial_ast_bootstrap_pseudocode or ""),
            serial_ast_bootstrap_source=str(serial_ast_bootstrap_source or ""),
            compile_feedback_style=str(remote_compile_feedback_style or "structured"),
            disable_bootstrap_guard=bool(disable_bootstrap_guard),
        )


        if print_prompts:
            print("\n" + "=" * 120)
            print(f"[REMOTE_FEEDBACK_PROMPT] id={task_id} sample={sample_idx} round={r}")
            print("-" * 120)
            if prompt_max_chars and prompt_max_chars > 0 and len(prompt_fb) > prompt_max_chars:
                print(prompt_fb[:prompt_max_chars] + "\n...[TRUNCATED]...")
            else:
                print(prompt_fb)
            print("=" * 120 + "\n")

        _maybe_dump("remote_prompt", prompt_fb, r)
        remote_result_in_payload = dict(last_res or {})
        remote_result_in_payload["_used_as_feedback_for_round"] = r
        remote_result_in_payload["_feedback_source"] = "last"
        if isinstance(last_verifier_report, dict) and last_verifier_report.get("remote_compile_diagnostics"):
            remote_result_in_payload["remote_compile_diagnostics"] = last_verifier_report.get("remote_compile_diagnostics")
        _dump_json("remote_result_in", remote_result_in_payload, r)

        out, repair_generation = _run_patch_first_repair(
            base_prompt=prompt_fb,
            prev_completion=completion,
            dump_tag_prefix="remote_feedback_repair",
            dump_round=r,
            controller_action=None,
        )
        completion = normalize_completion_snippet(out, func_name)
        if intrinsic == "SVE":
            completion, _pp_info = postprocess_sve_common_fixes(completion)

        if reapply_name_shape and whitelist_set is not None and whitelist_list is not None and op_index is not None:
            invalid = sorted([c for c in extract_calls(completion, whitelist_set, context_text=spec_text) if c not in whitelist_set])
            if invalid and reapply_name_max_iters > 0:
                completion, _ = rag_fix_names(
                    model=model,
                    tok=tok,
                    completion_in=completion,
                    whitelist_set=whitelist_set,
                    whitelist_list=whitelist_list,
                    op_index=op_index,
                    spec_text=spec_text,
                    max_iters=reapply_name_max_iters,
                    max_new_tokens=max_new_tokens,
                    top_k=8,
                    cutoff=0.55,
                    do_sample=False,
                    temperature=_repair_temperature_for_backend(model, temperature),
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    sid=task_id,
                    rank=0,
                    func_name=func_name,
                    attempts_per_iter=2,
                    print_prompts=False,
                    dump_prompts_dir=None,
                )

            if sigs and reapply_shape_max_iters > 0:
                completion, _ = rag_fix_shapes(
                    model=model,
                    tok=tok,
                    completion_in=completion,
                    whitelist_set=whitelist_set,
                    whitelist_list=whitelist_list,
                    sigs=sigs,
                    rets=rets,
                    max_iters=reapply_shape_max_iters,
                    spec_text=spec_text,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=_repair_temperature_for_backend(model, temperature),
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    check_types=True,
                    sid=task_id,
                    rank=0,
                    func_name=func_name,
                    attempts_per_iter=2,
                    print_summary=False,
                    print_prompts=False,
                    dump_prompts_dir=None,
                )

        verifier_res = _run_static_verifier(round_no=r, completion_text=completion)
        completion, verifier_res, _gate_res, gate_blocked = _apply_remote_cpp17_compile_gate(
            round_no=r,
            completion_text=completion,
            verifier_report=verifier_res,
        )
        if gate_blocked:
            res = _synthesize_gate_blocked_result(r, _gate_res)
        else:
            res = _remote_eval(round_no=r)
        verifier_res = _promote_failed_untagged_report(verifier_res, res)
        verifier_res = _promote_remote_compile_diagnostics_report(verifier_res, res)
        last_res = res
        last_completion = completion
        last_verifier_report = verifier_res
        cur_score_parts = _score_parts(res, verifier_res, completion)
        last_score_parts = cur_score_parts
        best_updated = False
        if _is_score_better(cur_score_parts, best_score_parts):
            best_res = res
            best_completion = completion
            best_round = r
            best_score_parts = cur_score_parts
            best_verifier_report = verifier_res
            best_updated = True
            semantic_no_improve_streak = 0

        # After name/shape reapply + current remote eval, run mismatch harness on the
        # current completion (delayed to avoid invalid API usage).
        if (
            serial_mismatch_harness
            and serial_ref_ok
            and isinstance(res, dict)
            and int(res.get("compile_ok", 0) or 0) == 1
            and int(res.get("run_ok", 0) or 0) == 0
        ):
            if serial_model is None:
                last_serial_mismatch_info = {"mismatch": True, "index": -1, "error": "serial_model_missing"}
            else:
                mm_round = 9200 + r * 10
                mm_prompt = build_serial_mismatch_harness_prompt(
                    spec_text=spec_text,
                    prompt_prefix=prompt_prefix,
                    func_name=func_name,
                    serial_ref_completion=serial_ref_completion or "",
                    simd_completion=completion or "",
                )
                if serial_mismatch_prompt_max_chars and serial_mismatch_prompt_max_chars > 0:
                    mm_prompt = _truncate_middle(mm_prompt, int(serial_mismatch_prompt_max_chars))
                _dump_text("serial_mismatch_prompt", mm_prompt, mm_round)

                _serial_gen_backend = (
                    "api" if serial_llm_backend in ("openai", "deepseek", "api") else "hf"
                )
                mm_out = generate_text(
                    serial_model,
                    serial_tok,
                    user_text=mm_prompt,
                    system_text=serial_mismatch_system_prompt,
                    max_new_tokens=int(serial_mismatch_max_new_tokens),
                    do_sample=serial_do_sample,
                    temperature=serial_temperature,
                    top_p=serial_top_p,
                    repetition_penalty=serial_repetition_penalty,
                    backend=_serial_gen_backend,
                    api_model=serial_api_model,
                    is_t0=(mm_round == 0),
                )
                mm_code = strip_noncode(mm_out)
                _dump_text("serial_mismatch_code", mm_code, mm_round)
                if not (mm_code or "").strip():
                    mm_round_retry = mm_round + 1
                    mm_prompt2 = build_serial_mismatch_harness_prompt_minimal(
                        func_name=func_name,
                        serial_ref_completion=serial_ref_completion or "",
                        simd_completion=completion or "",
                    )
                    if serial_mismatch_prompt_max_chars and serial_mismatch_prompt_max_chars > 0:
                        mm_prompt2 = _truncate_middle(mm_prompt2, int(serial_mismatch_prompt_max_chars))
                    _dump_text("serial_mismatch_prompt_retry", mm_prompt2, mm_round_retry)
                    mm_out2 = generate_text(
                        serial_model,
                        serial_tok,
                        user_text=mm_prompt2,
                        system_text=serial_mismatch_system_prompt,
                        max_new_tokens=int(serial_mismatch_max_new_tokens),
                        do_sample=serial_do_sample,
                        temperature=serial_temperature,
                        top_p=serial_top_p,
                        repetition_penalty=serial_repetition_penalty,
                        backend=_serial_gen_backend,
                        api_model=serial_api_model,
                        is_t0=False,
                    )
                    mm_code = strip_noncode(mm_out2)
                    _dump_text("serial_mismatch_code_retry", mm_code, mm_round_retry)
                    if not (mm_code or "").strip():
                        auto_code = try_build_auto_mismatch_harness(
                            func_name=func_name,
                            serial_ref_completion=serial_ref_completion or "",
                            simd_completion=completion or "",
                        )
                        if auto_code:
                            mm_round_auto = mm_round_retry + 1
                            _dump_text("serial_mismatch_code_auto", auto_code, mm_round_auto)
                            mm_res = _remote_eval_harness(mm_round_auto, auto_code)
                            _dump_json("serial_mismatch_result", mm_res, mm_round_auto)
                            last_serial_mismatch_info = mm_res
                        else:
                            last_serial_mismatch_info = {"mismatch": True, "index": -1, "error": "empty_harness_output"}
                    elif re.search(r"\bmain\s*\(", mm_code) is None:
                        auto_code = try_build_auto_mismatch_harness(
                            func_name=func_name,
                            serial_ref_completion=serial_ref_completion or "",
                            simd_completion=completion or "",
                        )
                        if auto_code:
                            mm_round_auto = mm_round_retry + 1
                            _dump_text("serial_mismatch_code_auto", auto_code, mm_round_auto)
                            mm_res = _remote_eval_harness(mm_round_auto, auto_code)
                            _dump_json("serial_mismatch_result", mm_res, mm_round_auto)
                            last_serial_mismatch_info = mm_res
                        else:
                            last_serial_mismatch_info = {"mismatch": True, "index": -1, "error": "no_main_in_harness"}
                    else:
                        mm_res = _remote_eval_harness(mm_round_retry, mm_code)
                        _dump_json("serial_mismatch_result", mm_res, mm_round_retry)
                        last_serial_mismatch_info = mm_res
                elif re.search(r"\bmain\s*\(", mm_code) is None:
                    auto_code = try_build_auto_mismatch_harness(
                        func_name=func_name,
                        serial_ref_completion=serial_ref_completion or "",
                        simd_completion=completion or "",
                    )
                    if auto_code:
                        mm_round_auto = mm_round + 1
                        _dump_text("serial_mismatch_code_auto", auto_code, mm_round_auto)
                        mm_res = _remote_eval_harness(mm_round_auto, auto_code)
                        _dump_json("serial_mismatch_result", mm_res, mm_round_auto)
                        last_serial_mismatch_info = mm_res
                    else:
                        last_serial_mismatch_info = {"mismatch": True, "index": -1, "error": "no_main_in_harness"}
                else:
                    mm_res = _remote_eval_harness(mm_round, mm_code)
                    _dump_json("serial_mismatch_result", mm_res, mm_round)
                    last_serial_mismatch_info = mm_res

            if info.get("serial_ref") is not None:
                info["serial_ref"]["mismatch"] = last_serial_mismatch_info
            if best_updated:
                best_serial_mismatch_info = last_serial_mismatch_info

        current_action = _extract_controller_action(verifier_res)
        semantic_route = False
        if best_updated:
            repair_cursor_completion = last_completion
            repair_cursor_res = last_res
            repair_cursor_verifier_report = last_verifier_report
            repair_cursor_score_parts = last_score_parts
            repair_cursor_serial_mismatch_info = last_serial_mismatch_info

        semantic_progress = False
        if semantic_route and not best_updated:
            semantic_progress = _has_semantic_progress(source_score_parts, cur_score_parts)
            if semantic_progress:
                semantic_no_improve_streak = 0
        early_stop_suppression_reason = ""
        if not best_updated:
            early_stop_suppression_reason = _no_improve_early_stop_suppression_reason(source_res_for_repair, res)

        info["history"].append(
            {
                "round": r,
                "static_verifier": verifier_res,
                "result": res,
                "composite_score": list(cur_score_parts.get("composite_score") or []),
                "composite_score_parts": cur_score_parts,
                "best_score_parts": best_score_parts,
                "semantic_progress": bool(semantic_progress),
                "repair_source_action": repair_source_action,
                "controller_action": current_action,
                "repair_source": "last_completion",
                "repair_generation": repair_generation,
                "early_stop_suppression_reason": early_stop_suppression_reason,
            }
        )

        if (not best_updated) and remote_early_stop_no_improve:
            allow_continue = bool(early_stop_suppression_reason)
            patience_limit = int(max(0, remote_semantic_no_improve_patience or 0))
            if semantic_route:
                if semantic_progress:
                    allow_continue = True
                elif semantic_no_improve_streak < patience_limit:
                    semantic_no_improve_streak += 1
                    allow_continue = True
            if allow_continue and early_stop_suppression_reason:
                info["early_stop_no_improve"] = {
                    "enabled": True,
                    "triggered": False,
                    "suppressed": True,
                    "suppression_reason": early_stop_suppression_reason,
                    "round": int(r),
                    "best_round": int(best_round),
                    "current_score": list(cur_score_parts.get("composite_score") or []),
                    "best_score": list(best_score_parts.get("composite_score") or []),
                    "current_reason": str((res or {}).get("reason") or ""),
                    "best_reason": str((best_res or {}).get("reason") or ""),
                }
            if not allow_continue:
                info["early_stop_no_improve"] = {
                    "enabled": True,
                    "triggered": True,
                    "round": int(r),
                    "best_round": int(best_round),
                    "current_score": list(cur_score_parts.get("composite_score") or []),
                    "best_score": list(best_score_parts.get("composite_score") or []),
                    "current_score_parts": cur_score_parts,
                    "best_score_parts": best_score_parts,
                    "semantic_route": bool(semantic_route),
                    "semantic_progress": bool(semantic_progress),
                    "semantic_no_improve_streak": int(semantic_no_improve_streak),
                    "semantic_no_improve_patience": int(max(0, remote_semantic_no_improve_patience or 0)),
                }
                if print_summary:
                    print("=" * 120)
                    print(
                        f"[REMOTE_EVAL] id={task_id} sample={sample_idx} round={r} stop_reason=no_improve "
                        f"score={cur_score_parts.get('composite_score')} "
                        f"best_score={best_score_parts.get('composite_score')} "
                        f"semantic_route={semantic_route} semantic_progress={semantic_progress} "
                        f"semantic_patience={semantic_no_improve_streak}/{max(0, remote_semantic_no_improve_patience or 0)}"
                    )
                    print("=" * 120)
                break

        if print_summary:
            print("=" * 120)
            print(f"[REMOTE_EVAL] id={task_id} sample={sample_idx} round={r} compile_ok={res.get('compile_ok')} run_ok={res.get('run_ok')} reason={res.get('reason')}")
            print("=" * 120)

        if _remote_result_is_success(res):
            info["final"] = res
            break

    _post_stage = ""
    _post_reason = ""
    _trigger_res: Optional[Dict[str, Any]] = None
    if (
        serial_fallback
        and (serial_ref_ok is False)
        and (serial_model is not None or serial_use_solution_scalar)
        and (
            (not serial_ref_attempted)
            or str(info["serial_ref"].get("serial_ref_failure_kind") or "none") == "infra"
        )
    ):
        if isinstance(last_res, dict):
            _post_stage, _post_reason = _classify_serial_trigger(last_res, last_verifier_report)
            if _post_stage:
                _trigger_res = last_res

        if _post_stage and _trigger_res is not None:
            _post_r = int(len(info.get("history") or [])) + 1
            serial_ref_triggered = True
            info["serial_ref"]["triggered"] = True
            _ensure_serial_reference(
                stage=_post_stage,
                reason=_post_reason,
                round_hint=_post_r,
                trigger_res=_trigger_res,
            )

    if (
        serial_ref_ok
        and _post_stage == "logic_fail"
        and isinstance(last_res, dict)
        and int(last_res.get("compile_ok", 0) or 0) == 1
        and int(last_res.get("run_ok", 0) or 0) == 0
    ):
        extra_round = int(len(info.get("history") or [])) + 50
        completion = last_completion

        serial_diff = ""
        if serial_ref_completion:
            a_code = serial_ref_completion
            b_code = completion
            try:
                if func_name:
                    a_body = try_extract_function_body(a_code, func_name)
                    b_body = try_extract_function_body(b_code, func_name)
                    if a_body:
                        a_code = a_body
                    if b_body:
                        b_code = b_body
            except Exception:
                pass
            serial_diff = make_unified_diff(
                a_code,
                b_code,
                fromfile="serial_ref",
                tofile="simd_current",
                context_lines=int(serial_diff_context_lines or 3),
                max_chars=int(serial_diff_max_chars or 4000),
            )

        prompt_fb = build_remote_feedback_prompt(
            spec_text=spec_text,
            prompt_prefix=prompt_prefix,
            prev_completion=completion,
            remote_result=last_res,
            compile_only=(remote_eval_mode == "compile_only"),
            static_verifier_report=last_verifier_report,
            serial_ref_completion=(serial_ref_completion if serial_ref_ok else None),
            serial_bootstrap_completion=(serial_bootstrap_completion if serial_ref_ok else None),
            serial_ref_result=(serial_ref_result if serial_ref_ok else None),
            serial_diff=serial_diff,
            serial_mismatch_info=None,
            serial_code_max_chars=int(serial_prompt_max_chars or 0),
            serial_feedback_style=serial_feedback_style_effective,
            serial_feedback_style_requested=serial_feedback_style_requested,
            serial_pseudocode_max_chars=int(serial_pseudocode_max_chars or 0),
            serial_diff_max_chars=int(serial_diff_max_chars or 4000),
            serial_bootstrap_stage=_post_stage,
            serial_bootstrap_reason=_post_reason,
            serial_bootstrap_source=str(
                info["serial_ref"].get("serial_bootstrap_source")
                or info["serial_ref"].get("source")
                or ""
            ),
            serial_ast_bootstrap_pseudocode=str(serial_ast_bootstrap_pseudocode or ""),
            serial_ast_bootstrap_source=str(serial_ast_bootstrap_source or ""),
            compile_feedback_style=str(remote_compile_feedback_style or "structured"),
            disable_bootstrap_guard=bool(disable_bootstrap_guard),
        )

        _maybe_dump("remote_prompt_post_loop", prompt_fb, extra_round)
        remote_result_in_payload = dict(last_res or {})
        remote_result_in_payload["_used_as_feedback_for_round"] = extra_round
        if isinstance(last_verifier_report, dict) and last_verifier_report.get("remote_compile_diagnostics"):
            remote_result_in_payload["remote_compile_diagnostics"] = last_verifier_report.get("remote_compile_diagnostics")
        remote_result_in_payload["_feedback_source"] = "post_loop_last"
        _dump_json("remote_result_in", remote_result_in_payload, extra_round)

        out, repair_generation = _run_patch_first_repair(
            base_prompt=prompt_fb,
            prev_completion=completion,
            dump_tag_prefix="remote_feedback_post_loop_repair",
            dump_round=extra_round,
            controller_action=None,
        )
        completion = normalize_completion_snippet(out, func_name)
        if intrinsic == "SVE":
            completion, _pp_info = postprocess_sve_common_fixes(completion)

        if reapply_name_shape and whitelist_set is not None and whitelist_list is not None and op_index is not None:
            invalid = sorted([c for c in extract_calls(completion, whitelist_set, context_text=spec_text) if c not in whitelist_set])
            if invalid and reapply_name_max_iters > 0:
                completion, _ = rag_fix_names(
                    model=model,
                    tok=tok,
                    completion_in=completion,
                    whitelist_set=whitelist_set,
                    whitelist_list=whitelist_list,
                    op_index=op_index,
                    spec_text=spec_text,
                    max_iters=reapply_name_max_iters,
                    max_new_tokens=max_new_tokens,
                    top_k=8,
                    cutoff=0.55,
                    do_sample=False,
                    temperature=_repair_temperature_for_backend(model, temperature),
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    sid=task_id,
                    rank=0,
                    func_name=func_name,
                    attempts_per_iter=2,
                    print_prompts=False,
                    dump_prompts_dir=None,
                )

            if sigs and reapply_shape_max_iters > 0:
                completion, _ = rag_fix_shapes(
                    model=model,
                    tok=tok,
                    completion_in=completion,
                    whitelist_set=whitelist_set,
                    whitelist_list=whitelist_list,
                    sigs=sigs,
                    rets=rets,
                    max_iters=reapply_shape_max_iters,
                    spec_text=spec_text,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=_repair_temperature_for_backend(model, temperature),
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    check_types=True,
                    sid=task_id,
                    rank=0,
                    func_name=func_name,
                    attempts_per_iter=2,
                    print_summary=False,
                    print_prompts=False,
                    dump_prompts_dir=None,
                )

        verifier_res = _run_static_verifier(round_no=extra_round, completion_text=completion)
        res = _remote_eval(round_no=extra_round)
        verifier_res = _promote_failed_untagged_report(verifier_res, res)
        verifier_res = _promote_remote_compile_diagnostics_report(verifier_res, res)
        info["history"].append(
            {
                "round": extra_round,
                "static_verifier": verifier_res,
                "result": res,
                "note": "post_loop_dataset_solution_extra",
                "repair_generation": repair_generation,
                "composite_score": list(_score_parts(res, verifier_res, completion).get("composite_score") or []),
            }
        )
        last_res = res
        last_completion = completion
        last_verifier_report = verifier_res
        cur_score_parts = _score_parts(res, verifier_res, completion)
        last_score_parts = cur_score_parts
        if _is_score_better(cur_score_parts, best_score_parts):
            best_res = res
            best_completion = completion
            best_round = extra_round
            best_score_parts = cur_score_parts
            best_verifier_report = verifier_res
        info["history"][-1]["composite_score_parts"] = cur_score_parts
        info["history"][-1]["best_score_parts"] = best_score_parts
        repair_cursor_completion = last_completion
        repair_cursor_res = last_res
        repair_cursor_verifier_report = last_verifier_report
        repair_cursor_score_parts = last_score_parts
        repair_cursor_serial_mismatch_info = last_serial_mismatch_info
    info["last"] = last_res
    info["best"] = best_res
    info["best_round"] = best_round
    info["final"] = last_res
    info["last_score_parts"] = last_score_parts
    info["best_score_parts"] = best_score_parts
    info["static_verifier"]["final"] = last_verifier_report
    return last_completion, info


# =============================================================================
# Model load (HF backend)
# =============================================================================

def load_model_and_tokenizer(
    *,
    model_path: str,
    adapter_path: str,
    device_map,
    torch_dtype,
    trust_remote_code: bool,
    local_files_only: bool,
    merge_adapter: bool,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "bf16",
    bnb_4bit_use_double_quant: bool = False,
) -> Tuple[object, object]:
    def _flatten_no_split_modules(value) -> List[str]:
        out: List[str] = []
        seen = set()

        def _visit(v):
            if v is None:
                return
            if isinstance(v, str):
                if v not in seen:
                    seen.add(v)
                    out.append(v)
                return
            if isinstance(v, (list, tuple, set)):
                for item in v:
                    _visit(item)

        _visit(value)
        return out

    def _normalize_no_split_modules_tree(root) -> None:
        for module in root.modules():
            if hasattr(module, "_no_split_modules"):
                normalized = _flatten_no_split_modules(getattr(module, "_no_split_modules", None))
                module._no_split_modules = normalized or None

    if not _HAVE_TRANSFORMERS:
        raise SystemExit(
            "transformers is required for HF backend but could not be imported. "
            f"Import error: {_TRANSFORMERS_IMPORT_ERR}"
        )
    if load_in_4bit or load_in_8bit:
        try:
            import bitsandbytes  # type: ignore  # noqa: F401
        except Exception as _e:
            raise SystemExit(
                "You requested quantized loading (--load_in_4bit/--load_in_8bit) but bitsandbytes is not available. "
                f"Import error: {repr(_e)}"
            )
    tok = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
        padding_side="left",
        local_files_only=local_files_only,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    quant_cfg = None
    if load_in_4bit and load_in_8bit:
        raise ValueError("Do not set both load_in_4bit and load_in_8bit.")

    if load_in_4bit:
        compute_dtype = torch.bfloat16 if bnb_4bit_compute_dtype == "bf16" else torch.float16
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        )
    elif load_in_8bit:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        quantization_config=quant_cfg,
        torch_dtype=torch_dtype if quant_cfg is None else None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if adapter_path:
        if not PEFT_AVAILABLE:
            raise RuntimeError("adapter_path provided but peft is not installed.")
        _normalize_no_split_modules_tree(model)
        peft_load_kwargs = {"local_files_only": local_files_only}
        hf_device_map = getattr(model, "hf_device_map", None)
        if isinstance(hf_device_map, dict) and set(hf_device_map.values()).intersection({"cpu", "disk"}):
            # PEFT + accelerate can choke on some Qwen3.5 offload layouts while re-balancing adapters.
            # Sequential avoids the failing balanced-memory path and keeps adapter loading deterministic.
            peft_load_kwargs["device_map"] = "sequential"
        model = PeftModel.from_pretrained(model, adapter_path, **peft_load_kwargs)

        if merge_adapter:
            if quant_cfg is not None:
                print("[WARN] merge_adapter requested but base model is 4bit/8bit quantized; skip merge.")
            else:
                try:
                    model = model.merge_and_unload()
                except Exception as e:
                    print("[WARN] merge_and_unload failed, continue without merge:", e)

    return model, tok

def shard_round_robin(items: List[dict], num_shards: int, shard_id: int) -> List[dict]:
    return [x for i, x in enumerate(items) if (i % num_shards) == shard_id]


def _work_queue_base_path(base_out: Path) -> Path:
    stem = base_out.stem
    digest = hashlib.sha1(str(base_out).encode("utf-8", errors="replace")).hexdigest()[:12]
    short_stem = stem if len(stem) <= 120 else f"{stem[:120]}.{digest}"
    return base_out.with_name(f"{short_stem}.dynamic_work_queue")


def _work_queue_paths(base_out: Path) -> Tuple[Path, Path, Path]:
    base = _work_queue_base_path(base_out)
    return (
        base.with_suffix(".state.json"),
        base.with_suffix(".lock"),
        base.with_suffix(".ready"),
    )


def _problem_order_fingerprint(problems: List[dict], *, n_samples: int, seed: int) -> str:
    h = hashlib.sha1()
    h.update(f"n_samples={n_samples}\nseed={seed}\ncount={len(problems)}\n".encode("utf-8"))
    for idx, p in enumerate(problems):
        h.update(f"{idx}\t{p.get('task_id', '')}\n".encode("utf-8", errors="replace"))
    return h.hexdigest()


def _read_json_file(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _write_json_file_atomic(path: Path, payload: Dict) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _with_work_queue_lock(lock_path: Path):
    """Small fcntl-based inter-process lock for ranks sharing one filesystem."""
    class _LockCtx:
        def __enter__(self):
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = lock_path.open("a+", encoding="utf-8")
            if fcntl is not None:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            return self._fh

        def __exit__(self, exc_type, exc, tb):
            try:
                if fcntl is not None:
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()

    return _LockCtx()


def init_dynamic_work_queue(
    *,
    base_out: Path,
    problems: List[dict],
    n_samples: int,
    seed: int,
    rank: int,
    world_size: int,
    timeout_s: int,
) -> Tuple[Path, Path, Path, str]:
    state_path, lock_path, ready_path = _work_queue_paths(base_out)
    fingerprint = _problem_order_fingerprint(problems, n_samples=n_samples, seed=seed)
    tasks = [
        {
            "index": idx,
            "task_id": str(p.get("task_id", "")),
            "key": f"{idx:08d}:{p.get('task_id', '')}",
        }
        for idx, p in enumerate(problems)
    ]

    if rank == 0:
        with _with_work_queue_lock(lock_path):
            state = {
                "version": 1,
                "mode": "dynamic_work_stealing_task_level",
                "fingerprint": fingerprint,
                "world_size": world_size,
                "n_samples": n_samples,
                "seed": seed,
                "created_at": time.time(),
                "cursor": 0,
                "tasks": tasks,
                "claimed": {},
                "completed": {},
            }
            _write_json_file_atomic(state_path, state)
            _atomic_write_text(ready_path, json.dumps({"fingerprint": fingerprint, "ts": time.time()}) + "\n")
    else:
        t0 = time.time()
        while True:
            state = _read_json_file(state_path)
            if state.get("fingerprint") == fingerprint and ready_path.exists():
                break
            if timeout_s > 0 and (time.time() - t0) > timeout_s:
                raise RuntimeError(
                    f"timed out waiting for dynamic work queue init: {state_path} "
                    f"(rank={rank}, expected_fingerprint={fingerprint})"
                )
            time.sleep(0.25)

    return state_path, lock_path, ready_path, fingerprint


def claim_dynamic_work_item(
    *,
    state_path: Path,
    lock_path: Path,
    fingerprint: str,
    rank: int,
) -> Optional[Dict]:
    with _with_work_queue_lock(lock_path):
        state = _read_json_file(state_path)
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"dynamic work queue fingerprint mismatch: {state_path} "
                f"got={state.get('fingerprint')} expected={fingerprint}"
            )
        tasks = list(state.get("tasks") or [])
        claimed = dict(state.get("claimed") or {})
        completed = dict(state.get("completed") or {})
        cursor = int(state.get("cursor") or 0)

        while cursor < len(tasks):
            item = tasks[cursor]
            cursor += 1
            key = str(item.get("key") or "")
            if not key or key in completed:
                continue
            claimed[key] = {
                "rank": rank,
                "pid": os.getpid(),
                "claimed_at": time.time(),
                "cursor": cursor,
            }
            state["cursor"] = cursor
            state["claimed"] = claimed
            state["completed"] = completed
            _write_json_file_atomic(state_path, state)
            out = dict(item)
            out["queue_pos"] = cursor
            out["queue_total"] = len(tasks)
            return out

        state["cursor"] = cursor
        state["claimed"] = claimed
        state["completed"] = completed
        _write_json_file_atomic(state_path, state)
        return None


def mark_dynamic_work_item_done(
    *,
    state_path: Path,
    lock_path: Path,
    fingerprint: str,
    item_key: str,
    rank: int,
    ok: bool,
    samples_written: int,
) -> None:
    if not item_key:
        return
    with _with_work_queue_lock(lock_path):
        state = _read_json_file(state_path)
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"dynamic work queue fingerprint mismatch while marking done: {state_path}"
            )
        claimed = dict(state.get("claimed") or {})
        completed = dict(state.get("completed") or {})
        claimed.pop(item_key, None)
        completed[item_key] = {
            "rank": rank,
            "pid": os.getpid(),
            "ok": bool(ok),
            "samples_written": int(samples_written),
            "done_at": time.time(),
        }
        state["claimed"] = claimed
        state["completed"] = completed
        _write_json_file_atomic(state_path, state)
# =============================================================================
# main
# =============================================================================


_TASK_BLOCK_RE = re.compile(r"/\*([\s\S]*?)\*/", re.MULTILINE)

def extract_task_text_from_prompt(raw_prompt: str, *, max_chars: int = 2000) -> str:
    """
    Extract a human-readable task description from the first non-empty /* ... */ block.
    Conservative: only used when p["task"] is empty or junk (e.g., 'generation').
    """
    if not raw_prompt:
        return ""
    m_all = list(_TASK_BLOCK_RE.finditer(raw_prompt))
    if not m_all:
        return ""
    # pick the first block that looks like a real task (has letters and maybe 'Given'/'Return' etc.)
    for m in m_all:
        blk = (m.group(1) or "").strip()
        if not blk:
            continue
        # strip leading '*' decorations
        lines = []
        for ln in blk.splitlines():
            ln2 = re.sub(r"^\s*\*\s?", "", ln.rstrip())
            lines.append(ln2)
        blk2 = "\n".join(lines).strip()

        # skip obvious non-task blocks
        low = blk2.lower()
        if "intrinsics_rules" in low or "expanded_spec" in low:
            continue
        if len(re.findall(r"[A-Za-z]", blk2)) < 20:
            continue

        if max_chars > 0 and len(blk2) > max_chars:
            blk2 = blk2[:max_chars].rstrip() + "\n...[TRUNCATED]..."
        return blk2
    return ""

def main() -> None:
    ap = argparse.ArgumentParser()

    # -------------------------------------------------------------------------
    # NEW: inference backend selector
    # -------------------------------------------------------------------------
    ap.add_argument(
        "--llm_backend",
        type=str,
        default="hf",
        choices=["hf", "openai", "deepseek"],
        help="Inference backend: hf=local transformers; openai=OpenAI API; deepseek=DeepSeek API",
    )
    ap.add_argument("--api_base_url", type=str, default="", help="API base URL (defaults depend on --llm_backend)")
    ap.add_argument("--api_key", type=str, default="", help="API key (or use env var OPENAI_API_KEY / DEEPSEEK_API_KEY)")
    ap.add_argument("--api_model", type=str, default="", help="API model name (default depends on --llm_backend)")
    ap.add_argument(
        "--api_endpoint",
        type=str,
        default="responses",
        choices=["responses", "chat_completions"],
        help="(openai only) Which endpoint style to use. responses=POST /v1/responses; chat_completions=POST /v1/chat/completions",
    )
    ap.add_argument("--api_timeout", type=int, default=60, help="API HTTP timeout seconds")
    ap.add_argument("--api_max_retries", type=int, default=8, help="API retry count on transient errors (429/5xx)")
    ap.add_argument("--api_retry_backoff", type=float, default=1.6, help="API retry backoff base seconds")
    ap.add_argument("--api_retry_max_sleep", type=float, default=20.0, help="API retry max sleep seconds")
    ap.add_argument("--api_extra_headers_json", type=str, default="", help="Extra HTTP headers as JSON object string")
    ap.add_argument("--api_extra_body_json", type=str, default="", help="Extra request-body fields as JSON object string (merged into payload)")
    ap.add_argument("--api_prompt_max_chars", type=int, default=0, help="If >0, truncate system/user prompt to this many chars (middle truncation)")
    ap.add_argument("--api_print_requests", action="store_true", help="Print API request metadata (no secrets)")

    # (torchrun) rank0 merge wait config (done-marker based merge)
    ap.add_argument(
        "--final_merge_timeout_s",
        type=int,
        default=0,
        help="Rank0 waits for all ranks' done markers before merging. 0=wait forever.",
    )
    ap.add_argument(
        "--final_merge_poll_s",
        type=float,
        default=1.0,
        help="Polling interval (seconds) while waiting for done markers.",
    )
    ap.add_argument(
        "--final_merge_print_every_s",
        type=float,
        default=30.0,
        help="Print missing ranks every N seconds while waiting.",
    )

    # HF backend args (required only when --llm_backend=hf)
    ap.add_argument("--model_path", "--model_base", dest="model_path", default="", help="Local HF model directory (required for --llm_backend=hf)")
    ap.add_argument("--adapter_path", "--model_adapter", dest="adapter_path", default="", help="Optional local LoRA/PEFT adapter dir")
    ap.add_argument("--merge_adapter", action="store_true", help="Merge adapter weights into base model (recommended if supported)")

    ap.add_argument("--problem_file", required=True, help="Problem jsonl (task_id + prompt required)")
    ap.add_argument("--intrinsic", default="SVE", choices=["SSE", "AVX", "SVE", "Neon", "RVV"])
    ap.add_argument("--output", required=True, help="Output samples jsonl (base path)")
    ap.add_argument(
        "--completion_mode",
        choices=[COMPLETION_MODE_SNIPPET, COMPLETION_MODE_FULL],
        default=COMPLETION_MODE_FULL,
        help=(
            "Completion compilation mode. 'snippet' emits only the function body which must be appended to the prompt prefix; "
            "'full' emits a standalone C/C++ translation unit (includes + full function definition) that is compiled as-is."
        ),
    )
    ap.add_argument("--n_samples", type=int, default=1, help="Samples per task (for pass@k)")

    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--do_sample", action="store_true", help="Force sampling on (else inferred from temperature>0)")

    # Repair params (used for name_fix / shape_fix / remote_feedback; keep lower than codegen for stability)
    ap.add_argument("--repair_temperature", type=float, default=0.8)
    ap.add_argument("--repair_top_p", type=float, default=0.95)
    ap.add_argument("--repair_repetition_penalty", type=float, default=1.1)

    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--num_shards", type=int, default=1, help="Manual data parallel: total shards (ignored under torchrun)")
    ap.add_argument("--shard_id", type=int, default=0, help="Manual data parallel: this shard id (ignored under torchrun)")
    ap.add_argument("--keep_shard_files", action="store_true", help="Keep per-shard output files (torchrun merge mode)")
    ap.add_argument(
        "--dynamic_work_stealing",
        dest="dynamic_work_stealing",
        action="store_true",
        default=True,
        help=(
            "Use a shared task queue across ranks so fast ranks keep claiming remaining full-task "
            "work units instead of idling after their static shard is done. Enabled by default "
            "when num_shards/world_size > 1."
        ),
    )
    ap.add_argument(
        "--no_dynamic_work_stealing",
        dest="dynamic_work_stealing",
        action="store_false",
        help="Disable shared-queue scheduling and restore static round-robin sharding.",
    )
    ap.add_argument(
        "--dynamic_work_queue_init_timeout_s",
        type=int,
        default=600,
        help="Seconds nonzero ranks wait for rank0 to initialize the dynamic work queue.",
    )

    # QLoRA / k-bit inference (HF backend only)
    ap.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit (bitsandbytes).")
    ap.add_argument("--load_in_8bit", action="store_true", help="Load base model in 8-bit (bitsandbytes).")
    ap.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--bnb_4bit_compute_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--bnb_4bit_use_double_quant", action="store_true")

    ap.add_argument("--bf16", action="store_true", help="Use bfloat16 weights (default)")
    ap.add_argument("--fp16", action="store_true", help="Use float16 weights")
    ap.add_argument("--trust_remote_code", action="store_true", default=True, help="(default: True)")
    ap.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false", help="Disable trust_remote_code")
    ap.add_argument("--local_files_only", action="store_true", default=True, help="(default: True) HF local_files_only=True")
    ap.add_argument("--no_local_files_only", dest="local_files_only", action="store_false", help="Disable local_files_only")

    # Repair options
    ap.add_argument("--whitelist", type=str, default="", help="Path to whitelist.json (SVE names + optional sigs)")
    ap.add_argument("--name_fix_max_iters", type=int, default=3)
    ap.add_argument("--name_fix_max_new_tokens", type=int, default=1024)
    ap.add_argument("--name_fix_top_k", type=int, default=8)
    ap.add_argument("--name_fix_cutoff", type=float, default=0.55)
    ap.add_argument("--name_fix_print_prompts", action="store_true")
    ap.add_argument("--name_fix_dump_prompts_dir", type=str, default="")
    ap.add_argument("--name_fix_prompt_max_chars", type=int, default=0)
    ap.add_argument("--name_fix_attempts_per_iter", type=int, default=3, help="Repair sampling attempts per name-fix iteration (best-of-N)")

    ap.add_argument("--shape_fix_max_iters", type=int, default=2)
    ap.add_argument("--shape_fix_max_new_tokens", type=int, default=1024)
    ap.add_argument("--shape_fix_check_types", action="store_true")
    ap.add_argument("--shape_fix_print_summary", action="store_true")
    ap.add_argument("--shape_fix_print_prompts", action="store_true")
    ap.add_argument("--shape_fix_dump_prompts_dir", type=str, default="")
    ap.add_argument("--shape_fix_prompt_max_chars", type=int, default=0)
    ap.add_argument("--shape_fix_attempts_per_iter", type=int, default=3, help="Repair sampling attempts per shape-fix iteration (best-of-N)")

    ap.add_argument("--append_intrinsics_rules", action="store_true", help="Append a strict rule block to the USER prompt")

    ap.add_argument("--save_intermediate", action="store_true", help="Save per-task artifacts")
    ap.add_argument("--work_dir", type=str, default="", help="Base dir for intermediates/logs (default: <output>.work/)")
    ap.add_argument("--log_file", type=str, default="", help="Write per-sample meta log jsonl to this path")

    # Remote feedback loop
    ap.add_argument("--remote_feedback_rounds", type=int, default=0, help=">0 to enable per-sample remote feedback loop via SSH")
    ap.add_argument(
        "--remote_early_stop_no_improve",
        action="store_true",
        help="Stop remote feedback early when a round does not improve the best remote score.",
    )
    ap.add_argument(
        "--remote_score_mode",
        type=str,
        default="legacy",
        choices=["legacy", "composite"],
        help="Remote round/sample scoring mode: legacy=(run_ok,compile_ok), composite=verifier-aware composite score.",
    )
    ap.add_argument(
        "--remote_repair_cursor_mode",
        type=str,
        default="latest",
        choices=["latest", "best", "dual"],
        help="Remote repair cursor mode flag retained for compatibility; repair now proceeds from the latest completion.",
    )
    ap.add_argument(
        "--remote_semantic_no_improve_patience",
        type=int,
        default=0,
        help="Additional patience for semantic repair path when composite score main tiers do not improve but semantic risk may still be improving.",
    )
    ap.add_argument("--remote_user", type=str, default="")
    ap.add_argument("--remote_host", type=str, default="")
    ap.add_argument("--remote_port", type=int, default=22)
    ap.add_argument("--remote_ssh_key", type=str, default="")
    ap.add_argument("--remote_no_strict_hostkey", action="store_true")
    ap.add_argument("--remote_tmp_root", type=str, default="~/simdbench_remote_tmp", help="Remote dir root to store uploaded files")

    # New default: simdbench_one (calls remote helper script to run evaluate_functional_correctness on ONE sample)
    ap.add_argument("--remote_eval_mode", type=str, default="simdbench_one",
                    choices=["compile_only", "cmd", "simdbench_one"],
                    help="compile_only: remote clang++ -c prompt+completion; "
                         "cmd: run user-provided remote command template that prints JSON; "
                         "simdbench_one: call remote helper script simdbench_remote_eval_one.py on uploaded one-line sample.jsonl")

    # cmd mode
    ap.add_argument("--remote_eval_cmd", type=str, default="",
                    help="(remote_eval_mode=cmd) remote command template with placeholders like "
                         "{remote_sample_file},{task_id},{intrinsic},{timeout},{remote_dir},... and prints JSON as last line")

    # simdbench_one mode
    ap.add_argument("--remote_simdbench_eval", type=str, default="python3 ~/simdbench_remote_eval_one.py",
                    help="(remote_eval_mode=simdbench_one) remote evaluator command (path to simdbench_remote_eval_one.py). "
                         "Must print JSON as last line. Example: 'python3 /path/simdbench_remote_eval_one.py'")
    ap.add_argument("--remote_simdbench_problem_file", type=str, default="",
                    help="(simdbench_one) optional remote problem jsonl path. Empty => remote uses simdbench.data.SIMD_BENCH")
    ap.add_argument("--remote_simdbench_scalar_problem_file", type=str, default="",
                    help="(simdbench_one) optional remote scalar-only problem jsonl path used for serial reference evals.")
    ap.add_argument("--remote_simdbench_k", type=str, default="1", help="(simdbench_one) k list, usually '1'")
    ap.add_argument("--remote_simdbench_n_workers", type=int, default=1, help="(simdbench_one) n_workers, recommend 1")
    ap.add_argument("--remote_simdbench_output_path", type=str, default="",
                    help="(simdbench_one) output_path passed to evaluator. Empty => auto repo root. "
                         "You may set it to '{remote_dir}' to isolate per-sample artifacts.")
    ap.add_argument("--remote_simdbench_tail_chars", type=int, default=8000, help="(simdbench_one) returned log tail chars")
    ap.add_argument(
        "--remote_simdbench_compile_timeout",
        type=float,
        default=0.0,
        help="(simdbench_one/cmd) compile timeout seconds on remote evaluator. 0 => use --remote_timeout",
    )
    # compile_only mode
    ap.add_argument("--remote_compiler", type=str, default="clang++", help="(compile_only) remote compiler executable")
    ap.add_argument("--remote_cflags", type=str, default="-O3 -march=armv8.2-a+sve", help="Remote compiler flags / eval flags")
    ap.add_argument("--remote_timeout", type=int, default=15, help="Remote eval timeout (seconds)")
    ap.add_argument("--remote_flock_path", type=str, default="", help="Remote lock file path (use flock -x)")

    ap.add_argument("--remote_fix_max_new_tokens", type=int, default=1024)
    ap.add_argument("--remote_print_summary", action="store_true")
    ap.add_argument("--remote_print_prompts", action="store_true")
    ap.add_argument("--remote_dump_prompts_dir", type=str, default="")
    ap.add_argument("--remote_prompt_max_chars", type=int, default=0)
    ap.add_argument(
        "--remote_compile_feedback_style",
        type=str,
        default="structured",
        choices=["structured", "hybrid", "raw"],
        help=(
            "Compile feedback injected into remote repair prompts. "
            "structured=diagnostic buckets only, with raw excerpt only as fallback; "
            "hybrid=structured buckets plus raw excerpt; raw=legacy raw excerpt."
        ),
    )
    ap.add_argument("--remote_reapply_name_shape", action="store_true")
    ap.add_argument("--remote_reapply_name_max_iters", type=int, default=1)
    ap.add_argument("--remote_reapply_shape_max_iters", type=int, default=1)
    ap.add_argument("--static_verifier", action="store_true",
                    help="Run local SVE static verifier before each remote eval round.")
    ap.add_argument("--static_verifier_script", type=str, default="",
                    help="Path to sve_static_verifier_v2.py")
    ap.add_argument("--static_verifier_timeout", type=int, default=20,
                    help="Timeout seconds for local static verifier")
    ap.add_argument("--remote_cpp17_compile_gate", action="store_true",
                    help="Run a remote clang++ C++17 compile-only gate before each formal remote eval round.")
    ap.add_argument("--remote_cpp17_compile_gate_mode", type=str, default="off", choices=["off", "required", "best_effort"],
                    help="Remote C++17 compile gate mode: off|required|best_effort.")
    ap.add_argument("--remote_cpp17_compile_gate_compiler", type=str, default="clang++",
                    help="Remote compiler executable used by the C++17 compile-only gate.")
    ap.add_argument("--remote_cpp17_compile_gate_cflags", type=str, default="-std=c++17 -fsyntax-only -march=armv8.2-a+sve -ferror-limit=20",
                    help="Remote compiler flags for the C++17 compile-only gate.")
    ap.add_argument("--remote_cpp17_compile_gate_timeout", type=int, default=20,
                    help="Timeout seconds for the remote C++17 compile-only gate.")
    ap.add_argument("--remote_cpp17_compile_gate_max_repair_iters", type=int, default=2,
                    help="Max local repair retries after a remote C++17 compile-only gate failure.")
    ap.add_argument("--remote_perf_precheck", action="store_true",
                    help="Before accepting remote correctness success, rerun correctness with perf-stage problem construction and optimization flags, without benchmark timing.")
    ap.add_argument("--remote_perf_precheck_optimization", type=str, default="-O1",
                    help="Optimization flag used by the remote perf-harness precheck.")
    ap.add_argument("--remote_perf_precheck_n_reputation", type=int, default=1,
                    help="Repetition count used by the remote perf-harness precheck.")
    ap.add_argument("--remote_perf_precheck_n_workers", type=int, default=1,
                    help="Worker count used by the remote perf-harness precheck.")
    ap.add_argument("--remote_perf_precheck_timeout", type=float, default=30.0,
                    help="Timeout seconds used by the remote perf-harness precheck.")
    ap.add_argument("--remote_perf_precheck_python", type=str, default="python3",
                    help="Remote Python executable used by the remote perf-harness precheck.")
    ap.add_argument("--remote_perf_precheck_repo_root", type=str, default="",
                    help="Remote simdbench repo root used by the remote perf-harness precheck.")


    # -------------------------------------------------------------------------
    # SERIAL reference fallback (differential debugging)
    # When remote eval says compile_ok=1 but run_ok=0, optionally generate a scalar reference
    # implementation using another model, run it remotely, and feed it back into the repair loop.
    # -------------------------------------------------------------------------
    ap.add_argument(
        "--serial_fallback",
        action="store_true",
        help="Enable SERIAL (scalar) reference generation when compile_ok=1 but run_ok=0. Requires --serial_llm_backend != none.",
    )
    ap.add_argument(
        "--serial_bootstrap_mode",
        type=str,
        default="phased",
        choices=["legacy", "phased"],
        help="Serial trigger policy: legacy=only compile_ok=1/run_ok=0, phased=pre_remote+compile_fail+logic_fail.",
    )
    ap.add_argument(
        "--serial_feedback_style",
        type=str,
        default="pseudocode",
        choices=[
            "pseudocode",
            "dataflow_pseudocode",
            "unified_dataflow_pseudocode",
            "constrained_pseudocode",
            "code",
            "both",
            "routed",
            "auto",
        ],
        help="How validated serial references are exposed to repair prompts. unified_dataflow_pseudocode is a code-like read/write/index AST bootstrap style when supplied by --serial_ast_bootstrap_jsonl.",
    )
    ap.add_argument(
        "--serial_pseudocode_max_chars",
        type=int,
        default=6000,
        help="Max chars for SEMANTIC_BOOTSTRAP in repair prompts; <=0 disables truncation.",
    )
    ap.add_argument(
        "--serial_ast_bootstrap_jsonl",
        type=str,
        default="",
        help=(
            "Optional JSONL from build_ast_semantic_bootstrap.py. "
            "If present, bootstrap pseudocode is taken from that AST-derived cache; "
            "serial correctness still uses the validated scalar reference."
        ),
    )
    ap.add_argument(
        "--boot_codegen",
        action="store_true",
        help=(
            "Inject the AST serial bootstrap block into the initial codegen prompt "
            "when --serial_ast_bootstrap_jsonl provides a matching task record. "
            "This does not change serial correctness/comparison; it only guides first-pass codegen."
        ),
    )
    ap.add_argument(
        "--disable_bootstrap_guard",
        action="store_true",
        help=(
            "Disable the conservative bootstrap guard that tells repair prompts not to force "
            "full-function vectorization when the bootstrap lacks explicit vectorization features."
        ),
    )
    ap.add_argument(
        "--serial_llm_backend",
        type=str,
        default="none",
        choices=["none", "hf", "openai", "deepseek"],
        help="Backend for the SERIAL reference model: none|hf|openai|deepseek",
    )

    # SERIAL HF backend
    ap.add_argument("--serial_model_path", type=str, default="", help="(serial hf) local HF model dir")
    ap.add_argument("--serial_adapter_path", type=str, default="", help="(serial hf) optional adapter dir")
    ap.add_argument(
        "--serial_device_map",
        type=str,
        default="",
        help="(serial hf) transformers device_map override (e.g. 'cpu', 'auto', 'cuda:0'). Empty => auto heuristic.",
    )

    # SERIAL API backend (defaults to main --api_* if left empty)
    ap.add_argument("--serial_api_base_url", type=str, default="", help="(serial api) base URL override")
    ap.add_argument("--serial_api_key", type=str, default="", help="(serial api) API key override")
    ap.add_argument("--serial_api_model", type=str, default="", help="(serial api) model name override")
    ap.add_argument(
        "--serial_api_endpoint",
        type=str,
        default="",
        help="(serial openai) endpoint style override: responses or chat_completions. Empty => reuse --api_endpoint",
    )

    # SERIAL generation params
    ap.add_argument("--serial_gen_max_new_tokens", type=int, default=1024)
    ap.add_argument("--serial_gen_do_sample", action="store_true")
    ap.add_argument("--serial_gen_temperature", type=float, default=0.2)
    ap.add_argument("--serial_gen_top_p", type=float, default=0.9)
    ap.add_argument("--serial_gen_repetition_penalty", type=float, default=1.0)

    # Prompt/diff size controls
    ap.add_argument("--serial_prompt_max_chars", type=int, default=0, help="Truncate SERIAL prompt prints / code blocks if >0")
    ap.add_argument("--serial_diff_max_chars", type=int, default=4000, help="Max chars of SERIAL-vs-SIMD diff included in repair prompt")
    ap.add_argument("--serial_diff_context_lines", type=int, default=3, help="Unified diff context lines")
    ap.add_argument("--serial_cache_jsonl", type=str, default="", help="JSONL cache for PASSED SERIAL refs (keyed by task_id). Default: <work_dir>/serial_ref_passed_cache.jsonl")
    ap.add_argument("--serial_cache_reload_on_miss", type=int, default=1, help="1=reload serial cache file on miss (pick up other ranks); 0=do not reload")
    ap.add_argument("--serial_ref_max_attempts_per_task", type=int, default=5, help="Max SERIAL reference attempts per task_id when triggered (compile_ok=1 but run_ok=0)")
    ap.add_argument("--serial_use_solution_scalar", action="store_true",
                    help="Use problem['solution_scalar'] (renamed to entrypoint_simd) as the validated serial oracle before LLM attempts.")
    ap.add_argument("--serial_audited_nohelper_scalar_field", type=str, default="audited_nohelper_scalar",
                    help="Optional problem field used only to generate serial bootstrap pseudocode/code hints. Correctness and serial-vs-SIMD comparison still use solution_scalar.")
    ap.add_argument("--serial_audited_nohelper_entrypoint_field", type=str, default="audited_nohelper_scalar_entrypoint",
                    help="Optional problem field naming the audited no-helper scalar entrypoint.")
    ap.add_argument("--serial_eval_intrinsic", type=str, default="scalar",
                    choices=["scalar", "SVE", "Neon", "AVX", "SSE", "RVV"],
                    help="Intrinsic passed to remote evaluator for SERIAL reference validation.")
    ap.add_argument("--serial_mismatch_harness", action="store_true",
                    help="Generate a serial-vs-simd mismatch harness (via SERIAL model) and feed mismatch info to repair prompt.")
    ap.add_argument("--serial_mismatch_max_new_tokens", type=int, default=512,
                    help="Max new tokens for serial mismatch harness generation.")
    ap.add_argument("--serial_mismatch_prompt_max_chars", type=int, default=20000,
                    help="Truncate serial mismatch harness prompt if too long.")


    # Pre-explain stage
    ap.add_argument("--pre_explain", action="store_true")
    ap.add_argument("--pre_explain_max_new_tokens", type=int, default=512)
    ap.add_argument("--pre_explain_temperature", type=float, default=0.2)
    ap.add_argument("--pre_explain_top_p", type=float, default=0.9)
    ap.add_argument("--pre_explain_repetition_penalty", type=float, default=1.0)
    ap.add_argument("--pre_explain_do_sample", action="store_true")
    ap.add_argument("--pre_explain_print_prompt", action="store_true")
    ap.add_argument("--pre_explain_print_output", action="store_true")
    ap.add_argument("--pre_explain_dump_dir", type=str, default="")
    ap.add_argument("--pre_explain_prompt_max_chars", type=int, default=0)
    ap.add_argument("--pre_explain_max_retries", type=int, default=2, help="Retries if pre-explain output is invalid (missing markers / looks like code)")

    # Dialogue trace output
    ap.add_argument("--dialogue_log_file", type=str, default="")
    ap.add_argument("--dialogue_dump_dir", type=str, default="")
    ap.add_argument("--dialogue_max_chars", type=int, default=0)
    ap.add_argument(
        "--progress_print_every",
        type=int,
        default=0,
        help="If >0, emit plain-text progress every N completed samples (plus failures/final sample).",
    )

        # -------------------------
    # Semantic gate for reduction usage (NEW)
    # -------------------------
    ap.add_argument(
        "--reduction_gate",
        type=str,
        default="soft",
        choices=["off", "soft", "hard"],
        help=(
            "How to handle SVE vector reductions (sv*addv/maxv/minv/orv/andv/eorv) when the task is inferred to be per-element. "
            "off: do nothing; soft: penalize + try semantic fix; hard: treat as semantic error (only when inference confidence is high)."
        ),
    )
    ap.add_argument(
        "--semantic_fix_max_iters",
        type=int,
        default=1,
        help="Semantic repair iterations when semantic gate triggers (0=disable).",
    )
    ap.add_argument("--semantic_fix_max_new_tokens", type=int, default=1024)
    ap.add_argument("--semantic_fix_attempts_per_iter", type=int, default=2)
    ap.add_argument("--semantic_fix_print_prompts", action="store_true")
    ap.add_argument("--semantic_fix_prompt_max_chars", type=int, default=0)

    args = ap.parse_args()
    global SEMANTIC_REDUCTION_GATE
    SEMANTIC_REDUCTION_GATE = str(args.reduction_gate)

    # Set global completion mode used by normalization + repair prompts.
    global COMPLETION_MODE
    COMPLETION_MODE = str(args.completion_mode)

    rank, world_size, local_rank, distributed = get_dist_info()
    args.remote_flock_path = resolve_remote_flock_path_for_rank(
        args.remote_flock_path,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )

    # Distributed init: for API backend we can use gloo (no GPU required)
    dist_backend = None
    if distributed:
    # No need to init torch.distributed for pure sharding inference.
        if torch.cuda.is_available() and args.llm_backend == "hf":
            torch.cuda.set_device(local_rank)
        shard_id = rank
        num_shards = world_size
    else:
        shard_id = args.shard_id
        num_shards = args.num_shards

    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    torch.manual_seed(args.seed + shard_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + shard_id)

    if args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16 if args.bf16 or not args.fp16 else torch.float16

    # -------------------------
    # Load backend (HF or API)
    # -------------------------
    model = None
    tok = None

    if args.llm_backend == "hf":
        if not args.model_path:
            raise SystemExit("--model_path is required when --llm_backend=hf")

        if distributed and torch.cuda.is_available() and args.llm_backend == "hf":
            device_map = {"": f"cuda:{local_rank}"}
        else:
            device_map = "auto"

        model_path = str(Path(args.model_path).expanduser())
        if not Path(model_path).is_dir():
            raise SystemExit(f"Model dir not found: {model_path}")

        adapter_path = str(Path(args.adapter_path).expanduser()) if args.adapter_path else ""
        if adapter_path and not Path(adapter_path).is_dir():
            raise SystemExit(f"Adapter dir not found: {adapter_path}")

        model, tok = load_model_and_tokenizer(
            model_path=model_path,
            adapter_path=adapter_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            merge_adapter=args.merge_adapter,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
    else:
        # API backend
        provider = args.llm_backend

        if args.api_base_url:
            base_url = args.api_base_url
        else:
            base_url = "https://api.openai.com/v1" if provider == "openai" else "https://api.deepseek.com"

        if args.api_key:
            api_key = args.api_key
        else:
            env_key = "OPENAI_API_KEY" if provider == "openai" else "DEEPSEEK_API_KEY"
            api_key = os.environ.get(env_key, "")

        if not api_key:
            raise SystemExit(f"Missing API key: set --api_key or env var {'OPENAI_API_KEY' if provider == 'openai' else 'DEEPSEEK_API_KEY'}")

        if args.api_model:
            api_model = args.api_model
        else:
            # Defaults: OpenAI -> GPT-5.2 family; DeepSeek -> reasoning model
            api_model = "gpt-5.2" if provider == "openai" else "deepseek-reasoner"

        endpoint = "responses" if provider == "openai" else "chat_completions"
        if provider == "openai":
            endpoint = args.api_endpoint

        extra_headers = _coerce_json_obj(args.api_extra_headers_json, what="--api_extra_headers_json") if args.api_extra_headers_json else {}
        extra_body = _coerce_json_obj(args.api_extra_body_json, what="--api_extra_body_json") if args.api_extra_body_json else {}

        model = ApiBackend(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=api_model,
            endpoint=endpoint,
            timeout_s=args.api_timeout,
            max_retries=args.api_max_retries,
            retry_backoff_s=args.api_retry_backoff,
            retry_max_sleep_s=args.api_retry_max_sleep,
            extra_headers=extra_headers,
            extra_body=extra_body,
            prompt_max_chars=args.api_prompt_max_chars,
            print_requests=args.api_print_requests,
        )
        tok = None


    # -------------------------
    # Optional SERIAL reference model (for differential debugging)
    # -------------------------
    serial_model = None
    serial_tok = None

    serial_cache = None  # SerialRefCache (initialized later once work_dir is known)

    if bool(getattr(args, "serial_fallback", False)) and str(getattr(args, "serial_llm_backend", "none")) != "none":
        if args.serial_llm_backend == "hf":
            if not args.serial_model_path:
                raise SystemExit("--serial_model_path is required when --serial_llm_backend=hf")

            serial_model_path = str(Path(args.serial_model_path).expanduser())
            if not Path(serial_model_path).is_dir():
                raise SystemExit(f"Serial model dir not found: {serial_model_path}")

            serial_adapter_path = str(Path(args.serial_adapter_path).expanduser()) if args.serial_adapter_path else ""
            if serial_adapter_path and not Path(serial_adapter_path).is_dir():
                raise SystemExit(f"Serial adapter dir not found: {serial_adapter_path}")

            # Heuristic: if main model is HF on GPU, default serial model to CPU to reduce OOM risk.
            if args.serial_device_map:
                serial_device_map = args.serial_device_map
            else:
                if torch.cuda.is_available() and args.llm_backend == "hf":
                    serial_device_map = "cpu"
                else:
                    serial_device_map = "auto"

            serial_model, serial_tok = load_model_and_tokenizer(
                model_path=serial_model_path,
                adapter_path=serial_adapter_path,
                device_map=serial_device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
                merge_adapter=args.merge_adapter,
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            )
        else:
            # API backend for SERIAL ref (defaults reuse main --api_* unless overridden)
            provider = str(args.serial_llm_backend)

            if args.serial_api_base_url:
                base_url = args.serial_api_base_url
            else:
                base_url = args.api_base_url if args.api_base_url else ("https://api.openai.com/v1" if provider == "openai" else "https://api.deepseek.com")

            if args.serial_api_key:
                api_key = args.serial_api_key
            elif args.api_key:
                api_key = args.api_key
            else:
                env_key = "OPENAI_API_KEY" if provider == "openai" else "DEEPSEEK_API_KEY"
                api_key = os.environ.get(env_key, "")

            if not api_key:
                raise SystemExit(f"Missing SERIAL API key for provider={provider}. Set --serial_api_key or reuse --api_key / env var.")

            if args.serial_api_model:
                api_model = args.serial_api_model
            elif args.api_model:
                api_model = args.api_model
            else:
                api_model = "gpt-5.2" if provider == "openai" else "deepseek-reasoner"

            endpoint = "responses" if provider == "openai" else "chat_completions"
            if provider == "openai":
                ep = (args.serial_api_endpoint or "").strip()
                if ep:
                    if ep not in ("responses", "chat_completions"):
                        raise SystemExit("--serial_api_endpoint must be 'responses' or 'chat_completions'")
                    endpoint = ep
                else:
                    endpoint = args.api_endpoint

            extra_headers = _coerce_json_obj(args.api_extra_headers_json, what="--api_extra_headers_json") if args.api_extra_headers_json else {}
            extra_body = _coerce_json_obj(args.api_extra_body_json, what="--api_extra_body_json") if args.api_extra_body_json else {}

            serial_model = ApiBackend(
                provider=provider,
                base_url=base_url,
                api_key=api_key,
                model=api_model,
                endpoint=endpoint,
                timeout_s=args.api_timeout,
                max_retries=args.api_max_retries,
                retry_backoff_s=args.api_retry_backoff,
                retry_max_sleep_s=args.api_retry_max_sleep,
                extra_headers=extra_headers,
                extra_body=extra_body,
                prompt_max_chars=args.api_prompt_max_chars,
                print_requests=args.api_print_requests,
            )
            serial_tok = None

    whitelist_set: Optional[Set[str]] = None
    whitelist_list: Optional[List[str]] = None
    sigs: Dict[str, List[List[str]]] = {}
    rets: Dict[str, List[str]] = {}
    op_index: Optional[Dict[str, List[str]]] = None

    if args.whitelist:
        wl_path = Path(args.whitelist).expanduser()
        if not wl_path.is_file():
            raise SystemExit(f"Whitelist file not found: {wl_path}")
        whitelist_set, whitelist_list, sigs, rets = load_whitelist(wl_path)
    if args.intrinsic == "SVE" and whitelist_set is not None and whitelist_list is not None:
        whitelist_set, whitelist_list, sigs, rets = augment_sve_whitelist(
            whitelist_set, whitelist_list, sigs, rets
        )
        op_index = build_op_index(whitelist_list)

    remote_enabled = args.remote_feedback_rounds > 0
    if remote_enabled:
        missing = []
        for k in ["remote_user", "remote_host", "remote_ssh_key"]:
            if getattr(args, k) in (None, ""):
                missing.append(k)
        if missing:
            raise SystemExit(f"--remote_feedback_rounds>0 but missing remote args: {missing}")

        if args.remote_eval_mode == "cmd" and not args.remote_eval_cmd:
            raise SystemExit("--remote_eval_mode=cmd but --remote_eval_cmd is empty")

        if args.remote_eval_mode == "simdbench_one" and not args.remote_simdbench_eval:
            raise SystemExit("--remote_eval_mode=simdbench_one but --remote_simdbench_eval is empty")
            
    problems = load_problems(args.problem_file)
    serial_ast_bootstrap_map: Dict[str, Dict[str, Any]] = {}
    if str(getattr(args, "serial_ast_bootstrap_jsonl", "") or "").strip():
        serial_ast_bootstrap_map = load_serial_ast_bootstrap_jsonl(str(args.serial_ast_bootstrap_jsonl))
        print(
            f"[SERIAL_AST_BOOTSTRAP] loaded {len(serial_ast_bootstrap_map)} records from {args.serial_ast_bootstrap_jsonl}"
        )
    rng = random.Random(args.seed)   # 保证可复现
    rng.shuffle(problems)

    base_out = Path(args.output)
    out_path = shard_output_path(base_out, shard_id=shard_id, num_shards=num_shards)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_work_queue_enabled = (
        bool(getattr(args, "dynamic_work_stealing", True))
        and num_shards > 1
        and len(problems) > 0
        and fcntl is not None
    )
    if bool(getattr(args, "dynamic_work_stealing", True)) and num_shards > 1 and fcntl is None:
        print("[WARN] --dynamic_work_stealing requested but fcntl is unavailable; falling back to static sharding.")

    work_queue_state_path: Optional[Path] = None
    work_queue_lock_path: Optional[Path] = None
    work_queue_ready_path: Optional[Path] = None
    work_queue_fingerprint = ""
    if dynamic_work_queue_enabled:
        (
            work_queue_state_path,
            work_queue_lock_path,
            work_queue_ready_path,
            work_queue_fingerprint,
        ) = init_dynamic_work_queue(
            base_out=base_out,
            problems=problems,
            n_samples=int(args.n_samples),
            seed=int(args.seed),
            rank=rank,
            world_size=num_shards,
            timeout_s=int(getattr(args, "dynamic_work_queue_init_timeout_s", 600) or 600),
        )
        if rank == 0:
            print(
                f"[WORK_STEALING] enabled task_level queue={work_queue_state_path} "
                f"tasks={len(problems)} n_samples={args.n_samples} ranks={num_shards}",
                flush=True,
            )

    my_problems = [] if dynamic_work_queue_enabled else shard_round_robin(problems, num_shards=num_shards, shard_id=shard_id)

    work_dir: Optional[Path] = None
    if args.save_intermediate or args.log_file or args.remote_dump_prompts_dir or args.name_fix_dump_prompts_dir or args.shape_fix_dump_prompts_dir or args.dialogue_dump_dir:
        if args.work_dir:
            work_dir = Path(args.work_dir)
        else:
            work_dir = base_out.with_suffix(base_out.suffix + ".work")
        work_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------
        # SERIAL reference cache (PASSED serial refs per task_id)
        # -------------------------
        if args.serial_fallback and (serial_model is not None):
            serial_cache_jsonl = (args.serial_cache_jsonl or "").strip()
            if not serial_cache_jsonl:
                try:
                    serial_cache_jsonl = str(work_dir / "serial_ref_passed_cache.jsonl")
                except Exception:
                    serial_cache_jsonl = ""
            if serial_cache_jsonl:
                serial_cache = SerialRefCache(
                    serial_cache_jsonl,
                    reload_on_miss=bool(int(getattr(args, "serial_cache_reload_on_miss", 1))),
                    rank=rank,
                )
                try:
                    serial_cache.load()
                except Exception:
                    pass


    meta_log_path: Optional[Path] = None
    if args.log_file:
        meta_log_path = Path(args.log_file)
        if distributed:
            meta_log_path = meta_log_path.with_name(meta_log_path.name + f".rank{rank}.jsonl")

    dialogue_log_path: Optional[Path] = None
    if args.dialogue_log_file:
        dialogue_log_path = Path(args.dialogue_log_file)
        if distributed:
            dialogue_log_path = dialogue_log_path.with_name(dialogue_log_path.name + f".rank{rank}.jsonl")

    total_local = (len(problems) if dynamic_work_queue_enabled else len(my_problems)) * args.n_samples
    local_task_total = len(problems) if dynamic_work_queue_enabled else len(my_problems)
    total_global = len(problems) * args.n_samples
    progress_print_every = int(max(0, getattr(args, "progress_print_every", 0) or 0))
    progress_done_local = 0
    pbar = None
    if tqdm is not None:
        pos = local_rank if distributed else shard_id
        name = f"rank{rank}" if distributed else f"shard{shard_id}"
        pbar = tqdm(
            total=(None if dynamic_work_queue_enabled else total_local),
            desc=f"{name}/{num_shards}{' dynamic' if dynamic_work_queue_enabled else ''} -> {out_path.name}",
            position=pos,
            dynamic_ncols=True,
            leave=True,
        )

    def emit_progress(
        event: str,
        *,
        task_id: str = "",
        task_local_idx: int = 0,
        sample_idx: Optional[int] = None,
        status: str = "",
        elapsed_s: Optional[float] = None,
        remote_rounds_used: Optional[int] = None,
        remote_reason: str = "",
    ) -> None:
        if progress_print_every <= 0:
            return
        parts = [
            "[PROGRESS]",
            f"event={event}",
            f"rank={rank}",
            f"shard={shard_id + 1}/{num_shards}",
            f"local_done={progress_done_local}/{total_local}",
            f"global_total={total_global}",
        ]
        if task_local_idx > 0:
            parts.append(f"local_task={task_local_idx}/{local_task_total}")
        if task_id:
            parts.append(f"id={task_id}")
        if sample_idx is not None:
            parts.append(f"sample={sample_idx}/{args.n_samples}")
        if status:
            parts.append(f"status={status}")
        if elapsed_s is not None:
            parts.append(f"elapsed_s={elapsed_s:.2f}")
        if remote_rounds_used is not None:
            parts.append(f"remote_rounds_used={remote_rounds_used}")
        if remote_reason:
            parts.append(f"remote_reason={remote_reason}")
        print(" ".join(parts), flush=True)

    do_sample = args.do_sample or (args.temperature > 0)
    do_sample_repair = args.do_sample or (args.repair_temperature > 0)

    intrinsics_rules_block = ""
    if args.append_intrinsics_rules:
        include_hint = {
            "SVE": "#include <arm_sve.h>",
            "Neon": "#include <arm_neon.h>",
            "AVX": "#include <immintrin.h>",
            "SSE": "#include <immintrin.h>",
            "RVV": "#include <riscv_vector.h>",
        }
        intrinsics_rules_block = (
            "/*\n"
            "[INTRINSICS_RULES - STRICT]\n"
            "- Output ONLY C/C++ code. No markdown, no explanation.\n"
            f"- Always include: {include_hint.get(args.intrinsic, '')}\n"
            "[END_INTRINSICS_RULES - STRICT]\n"
            "*/\n"
        )

    tmp_out_path = out_path.with_suffix(out_path.suffix + ".tmp")
    wf = tmp_out_path.open("w", encoding="utf-8")
    lf = meta_log_path.open("w", encoding="utf-8") if meta_log_path is not None else None
    df = dialogue_log_path.open("w", encoding="utf-8") if dialogue_log_path is not None else None

    fatal_error = ""
    try:
        def iter_work_items():
            if dynamic_work_queue_enabled:
                local_claim_count = 0
                assert work_queue_state_path is not None
                assert work_queue_lock_path is not None
                while True:
                    item = claim_dynamic_work_item(
                        state_path=work_queue_state_path,
                        lock_path=work_queue_lock_path,
                        fingerprint=work_queue_fingerprint,
                        rank=rank,
                    )
                    if item is None:
                        break
                    local_claim_count += 1
                    problem_index = int(item["index"])
                    p_item = problems[problem_index]
                    yield local_claim_count, p_item, str(item.get("key") or ""), int(item.get("queue_pos") or 0), int(item.get("queue_total") or len(problems))
            else:
                for local_idx, p_item in enumerate(my_problems, start=1):
                    yield local_idx, p_item, "", local_idx, len(my_problems)

        for task_local_idx, p, work_item_key, work_queue_pos, work_queue_total in iter_work_items():
            task_id = str(p["task_id"])
            emit_progress("task_start", task_id=task_id, task_local_idx=task_local_idx)
            if dynamic_work_queue_enabled:
                print(
                    f"[WORK_STEALING] rank={rank} claimed {work_queue_pos}/{work_queue_total} "
                    f"id={task_id} local_claim={task_local_idx}",
                    flush=True,
                )
            task_completed_for_queue = False
            try:
                task_text = str(p.get("task", "") or "").strip()
                
                intrinsic = str(p.get("intrinsic", args.intrinsic) or args.intrinsic)

                raw_prompt = str(p["prompt"])
                user_prompt = normalize_prompt_prefix(raw_prompt)
                func_name = extract_function_name_from_prompt_prefix(user_prompt)
                # If task_text is missing or useless, extract from prompt comment.
                if (not task_text) or (task_text.lower() in {"generation", "gen", "todo"}):
                    extracted = extract_task_text_from_prompt(raw_prompt)
                    if extracted:
                        task_text = extracted

                # In completion "full" mode we need the model output to be a standalone
                # translation unit, so we capture any required <...> includes that
                # appear in the prompt prefix and later ensure they exist in the
                # emitted completion.
                global CURRENT_REQUIRED_INCLUDES
                CURRENT_REQUIRED_INCLUDES = extract_angle_includes(user_prompt) or extract_angle_includes(raw_prompt)
                global CURRENT_TARGET_FUNC_DECL
                CURRENT_TARGET_FUNC_DECL = (
                    extract_function_decl_from_prompt_prefix(raw_prompt, func_name)
                    or extract_function_decl_from_prompt_prefix(user_prompt, func_name)
                    or ""
                )

                task_dialogue_prefix: List[Dict] = []

                expanded_spec_text = ""
                explain_prompt_used = ""
                explain_did_replace = False

                if args.pre_explain and intrinsic.upper() == "SVE":
                    explain_system_prompt = sys_prompt_explain(intrinsic)
                    do_sample_explain = bool(args.pre_explain_do_sample)

                    # Main (attempt 0): original prompt with replacement text
                    explain_prompt_used, explain_did_replace = build_pre_explain_prompt(raw_prompt)

                    dlg_set_capture(task_dialogue_prefix, max_chars=args.dialogue_max_chars)
                    dlg_set_stage("pre_explain")

                    pre_explain_ok = False
                    pre_explain_reason = ""
                    explain_raw_out = ""

                    max_tries = max(0, int(args.pre_explain_max_retries))
                    for attempt in range(max_tries + 1):
                        if attempt == 0:
                            user_text_explain = explain_prompt_used
                        else:
                            # Fallback: keep only comment blocks to reduce code-echo.
                            comment_only = extract_comment_blocks_for_explain(raw_prompt)
                            user_text_explain, _ = build_pre_explain_prompt(comment_only)

                        if args.pre_explain_print_prompt:
                            print("\n" + "=" * 120)
                            print(f"[PRE_EXPLAIN_PROMPT] rank={rank} id={task_id} attempt={attempt} replaced={int(explain_did_replace) if attempt==0 else -1}")
                            print("-" * 120)
                            if args.pre_explain_prompt_max_chars and args.pre_explain_prompt_max_chars > 0 and len(user_text_explain) > args.pre_explain_prompt_max_chars:
                                print(user_text_explain[:args.pre_explain_prompt_max_chars] + "\n...[TRUNCATED]...")
                            else:
                                print(user_text_explain)
                            print("=" * 120 + "\n")

                        try:
                            explain_out = generate_text(
                                model,
                                tok,
                                user_text=user_text_explain,
                                system_text=explain_system_prompt,
                                max_new_tokens=args.pre_explain_max_new_tokens,
                                do_sample=do_sample_explain,
                                temperature=args.pre_explain_temperature,
                                top_p=args.pre_explain_top_p,
                                repetition_penalty=args.pre_explain_repetition_penalty,
                            )
                        except Exception as _e_explain:
                            pre_explain_reason = f"pre_explain_generate_error: {_e_explain!r}"
                            if args.pre_explain_print_prompt:
                                print(f"[WARN] pre_explain generate_text failed rank={rank} id={task_id} attempt={attempt}: {_e_explain!r}")
                            continue
                        explain_raw_out = (explain_out or "").strip()

                        ok_spec, cleaned_spec, reason = validate_expanded_spec(explain_raw_out)
                        pre_explain_reason = reason
                        if ok_spec:
                            expanded_spec_text = cleaned_spec
                            # Record which prompt actually succeeded (useful for debugging)
                            explain_prompt_used = user_text_explain
                            pre_explain_ok = True
                            break

                    dlg_set_capture(None)

                    if args.pre_explain_print_output:
                        print("\n" + "=" * 120)
                        print(f"[PRE_EXPLAIN_OUTPUT] rank={rank} id={task_id} ok={int(pre_explain_ok)} reason={pre_explain_reason} chars={len(expanded_spec_text)}")
                        print("-" * 120)
                        if pre_explain_ok:
                            print(expanded_spec_text)
                        else:
                            # Print a small tail to help debug without flooding logs
                            print(_truncate_middle(explain_raw_out, 2000))
                        print("=" * 120 + "\n")

                    if (not pre_explain_ok) or (not expanded_spec_text.strip()):
                        expanded_spec_text = ""

                    if args.pre_explain_dump_dir:
                        d = Path(args.pre_explain_dump_dir) / f"rank{rank}" / task_id
                        d.mkdir(parents=True, exist_ok=True)
                        (d / "pre_explain_prompt.txt").write_text(explain_prompt_used, encoding="utf-8")
                        (d / "pre_explain_output_raw.txt").write_text(explain_raw_out, encoding="utf-8")
                        (d / "pre_explain_output_validated.txt").write_text(expanded_spec_text, encoding="utf-8")
                        (d / "pre_explain_meta.json").write_text(
                            json.dumps(
                                {
                                    "task_id": task_id,
                                    "rank": rank,
                                    "did_replace": bool(explain_did_replace),
                                    "ok": bool(pre_explain_ok),
                                    "reason": pre_explain_reason,
                                    "max_new_tokens": args.pre_explain_max_new_tokens,
                                    "temperature": args.pre_explain_temperature,
                                    "top_p": args.pre_explain_top_p,
                                    "repetition_penalty": args.pre_explain_repetition_penalty,
                                    "max_retries": int(args.pre_explain_max_retries),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )

                boot_codegen_block = ""
                if bool(getattr(args, "boot_codegen", False)):
                    boot_rec = lookup_serial_ast_bootstrap_record(serial_ast_bootstrap_map, str(task_id))
                    if isinstance(boot_rec, dict):
                        boot_pseudocode = str(boot_rec.get("bootstrap_pseudocode") or "").strip()
                        if boot_pseudocode:
                            max_pseudo = int(getattr(args, "serial_pseudocode_max_chars", 6000) or 0)
                            if max_pseudo > 0 and len(boot_pseudocode) > max_pseudo:
                                boot_pseudocode = _truncate_middle(boot_pseudocode, max_pseudo)
                            boot_key = str(boot_rec.get("_lookup_key") or "").strip()
                            boot_source = str(getattr(args, "serial_ast_bootstrap_jsonl", "") or "serial_ast_bootstrap_jsonl").strip()
                            if boot_key:
                                boot_source = f"{boot_source}:{boot_key}"
                            boot_style = str(boot_rec.get("route_style") or "").strip()
                            boot_pattern = str(boot_rec.get("route_pattern") or "").strip()
                            boot_reason = str(boot_rec.get("route_reason") or "").strip()
                            boot_guard = ""
                            if (
                                not bool(getattr(args, "disable_bootstrap_guard", False))
                                and requested_routed_feedback_style(getattr(args, "serial_feedback_style", ""))
                            ):
                                boot_guard = BOOTSTRAP_VECTORIZATION_GUARD + "\n"
                            boot_codegen_block = (
                                "\n\n[BOOTSTRAP]\n"
                                "- stage: initial_codegen"
                                + (f"; route_style: {boot_style}" if boot_style else "")
                                + (f"; route_pattern: {boot_pattern}" if boot_pattern else "")
                                + (f"; reason: {boot_reason}" if boot_reason else "")
                                + "\n"
                                "- Use the semantic bootstrap below as a task-specific repair guide for dataflow, indexing, predicates, reductions, and vectorizable regions.\n"
                                "- Keep the exact target function signature from the prompt prefix.\n"
                                + boot_guard
                                + "[/BOOTSTRAP]\n\n"
                                "[SEMANTIC_BOOTSTRAP]\n"
                                + boot_pseudocode
                                + "\n[/SEMANTIC_BOOTSTRAP]\n"
                            )

                if intrinsics_rules_block:
                    user_prompt_gen = user_prompt.rstrip() + "\n\n" + intrinsics_rules_block + boot_codegen_block
                else:
                    user_prompt_gen = user_prompt.rstrip() + boot_codegen_block

                system_prompt = sys_prompt(intrinsic, task_text)
                if expanded_spec_text.strip():
                    system_prompt += "\n\n[EXPANDED_SPEC_AND_SVE_PLAN]\n" + sanitize_expanded_spec_for_codegen(expanded_spec_text, intrinsic).strip() + "\n[END_EXPANDED_SPEC_AND_SVE_PLAN]\n"

                spec_text = ""
                if task_text.strip():
                    spec_text += "[TASK]\n" + task_text.strip() + "\n\n"
                spec_text += "[PROMPT_PREFIX]\n" + user_prompt.rstrip() + "\n"
                if expanded_spec_text.strip():
                    spec_text += "\n[EXPANDED_SPEC_AND_SVE_PLAN]\n" + sanitize_expanded_spec_for_codegen(expanded_spec_text, intrinsic).strip() + "\n[END_EXPANDED_SPEC_AND_SVE_PLAN]\n"

                for sidx in range(args.n_samples):
                    rec: Dict = {"task_id": task_id, "sample_idx": sidx, "rank": rank, "ok": False, "llm_backend": args.llm_backend}
                    t0 = time.time()
                    name_info = None
                    shape_info = None
                    remote_info = None

                    case_dir: Optional[Path] = None
                    if work_dir is not None and (args.save_intermediate or remote_enabled or args.dialogue_dump_dir):
                        case_dir = work_dir / f"{task_id}" / f"sample{sidx}" / f"rank{rank}"
                        case_dir.mkdir(parents=True, exist_ok=True)

                    dialogue_turns: List[Dict] = [x.copy() for x in task_dialogue_prefix]
                    dlg_set_capture(dialogue_turns, max_chars=args.dialogue_max_chars)
                    dlg_set_stage("codegen")

                    try:
                        dlg_set_stage("codegen")
                        raw = generate_text(
                            model,
                            tok,
                            user_text=user_prompt_gen,
                            system_text=system_prompt,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=do_sample,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            repetition_penalty=args.repetition_penalty,
                        )
                        completion = normalize_completion_snippet(raw, func_name)
                        if intrinsic == "SVE":
                            completion, _pp_info = postprocess_sve_common_fixes(completion)

                        if whitelist_set is not None and whitelist_list is not None and op_index is not None:
                            dlg_set_stage("name_fix")
                            per_name_dump = None
                            if args.name_fix_dump_prompts_dir:
                                per_name_dump = Path(args.name_fix_dump_prompts_dir) / f"rank{rank}" / task_id / f"sample{sidx}"
                            completion, name_info = rag_fix_names(
                                model=model,
                                tok=tok,
                                completion_in=completion,
                                whitelist_set=whitelist_set,
                                whitelist_list=whitelist_list,
                                op_index=op_index,
                                spec_text=spec_text,
                                max_iters=args.name_fix_max_iters,
                                max_new_tokens=args.name_fix_max_new_tokens,
                                top_k=args.name_fix_top_k,
                                cutoff=args.name_fix_cutoff,
                                do_sample=do_sample_repair,
                                temperature=args.repair_temperature,
                                top_p=args.repair_top_p,
                                repetition_penalty=args.repair_repetition_penalty,
                                sid=task_id,
                                rank=rank,
                                func_name=func_name,
                                attempts_per_iter=args.name_fix_attempts_per_iter,
                                print_prompts=args.name_fix_print_prompts,
                                dump_prompts_dir=per_name_dump,
                                prompt_max_chars=args.name_fix_prompt_max_chars,
                            )

                            if intrinsic == "SVE":
                                completion, _pp_info = postprocess_sve_common_fixes(completion)

                            dlg_set_stage("shape_fix")
                            per_shape_dump = None
                            if args.shape_fix_dump_prompts_dir:
                                per_shape_dump = Path(args.shape_fix_dump_prompts_dir) / f"rank{rank}" / task_id / f"sample{sidx}"
                            completion, shape_info = rag_fix_shapes(
                                model=model,
                                tok=tok,
                                completion_in=completion,
                                whitelist_set=whitelist_set,
                                whitelist_list=whitelist_list,
                                sigs=sigs,
                                rets=rets,
                                max_iters=args.shape_fix_max_iters,
                                spec_text=spec_text,
                                max_new_tokens=args.shape_fix_max_new_tokens,
                                do_sample=do_sample_repair,
                                temperature=args.repair_temperature,
                                top_p=args.repair_top_p,
                                repetition_penalty=args.repair_repetition_penalty,
                                check_types=args.shape_fix_check_types,
                                sid=task_id,
                                rank=rank,
                                func_name=func_name,
                                attempts_per_iter=args.shape_fix_attempts_per_iter,
                                print_summary=args.shape_fix_print_summary,
                                print_prompts=args.shape_fix_print_prompts,
                                dump_prompts_dir=per_shape_dump,
                                prompt_max_chars=args.shape_fix_prompt_max_chars,
                            )

                            if intrinsic == "SVE":
                                completion, _pp_info = postprocess_sve_common_fixes(completion)

                                            # -------------------------
                        # Semantic reduction repair (NEW)
                        # -------------------------
                        if int(getattr(args, "semantic_fix_max_iters", 0) or 0) > 0:
                            dlg_set_stage("semantic_fix_reduction")
                            completion, sem_red_info = semantic_repair_reduction_loop(
                                model,
                                tok,
                                completion_in=completion,
                                spec_text=spec_text,
                                func_name=func_name,
                                max_iters=int(args.semantic_fix_max_iters),
                                max_new_tokens=int(args.semantic_fix_max_new_tokens),
                                do_sample=do_sample_repair,
                                temperature=args.repair_temperature,
                                top_p=args.repair_top_p,
                                repetition_penalty=args.repair_repetition_penalty,
                                attempts_per_iter=int(args.semantic_fix_attempts_per_iter),
                                print_prompts=bool(args.semantic_fix_print_prompts),
                                prompt_max_chars=int(args.semantic_fix_prompt_max_chars or 0),
                            )
                            rec["semantic_reduction_triggered"] = bool(sem_red_info.get("triggered", False))
                            rec["semantic_reduction_iters"] = int(sem_red_info.get("iters", 0) or 0)
                            rec["semantic_reduction_final_issue"] = sem_red_info.get("final_issue", "none")

                        if remote_enabled:
                            dlg_set_stage("remote_feedback")
                            per_remote_dump = None
                            if args.remote_dump_prompts_dir:
                                per_remote_dump = Path(args.remote_dump_prompts_dir) / f"rank{rank}" / task_id / f"sample{sidx}"
                            elif case_dir is not None:
                                per_remote_dump = case_dir / "_remote"
                            serial_ast_bootstrap_rec = lookup_serial_ast_bootstrap_record(
                                serial_ast_bootstrap_map, str(task_id)
                            )
                            serial_ast_bootstrap_pseudocode = ""
                            serial_ast_bootstrap_style = ""
                            serial_ast_bootstrap_pattern = ""
                            serial_ast_bootstrap_reason = ""
                            serial_ast_bootstrap_source = ""
                            if isinstance(serial_ast_bootstrap_rec, dict):
                                serial_ast_bootstrap_pseudocode = str(
                                    serial_ast_bootstrap_rec.get("bootstrap_pseudocode") or ""
                                ).strip()
                                serial_ast_bootstrap_style = str(
                                    serial_ast_bootstrap_rec.get("route_style") or ""
                                ).strip()
                                serial_ast_bootstrap_pattern = str(
                                    serial_ast_bootstrap_rec.get("route_pattern") or ""
                                ).strip()
                                serial_ast_bootstrap_reason = str(
                                    serial_ast_bootstrap_rec.get("route_reason") or ""
                                ).strip()
                                if serial_ast_bootstrap_pseudocode:
                                    lookup_key = str(serial_ast_bootstrap_rec.get("_lookup_key") or "").strip()
                                    base_source = str(
                                        getattr(args, "serial_ast_bootstrap_jsonl", "")
                                        or "serial_ast_bootstrap_jsonl"
                                    ).strip()
                                    serial_ast_bootstrap_source = (
                                        f"{base_source}:{lookup_key}" if lookup_key else base_source
                                    )

                            completion, remote_info = remote_feedback_loop(
                                model=model,
                                tok=tok,
                                task_id=task_id,
                                sample_idx=sidx,
                                intrinsic=intrinsic,
                                prompt_prefix=user_prompt,
                                spec_text=spec_text,
                                completion_in=completion,
                                func_name=func_name,
                                whitelist_set=whitelist_set,
                                whitelist_list=whitelist_list,
                                op_index=op_index,
                                sigs=sigs,
                                rets=rets,
                                do_sample=do_sample_repair,
                                temperature=args.repair_temperature,
                                top_p=args.repair_top_p,
                                repetition_penalty=args.repair_repetition_penalty,
                                max_new_tokens=args.remote_fix_max_new_tokens,
                                rounds=args.remote_feedback_rounds,
                                remote_early_stop_no_improve=bool(getattr(args, "remote_early_stop_no_improve", False)),
                                remote_score_mode=str(getattr(args, "remote_score_mode", "legacy") or "legacy"),
                                remote_repair_cursor_mode=str(getattr(args, "remote_repair_cursor_mode", "latest") or "latest"),
                                remote_semantic_no_improve_patience=int(getattr(args, "remote_semantic_no_improve_patience", 0) or 0),
                                remote_compile_feedback_style=str(getattr(args, "remote_compile_feedback_style", "structured") or "structured"),
                                disable_bootstrap_guard=bool(getattr(args, "disable_bootstrap_guard", False)),
                                remote_user=args.remote_user,
                                remote_host=args.remote_host,
                                remote_port=args.remote_port,
                                remote_ssh_key=str(Path(args.remote_ssh_key).expanduser()),
                                remote_tmp_root=args.remote_tmp_root,
                                remote_eval_mode=args.remote_eval_mode,
                                remote_eval_cmd=args.remote_eval_cmd,
                                remote_simdbench_eval=args.remote_simdbench_eval,
                                remote_simdbench_problem_file=args.remote_simdbench_problem_file,
                                remote_simdbench_scalar_problem_file=args.remote_simdbench_scalar_problem_file,
                                remote_simdbench_k=args.remote_simdbench_k,
                                remote_simdbench_n_workers=args.remote_simdbench_n_workers,
                                remote_simdbench_output_path=args.remote_simdbench_output_path,
                                remote_simdbench_tail_chars=args.remote_simdbench_tail_chars,
                                remote_simdbench_compile_timeout=args.remote_simdbench_compile_timeout,
                                remote_compiler=args.remote_compiler,
                                remote_cflags=args.remote_cflags,
                                remote_timeout=args.remote_timeout,
                                remote_flock_path=args.remote_flock_path,
                                remote_no_strict_hostkey=args.remote_no_strict_hostkey,
                                reapply_name_shape=args.remote_reapply_name_shape,
                                reapply_name_max_iters=args.remote_reapply_name_max_iters,
                                reapply_shape_max_iters=args.remote_reapply_shape_max_iters,
                                print_summary=args.remote_print_summary,
                                print_prompts=args.remote_print_prompts,
                                dump_dir=per_remote_dump,
                                # SERIAL reference fallback (optional)
                                serial_model=serial_model,
                                serial_tok=serial_tok,
                                serial_llm_backend=str(getattr(args, "serial_llm_backend", "none")),
                                serial_api_model=str(getattr(args, "serial_api_model", "")),
                                serial_fallback=bool(getattr(args, "serial_fallback", False)),
                                serial_bootstrap_mode=str(getattr(args, "serial_bootstrap_mode", "phased") or "phased"),
                                serial_do_sample=bool(getattr(args, "serial_gen_do_sample", False)),
                                serial_temperature=float(getattr(args, "serial_gen_temperature", 0.2)),
                                serial_top_p=float(getattr(args, "serial_gen_top_p", 0.9)),
                                serial_repetition_penalty=float(getattr(args, "serial_gen_repetition_penalty", 1.0)),
                                serial_max_new_tokens=int(getattr(args, "serial_gen_max_new_tokens", 1024)),
                                serial_prompt_max_chars=int(getattr(args, "serial_prompt_max_chars", 0)),
                                serial_diff_max_chars=int(getattr(args, "serial_diff_max_chars", 4000)),
                                serial_diff_context_lines=int(getattr(args, "serial_diff_context_lines", 3)),
                                serial_cache=serial_cache,
                                serial_ref_max_attempts_per_task=int(getattr(args, "serial_ref_max_attempts_per_task", 5)),
                                serial_use_solution_scalar=bool(getattr(args, "serial_use_solution_scalar", False)),
                                serial_solution_scalar=str(p.get("solution_scalar", "") or ""),
                                serial_audited_nohelper_scalar=str(
                                    p.get(str(getattr(args, "serial_audited_nohelper_scalar_field", "audited_nohelper_scalar") or "audited_nohelper_scalar"), "")
                                    or ""
                                ),
                                serial_entrypoint_scalar=str(p.get("entrypoint_scalar", "") or ""),
                                serial_audited_nohelper_entrypoint=str(
                                    p.get(str(getattr(args, "serial_audited_nohelper_entrypoint_field", "audited_nohelper_scalar_entrypoint") or "audited_nohelper_scalar_entrypoint"), "")
                                    or p.get("entrypoint_simd", "")
                                    or func_name
                                    or ""
                                ),
                                serial_audited_nohelper_source=str(
                                    p.get("audited_nohelper_scalar_source", "")
                                    or p.get("audited_nohelper_scalar_source_label", "")
                                    or str(getattr(args, "serial_audited_nohelper_scalar_field", "audited_nohelper_scalar") or "audited_nohelper_scalar")
                                ),
                                serial_entrypoint_simd=str(p.get("entrypoint_simd", "") or func_name or ""),
                                serial_eval_intrinsic=str(getattr(args, "serial_eval_intrinsic", "scalar") or "scalar"),
                                serial_mismatch_harness=bool(getattr(args, "serial_mismatch_harness", False)),
                                serial_mismatch_max_new_tokens=int(getattr(args, "serial_mismatch_max_new_tokens", 512)),
                                serial_mismatch_prompt_max_chars=int(getattr(args, "serial_mismatch_prompt_max_chars", 20000)),
                                serial_feedback_style=str(getattr(args, "serial_feedback_style", "pseudocode") or "pseudocode"),
                                serial_pseudocode_max_chars=int(getattr(args, "serial_pseudocode_max_chars", 6000) or 0),
                                serial_ast_bootstrap_pseudocode=serial_ast_bootstrap_pseudocode,
                                serial_ast_bootstrap_style=serial_ast_bootstrap_style,
                                serial_ast_bootstrap_pattern=serial_ast_bootstrap_pattern,
                                serial_ast_bootstrap_reason=serial_ast_bootstrap_reason,
                                serial_ast_bootstrap_source=serial_ast_bootstrap_source,
                                static_verifier=bool(getattr(args, "static_verifier", False)),
                                static_verifier_script=str(getattr(args, "static_verifier_script", "") or ""),
                                static_verifier_timeout=int(getattr(args, "static_verifier_timeout", 20) or 20),
                                remote_cpp17_compile_gate=bool(getattr(args, "remote_cpp17_compile_gate", False)),
                                remote_cpp17_compile_gate_mode=str(getattr(args, "remote_cpp17_compile_gate_mode", "off") or "off"),
                                remote_cpp17_compile_gate_compiler=str(getattr(args, "remote_cpp17_compile_gate_compiler", "clang++") or "clang++"),
                                remote_cpp17_compile_gate_cflags=str(getattr(args, "remote_cpp17_compile_gate_cflags", "") or ""),
                                remote_cpp17_compile_gate_timeout=int(getattr(args, "remote_cpp17_compile_gate_timeout", 20) or 20),
                                remote_cpp17_compile_gate_max_repair_iters=int(getattr(args, "remote_cpp17_compile_gate_max_repair_iters", 2) or 2),
                                nl_description_ds_r1=str(p.get("nl_description_ds_r1", "") or ""),
                                serial_c_code=str(p.get("serial_c_code", "") or ""),
                                test_harness_code=str(p.get("test_harness_code", "") or ""),
                                source_type=str(p.get("source_type", "") or ""),
                                problem_type=str(p.get("type", "") or ""),
                                problem_subtype=str(p.get("subtype", "") or ""),
                                source_name=str(p.get("source_name", "") or ""),
                                remote_perf_precheck=bool(getattr(args, "remote_perf_precheck", False)),
                                remote_perf_precheck_optimization=str(getattr(args, "remote_perf_precheck_optimization", "-O1") or "-O1"),
                                remote_perf_precheck_n_reputation=int(getattr(args, "remote_perf_precheck_n_reputation", 1) or 1),
                                remote_perf_precheck_n_workers=int(getattr(args, "remote_perf_precheck_n_workers", 1) or 1),
                                remote_perf_precheck_timeout=float(getattr(args, "remote_perf_precheck_timeout", 30.0) or 30.0),
                                remote_perf_precheck_python=str(getattr(args, "remote_perf_precheck_python", "python3") or "python3"),
                                remote_perf_precheck_repo_root=str(getattr(args, "remote_perf_precheck_repo_root", "") or ""),
                                semantic_plan=(p.get("semantic_plan") if isinstance(p.get("semantic_plan"), dict) else None),
                                prompt_max_chars=args.remote_prompt_max_chars,
                            )

                        completion = normalize_completion_snippet(completion, func_name)
                        if intrinsic == "SVE":
                            completion, _pp_info = postprocess_sve_common_fixes(completion)
                        wf.write(json.dumps({"task_id": task_id, "completion": completion}, ensure_ascii=False) + "\n")
                        wf.flush()

                        dlg_set_stage("final_completion")
                        dlg_record(
                            system_text="",
                            user_text="",
                            assistant_text=completion,
                            gen_args={"note": "final completion after clean/repairs"},
                        )

                        rec["ok"] = True
                        rec["completion_len"] = len(completion)
                        if name_info is not None:
                            rec["name_invalid_before"] = name_info.get("invalid_before_count")
                            rec["name_invalid_after"] = name_info.get("invalid_after_count")
                            rec["name_fix_iters"] = name_info.get("iters")
                        if shape_info is not None:
                            rec["shape_fix_iters"] = shape_info.get("iters")
                            rec["shape_mismatch_before"] = shape_info.get("mismatch_before_count")
                            rec["shape_mismatch_after"] = shape_info.get("mismatch_after_count")
                            if shape_info.get("skipped"):
                                rec["shape_fix_skipped"] = shape_info["skipped"]
                        if remote_info is not None:
                            hist = remote_info.get("history", [])
                            rec["remote_enabled"] = True
                            rec["remote_rounds_used"] = max(0, len(hist) - 1)
                            rec["serial_ref"] = remote_info.get("serial_ref")
                            serial_ref_info = remote_info.get("serial_ref") or {}
                            rec["serial_bootstrap_mode"] = serial_ref_info.get("bootstrap_mode")
                            rec["serial_bootstrap_stage"] = serial_ref_info.get("serial_bootstrap_stage")
                            rec["serial_bootstrap_reason"] = serial_ref_info.get("serial_bootstrap_reason")
                            rec["serial_bootstrap_source"] = serial_ref_info.get("serial_bootstrap_source")
                            rec["serial_bootstrap_attempt_source"] = serial_ref_info.get("serial_bootstrap_attempt_source")
                            rec["serial_ref_validated"] = bool(serial_ref_info.get("serial_ref_validated", False))
                            rec["serial_ref_failure_kind"] = serial_ref_info.get("serial_ref_failure_kind")
                            rec["serial_ref_budget_consumed"] = bool(serial_ref_info.get("serial_ref_budget_consumed", False))
                            rec["serial_feedback_style_requested"] = serial_ref_info.get("feedback_style_requested")
                            rec["serial_feedback_style_effective"] = (
                                serial_ref_info.get("feedback_style_effective")
                                or serial_ref_info.get("feedback_style")
                            )
                            serial_route_info = serial_ref_info.get("routing") or {}
                            if isinstance(serial_route_info, dict):
                                rec["serial_feedback_route_source"] = serial_route_info.get("source")
                                rec["serial_feedback_route_pattern"] = serial_route_info.get("pattern")
                                rec["serial_feedback_route_reason"] = serial_route_info.get("reason")
                                rec["serial_feedback_route_has_scalar_reference"] = bool(
                                    serial_route_info.get("has_scalar_reference", False)
                                )
                            rec["pre_remote_selection"] = remote_info.get("pre_remote_selection")
                            rec["remote_scoring"] = remote_info.get("scoring")
                            rec["remote_best_score_parts"] = remote_info.get("best_score_parts")
                            rec["remote_last_score_parts"] = remote_info.get("last_score_parts")
                            verifier_info = remote_info.get("static_verifier", {}) or {}
                            remote_gate_info = remote_info.get("remote_cpp17_compile_gate", {}) or {}
                            remote_perf_info = remote_info.get("remote_perf_precheck", {}) or {}
                            pre_remote_verifier = verifier_info.get("pre_remote_selected", {}) or {}
                            final_verifier = verifier_info.get("final", {}) or {}
                            compile_risk_tags = final_verifier.get("compile_risk_tags") or []
                            semantic_risk_tags = final_verifier.get("semantic_risk_tags") or []
                            repair_payload = final_verifier.get("repair_prompt_payload") or {}
                            pre_remote_payload = pre_remote_verifier.get("repair_prompt_payload") or {}
                            symbol_closure_targets = (
                                repair_payload.get("symbol_closure_targets")
                                if isinstance(repair_payload, dict)
                                else {}
                            )
                            remote_compile_diagnostics = (
                                final_verifier.get("remote_compile_diagnostics")
                                if isinstance(final_verifier.get("remote_compile_diagnostics"), dict)
                                else {}
                            )
                            compile_risk_tag_names = []
                            for item in compile_risk_tags:
                                if isinstance(item, dict):
                                    tag = str(item.get("tag") or "").strip()
                                else:
                                    tag = str(item).strip()
                                if tag:
                                    compile_risk_tag_names.append(tag)
                            semantic_risk_tag_names = []
                            for item in semantic_risk_tags:
                                if isinstance(item, dict):
                                    tag = str(item.get("tag") or "").strip()
                                else:
                                    tag = str(item).strip()
                                if tag:
                                    semantic_risk_tag_names.append(tag)
                            verifier_focus_areas = []
                            verifier_must_fix_tags = []
                            if isinstance(repair_payload, dict):
                                for item in repair_payload.get("focus_areas") or []:
                                    tag = str(item).strip()
                                    if tag:
                                        verifier_focus_areas.append(tag)
                                for item in repair_payload.get("must_fix_tags") or []:
                                    tag = str(item).strip()
                                    if tag:
                                        verifier_must_fix_tags.append(tag)
                            rec["static_verifier_enabled"] = bool(verifier_info.get("enabled", False))
                            rec["static_verifier_compile_risk_count"] = len(compile_risk_tags)
                            rec["static_verifier_semantic_risk_count"] = len(semantic_risk_tags)
                            rec["static_verifier_compile_risk_tags"] = compile_risk_tag_names
                            rec["static_verifier_semantic_risk_tags"] = semantic_risk_tag_names
                            rec["static_verifier_focus_areas"] = verifier_focus_areas
                            rec["static_verifier_must_fix_tags"] = verifier_must_fix_tags
                            rec["static_verifier_symbol_closure_targets"] = symbol_closure_targets
                            rec["static_verifier_expected_signature_line"] = (
                                repair_payload.get("expected_signature_line")
                                if isinstance(repair_payload, dict)
                                else None
                            )
                            rec["static_verifier_actual_signature_line"] = (
                                repair_payload.get("actual_signature_line")
                                if isinstance(repair_payload, dict)
                                else None
                            )
                            signature_check_final = (
                                repair_payload.get("signature_check")
                                if isinstance(repair_payload, dict) and isinstance(repair_payload.get("signature_check"), dict)
                                else {}
                            )
                            rec["static_verifier_signature_match_final"] = (
                                signature_check_final.get("match")
                                if isinstance(signature_check_final, dict)
                                else None
                            )
                            rec["static_verifier_signature_mismatches_final"] = (
                                list(signature_check_final.get("mismatches") or [])
                                if isinstance(signature_check_final, dict)
                                else []
                            )
                            rec["static_verifier_action"] = final_verifier.get("repair_action")
                            rec["static_verifier_controller_action_pre_remote"] = (
                                pre_remote_payload.get("controller_action")
                                if isinstance(pre_remote_payload, dict)
                                else None
                            )
                            rec["static_verifier_controller_action_final"] = (
                                repair_payload.get("controller_action")
                                if isinstance(repair_payload, dict)
                                else None
                            )
                            rec["static_verifier_controller_action"] = (
                                repair_payload.get("controller_action") if isinstance(repair_payload, dict) else None
                            )
                            rec["static_verifier_primary_focus_area"] = (
                                repair_payload.get("primary_focus_area") if isinstance(repair_payload, dict) else None
                            )
                            rec["remote_cpp17_compile_gate_enabled"] = bool(remote_gate_info.get("enabled", False))
                            rec["remote_cpp17_compile_gate_mode"] = remote_gate_info.get("mode")
                            rec["remote_cpp17_compile_gate_blocked"] = bool(remote_gate_info.get("blocked", False))
                            rec["remote_cpp17_compile_gate_blocked_rounds"] = list(remote_gate_info.get("blocked_rounds") or [])
                            rec["remote_cpp17_compile_gate_recovered_rounds"] = list(remote_gate_info.get("recovered_rounds") or [])
                            rec["remote_cpp17_compile_gate_attempted_rounds"] = len(remote_gate_info.get("history") or [])
                            rec["remote_cpp17_compile_gate_failed_rounds"] = sum(
                                1
                                for item in (remote_gate_info.get("history") or [])
                                if isinstance(item, dict) and not bool(item.get("compile_ok", False))
                            )
                            final_gate = remote_gate_info.get("final") if isinstance(remote_gate_info.get("final"), dict) else {}
                            rec["remote_cpp17_compile_gate_final_reason"] = final_gate.get("reason")
                            rec["remote_cpp17_compile_gate_final_compile_ok"] = final_gate.get("compile_ok")
                            rec["remote_cpp17_compile_gate_final_diagnostics"] = final_gate.get("diagnostics")
                            rec["remote_perf_precheck_enabled"] = bool(remote_perf_info.get("enabled", False))
                            rec["remote_perf_precheck_attempted_rounds"] = len(remote_perf_info.get("history") or [])
                            rec["remote_perf_precheck_availability_errors"] = list(remote_perf_info.get("availability_errors") or [])
                            final_perf = remote_perf_info.get("final") if isinstance(remote_perf_info.get("final"), dict) else {}
                            rec["remote_perf_precheck_final_reason"] = final_perf.get("reason")
                            rec["remote_perf_precheck_detail_reason"] = final_perf.get("detail_reason")
                            rec["remote_perf_precheck_final_ok"] = final_perf.get("perf_ok")
                            rec["remote_perf_precheck_candidate_failed"] = final_perf.get("candidate_failed")
                            rec["remote_perf_precheck_speedup"] = final_perf.get("speedup")
                            rec["remote_perf_precheck_speedup_median"] = final_perf.get("speedup_median")
                            rec["remote_perf_precheck_speedup_mean"] = final_perf.get("speedup_mean")
                            final_res = remote_info.get("final", {}) or {}
                            rec["remote_compile_ok"] = final_res.get("compile_ok")
                            rec["remote_run_ok"] = final_res.get("run_ok")
                            rec["remote_reason"] = final_res.get("reason")
                            rec["remote_compile_diagnostics"] = remote_compile_diagnostics
                            rec["remote_compile_undeclared_identifier_count"] = len(
                                remote_compile_diagnostics.get("undeclared_identifiers") or []
                            )
                            rec["remote_compile_missing_helper_count"] = len(
                                remote_compile_diagnostics.get("missing_helper_symbols") or []
                            )
                            rec["remote_compile_missing_index_helper_count"] = len(
                                remote_compile_diagnostics.get("missing_index_symbols") or []
                            )
                        rec["pre_explain_enabled"] = bool(args.pre_explain and intrinsic.upper() == "SVE")
                        rec["pre_explain_chars"] = len(expanded_spec_text) if expanded_spec_text else 0

                        if args.save_intermediate and case_dir is not None:
                            (case_dir / "_system_prompt.txt").write_text(system_prompt, encoding="utf-8")
                            (case_dir / "_user_prompt_prefix.txt").write_text(user_prompt, encoding="utf-8")
                            (case_dir / "_user_prompt_used.txt").write_text(user_prompt_gen, encoding="utf-8")
                            (case_dir / "_raw_model_output.txt").write_text(raw, encoding="utf-8")
                            (case_dir / "_completion_final.txt").write_text(completion, encoding="utf-8")
                            if name_info is not None:
                                (case_dir / "_name_fix_info.json").write_text(json.dumps(name_info, ensure_ascii=False, indent=2), encoding="utf-8")
                            if shape_info is not None:
                                (case_dir / "_shape_fix_info.json").write_text(json.dumps(shape_info, ensure_ascii=False, indent=2), encoding="utf-8")
                            if remote_info is not None:
                                (case_dir / "_remote_feedback_info.json").write_text(json.dumps(remote_info, ensure_ascii=False, indent=2), encoding="utf-8")
                            if expanded_spec_text.strip():
                                (case_dir / "_pre_explain_output.txt").write_text(expanded_spec_text, encoding="utf-8")

                    except Exception as e:
                        rec["err"] = repr(e)
                        rec["trace"] = traceback.format_exc()

                        dlg_set_stage("exception")
                        dlg_record(
                            system_text="",
                            user_text="",
                            assistant_text="[EXCEPTION]\n" + repr(e) + "\n\n" + traceback.format_exc(),
                            gen_args={"note": "exception during generation/repair"},
                        )

                        try:
                            wf.write(json.dumps({"task_id": task_id, "completion": ""}, ensure_ascii=False) + "\n")
                            wf.flush()
                        except Exception:
                            pass
                    finally:
                        dlg_set_capture(None)

                        dialogue_payload = {
                            "task_id": task_id,
                            "sample_idx": sidx,
                            "rank": rank,
                            "intrinsic": intrinsic,
                            "turns": dialogue_turns,
                        }

                        dump_path: Optional[Path] = None
                        if args.dialogue_dump_dir:
                            dd = Path(args.dialogue_dump_dir) / f"rank{rank}" / task_id / f"sample{sidx}"
                            dd.mkdir(parents=True, exist_ok=True)
                            dump_path = dd / "_dialogue_full.json"
                        elif args.save_intermediate and case_dir is not None:
                            dump_path = case_dir / "_dialogue_full.json"

                        if dump_path is not None:
                            dump_path.write_text(
                                json.dumps(dialogue_payload, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                                errors="replace",
                            )

                        if df is not None:
                            df.write(json.dumps(dialogue_payload, ensure_ascii=False) + "\n")
                            df.flush()

                        rec["elapsed_s"] = time.time() - t0
                        if lf is not None:
                            lf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            lf.flush()
                        progress_done_local += 1
                        if pbar is not None:
                            pbar.update(1)
                        if progress_print_every > 0:
                            should_print = (
                                (progress_done_local % progress_print_every) == 0
                                or progress_done_local == total_local
                                or not rec.get("ok", False)
                            )
                            if should_print:
                                hist = []
                                if isinstance(remote_info, dict):
                                    hist = remote_info.get("history", []) or []
                                emit_progress(
                                    "sample_done",
                                    task_id=task_id,
                                    task_local_idx=task_local_idx,
                                    sample_idx=sidx + 1,
                                    status="ok" if rec.get("ok", False) else "err",
                                    elapsed_s=float(rec.get("elapsed_s", 0.0) or 0.0),
                                    remote_rounds_used=max(0, len(hist) - 1),
                                    remote_reason=str(rec.get("remote_reason") or ""),
	                                )

                if dynamic_work_queue_enabled and work_queue_state_path is not None and work_queue_lock_path is not None:
                    mark_dynamic_work_item_done(
                        state_path=work_queue_state_path,
                        lock_path=work_queue_lock_path,
                        fingerprint=work_queue_fingerprint,
                        item_key=work_item_key,
                        rank=rank,
                        ok=True,
                        samples_written=int(args.n_samples),
                    )
                    task_completed_for_queue = True

            except Exception as _e_task:
                _task_trace = traceback.format_exc()
                print(f"[TASK_ERROR] rank={rank} id={task_id} err={_e_task!r}")
                # Keep output shape: write empty completion for each sample and log the error
                for _sidx in range(int(args.n_samples)):
                    wf.write(json.dumps({"task_id": task_id, "completion": ""}, ensure_ascii=False) + "\n")
                    wf.flush()
                    if lf is not None:
                        lf.write(json.dumps({
                            "task_id": task_id,
                            "sample_idx": _sidx,
                            "rank": rank,
                            "ok": False,
                            "llm_backend": args.llm_backend,
                            "fatal_task_error": True,
                            "err": repr(_e_task),
                            "trace": _task_trace,
                        }, ensure_ascii=False) + "\n")
                        lf.flush()
                    progress_done_local += 1
                    if pbar is not None:
                        pbar.update(1)
                    emit_progress(
                        "sample_done",
                        task_id=task_id,
	                        task_local_idx=task_local_idx,
	                        sample_idx=_sidx + 1,
	                        status="task_error",
	                    )
                if dynamic_work_queue_enabled and work_queue_state_path is not None and work_queue_lock_path is not None and not task_completed_for_queue:
                    mark_dynamic_work_item_done(
                        state_path=work_queue_state_path,
                        lock_path=work_queue_lock_path,
                        fingerprint=work_queue_fingerprint,
                        item_key=work_item_key,
                        rank=rank,
                        ok=False,
                        samples_written=int(args.n_samples),
                    )
                continue
        if pbar is not None:
            pbar.close()

    except Exception as _e_fatal:
        fatal_error = traceback.format_exc()
        print(f"[FATAL] rank{rank} aborted with unhandled error: {_e_fatal!r}")
        print(fatal_error)
    finally:
        wf.close()
        try:
            # 原子替换，保证 shard 要么完整出现，要么不出现
            os.replace(tmp_out_path, out_path)
        except Exception as e:
            print(f"[WARN] rank{rank} cannot move tmp shard -> final: {e}")
        if lf is not None:
            lf.close()
        if df is not None:
            df.close()

    if distributed:
        # 1) each rank writes a done marker AFTER all files are closed
        done_payload = {
            "rank": rank,
            "world_size": world_size,
            "pid": os.getpid(),
            "ts": time.time(),
            "out_path": str(out_path),
            "meta_log_path": str(meta_log_path) if meta_log_path is not None else "",
            "dialogue_log_path": str(dialogue_log_path) if dialogue_log_path is not None else "",
            "fatal_error": fatal_error,
            "ok": (fatal_error == ""),
        }
        write_done_marker(base_out, rank, num_shards, done_payload)

        # 2) rank0 waits and merges
        if rank == 0:
            timeout_s = int(getattr(args, "final_merge_timeout_s", 0) or 0)
            poll_s = float(getattr(args, "final_merge_poll_s", 1.0) or 1.0)
            print_every_s = float(getattr(args, "final_merge_print_every_s", 30.0) or 30.0)

            ok_all, missing = wait_for_done_markers(
                base_out,
                num_shards,
                timeout_s=timeout_s,
                poll_s=poll_s,
                print_every_s=print_every_s,
            )
            if not ok_all:
                print(f"[WARN] merge timeout. Missing ranks: {missing}. "
                    f"Will merge existing shard files and KEEP shard files for recovery.")

            merge_shard_outputs(base_out, num_shards=num_shards)

            # merge logs (best-effort; keep if some ranks missing)
            if args.log_file:
                final_log_path = Path(args.log_file)
                tmp_final = final_log_path.with_name(final_log_path.name + f".merge_tmp_{os.getpid()}")
                with tmp_final.open("w", encoding="utf-8") as out_f:
                    for rnk in range(world_size):
                        p = final_log_path.with_name(final_log_path.name + f".rank{rnk}.jsonl")
                        if p.exists():
                            out_f.write(p.read_text(encoding="utf-8", errors="replace"))
                tmp_final.replace(final_log_path)

            if args.dialogue_log_file:
                final_dlg_path = Path(args.dialogue_log_file)
                tmp_final = final_dlg_path.with_name(final_dlg_path.name + f".merge_tmp_{os.getpid()}")
                with tmp_final.open("w", encoding="utf-8") as out_f:
                    for rnk in range(world_size):
                        p = final_dlg_path.with_name(final_dlg_path.name + f".rank{rnk}.jsonl")
                        if p.exists():
                            out_f.write(p.read_text(encoding="utf-8", errors="replace"))
                tmp_final.replace(final_dlg_path)

            # Only delete shard files if ALL ranks finished (otherwise you会丢失后续补齐的机会)
            if ok_all and (not args.keep_shard_files):
                for sid in range(num_shards):
                    part = shard_output_path(base_out, shard_id=sid, num_shards=num_shards)
                    try:
                        part.unlink()
                    except FileNotFoundError:
                        pass
                # also delete done markers
                for r in range(num_shards):
                    try:
                        _done_marker_path(base_out, r, num_shards).unlink()
                    except FileNotFoundError:
                        pass

            print(f"[OK] wrote merged: {base_out} (tasks={len(problems)} shards={num_shards} samples_per_task={args.n_samples})")

    else:
        if num_shards == 1:
            print(f"[OK] wrote: {out_path} (tasks={len(problems)} samples_per_task={args.n_samples})")
        else:
            print(f"[OK] wrote shard: {out_path} (shard {shard_id+1}/{num_shards}, shard_tasks={len(my_problems)}, total_tasks={len(problems)})")


if __name__ == "__main__":
    main()
