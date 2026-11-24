"""Answer extraction and equivalence checking for MATH-style problems.

``extract_answer`` takes the last ``\\boxed{...}`` (or ``\\fbox``) in a
completion, falling back to "The answer is X". ``is_equiv`` normalises
LaTeX in the style of Hendrycks et al. (2021) and also compares numerically.
If the optional ``math-verify`` package is installed (``pip install
passk-smooth[grading]``) it is consulted as a second opinion, which catches
symbolic equivalences that string normalisation misses.
"""

from __future__ import annotations

import re
from functools import lru_cache


def _last_braced(s: str, start: int) -> str | None:
    """Return the contents of the balanced {...} that opens at or after ``start``."""
    i = s.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j]
    return None


def extract_answer(text: str) -> str | None:
    for tag in ("\\boxed", "\\fbox"):
        pos = text.rfind(tag)
        if pos >= 0:
            after = text[pos + len(tag) :]
            if after.startswith(" ") and not after.lstrip().startswith("{"):
                # "\boxed 5" form
                tok = after.split()
                return tok[0].rstrip(".$") if tok else None
            got = _last_braced(text, pos)
            if got is not None:
                return got.strip()
    m = re.findall(r"answer is[:\s]*\$?([^\n$]+?)\$?(?:\.|\n|$)", text, flags=re.IGNORECASE)
    return m[-1].strip() if m else None


_REMOVE = [
    "\\left",
    "\\right",
    "\\!",
    "\\,",
    "\\;",
    "\\:",
    "\\displaystyle",
    "$",
    "\\%",
    "%",
    "^\\circ",
    "^{\\circ}",
    "\\text{ }",
    "\\mbox{ }",
]
_UNITS = ["degrees", "degree", "cm", "inches", "inch", "units", "unit", "meters", "dollars", "cents"]


def _fix_fracs(s: str) -> str:
    # \frac12 -> \frac{1}{2}, \frac1{2} -> \frac{1}{2}
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", s)
    s = re.sub(r"\\frac\{([^{}]+)\}(\d)", r"\\frac{\1}{\2}", s)
    return s


def normalize(ans: str) -> str:
    s = ans.strip()
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^{}]*)\}", r"\1", s)
    for r in _REMOVE:
        s = s.replace(r, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for u in _UNITS:
        s = re.sub(rf"(\d)\s*{u}\b", r"\1", s)
    s = _fix_fracs(s)
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)
    s = s.replace(" ", "")
    s = re.sub(r"^[a-zA-Z]\s*=", "", s)  # "x=5" -> "5"
    s = s.rstrip(".")
    if s.startswith("0.") is False and s.startswith("."):
        s = "0" + s
    # 1,000 -> 1000 (only digit groups)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    # simple a/b -> \frac{a}{b}
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m:
        s = f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}"
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def _to_float(s: str) -> float | None:
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+(?:\.\d+)?)\}\{(\d+(?:\.\d+)?)\}", s)
    try:
        if m:
            v = float(m.group(2)) / float(m.group(3))
            return -v if m.group(1) else v
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


@lru_cache(maxsize=1)
def _math_verify():  # pragma: no cover - optional dependency
    try:
        from math_verify import parse, verify

        return parse, verify
    except Exception:
        return None


def is_equiv(pred: str | None, gold: str, use_math_verify: bool = True) -> bool:
    if pred is None:
        return False
    a, b = normalize(pred), normalize(gold)
    if a == b:
        return True
    fa, fb = _to_float(a), _to_float(b)
    if fa is not None and fb is not None and abs(fa - fb) <= 1e-6 * max(1.0, abs(fb)):
        return True
    mv = _math_verify() if use_math_verify else None
    if mv is not None:  # pragma: no cover - optional dependency
        parse, verify = mv
        try:
            return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
        except Exception:
            return False
    return False


def grade(completion: str, gold: str, use_math_verify: bool = True) -> bool:
    return is_equiv(extract_answer(completion), gold, use_math_verify)
