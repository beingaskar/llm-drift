from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Tuple

_CALL_RE = re.compile(r"^output\.(\w+)\((.*)\)$")

_BUILTINS = {"is_valid_json", "contains", "not_contains", "min_length", "max_length"}


class AssertionParseError(Exception):
    pass


@dataclass
class AssertionResult:
    expression: str
    passed: bool
    error: str = ""


def _parse(expression: str) -> Tuple[str, str]:
    m = _CALL_RE.match(expression.strip())
    if not m:
        raise AssertionParseError(f"Cannot parse assertion: {expression!r}")
    return m.group(1), m.group(2).strip()


def _coerce(raw: str):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip("'\"")


def _evaluate(output: str, method: str, arg) -> Tuple[bool, str]:
    if method not in _BUILTINS:
        raise AssertionParseError(f"Unknown assertion method: {method!r}")
    if method == "is_valid_json":
        try:
            json.loads(output.strip())
            return True, ""
        except json.JSONDecodeError:
            return False, "output is not valid JSON"
    if method == "contains":
        ok = str(arg) in output
        return ok, "" if ok else f"output does not contain {arg!r}"
    if method == "not_contains":
        ok = str(arg) not in output
        return ok, "" if ok else f"output unexpectedly contains {arg!r}"
    if method == "min_length":
        n = int(arg)
        ok = len(output) >= n
        return ok, "" if ok else f"output length {len(output)} < {n}"
    if method == "max_length":
        n = int(arg)
        ok = len(output) <= n
        return ok, "" if ok else f"output length {len(output)} > {n}"
    raise AssertionParseError(f"Unknown assertion method: {method!r}")  # unreachable


class AssertionRunner:
    def run(self, output: str, assertions: List[str]) -> List[AssertionResult]:
        results = []
        for expr in assertions:
            method, raw_arg = _parse(expr)
            if method not in _BUILTINS:
                raise AssertionParseError(f"Unknown assertion method: {method!r}")
            passed, error = _evaluate(output, method, _coerce(raw_arg))
            results.append(AssertionResult(expression=expr, passed=passed, error=error))
        return results
