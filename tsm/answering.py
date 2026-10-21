"""Answering from recalled memories with tools for counting and arithmetic.

The reading prompt (``tsm.reading``) asks the model to count, total and work
out dates itself. Of the questions it still gets wrong with all the facts in
front of it, most are exactly that: a total that misses an item or counts
one twice, a subtraction, the number of days between two dates. A model is
reliable at finding the numbers in its notes and unreliable at the
arithmetic, so ``answer_with_tools`` gives it four deterministic tools:

- ``calculate``: an arithmetic expression (``+ - * / // % **``, parentheses,
  ``round``, ``abs``, ``min``, ``max``, ``sum``),
- ``days_between``: the days, weeks and months between two dates,
- ``add_days``: the date a number of days before or after another,
- ``count_distinct``: how many different items a list names.

The memories, the steps and the ``Answer:`` line are those of
``tsm.reading.answer_prompt``. The model calls tools as it needs them (any
OpenAI-compatible chat client that supports tool calls), and its last reply
is read with ``final_answer``. The tools are plain Python and are tested on
their own.
"""

import ast
import json
import logging
import math
import operator
import re
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence

from ._retry import call_with_retries
from .reading import ANSWER_SYSTEM, Memory, answer_prompt, final_answer

logger = logging.getLogger("tsm.answering")

TOOLS_NOTE = (
    "\nYou have tools: calculate (arithmetic), days_between (the time between two "
    "dates), add_days (a date some days before or after another) and count_distinct "
    "(how many different items a list names). Whenever the answer needs a sum, a "
    "difference, an average, a count, or the time between two dates, take the values "
    "from the notes and call the tool instead of working it out yourself. Then finish "
    "as above."
)

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression, e.g. '(120 + 85) * 2 / 3'. "
                       "Supports + - * / // % **, parentheses, round, abs, min, max, sum.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "days_between",
        "description": "The time from start to end (YYYY-MM-DD): days, weeks and months.",
        "parameters": {"type": "object", "properties": {
            "start": {"type": "string"}, "end": {"type": "string"}},
            "required": ["start", "end"]}}},
    {"type": "function", "function": {
        "name": "add_days",
        "description": "The date that is `days` after `date` (YYYY-MM-DD); negative for before.",
        "parameters": {"type": "object", "properties": {
            "date": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["date", "days"]}}},
    {"type": "function", "function": {
        "name": "count_distinct",
        "description": "How many different items the list names, after ignoring case, "
                       "punctuation and repeats. Give one short name per item.",
        "parameters": {"type": "object", "properties": {
            "items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"]}}},
]

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
           ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {"round": round, "abs": abs, "min": min, "max": max, "sum": sum}
_DAY = re.compile(r"^\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})")


def _number(value) -> str:
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.6g}" if abs(value) >= 1e6 else str(round(value, 4))
    return str(value)


def calculate(expression: str) -> str:
    """The value of an arithmetic expression, or an error message (never an
    exception: the model reads the result)."""
    def value(node):
        if isinstance(node, ast.Expression):
            return value(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = value(node.left), value(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BINARY[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](value(node.operand))
        if isinstance(node, (ast.List, ast.Tuple)):
            return [value(item) for item in node.elts]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCTIONS and not node.keywords):
            return _FUNCTIONS[node.func.id](*(value(arg) for arg in node.args))
        raise ValueError("only numbers, + - * / // % **, parentheses and "
                         "round/abs/min/max/sum are allowed")
    try:
        # "1,200" is a number; "max(3, 5)" has two arguments.
        text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", (expression or "").replace("$", "")).strip()
        if len(text) > 300:
            return "error: expression too long"
        result = value(ast.parse(text, mode="eval"))
        if isinstance(result, list) or not math.isfinite(float(result)):
            return "error: not a single finite number"
        return _number(result)
    except ZeroDivisionError:
        return "error: division by zero"
    except (SyntaxError, ValueError, TypeError, OverflowError) as e:
        return f"error: {e}"


def _parse_day(text: str) -> Optional[date]:
    match = _DAY.match(text or "")
    if not match:
        return None
    try:
        return date(*(int(g) for g in match.groups()))
    except ValueError:
        return None


def days_between(start: str, end: str) -> str:
    """The time from ``start`` to ``end``, in days and in weeks and months."""
    first, last = _parse_day(start), _parse_day(end)
    if first is None or last is None:
        return "error: give both dates as YYYY-MM-DD"
    days = (last - first).days
    months = (last.year - first.year) * 12 + (last.month - first.month)
    if last.day < first.day:
        months -= 1 if days >= 0 else 0
    weeks, rest = divmod(abs(days), 7)
    sign = "-" if days < 0 else ""
    return (f"{days} days ({sign}{weeks} weeks and {rest} days; "
            f"{months} whole months; {days / 365.25:.2f} years)")


def add_days(day: str, days) -> str:
    start = _parse_day(day)
    try:
        offset = int(days)
    except (TypeError, ValueError):
        return "error: days must be a whole number"
    if start is None:
        return "error: give the date as YYYY-MM-DD"
    try:
        result = start + timedelta(days=offset)
    except OverflowError:
        return "error: out of range"
    return f"{result:%Y-%m-%d} ({result:%A})"


def count_distinct(items: Sequence[str]) -> str:
    if not isinstance(items, (list, tuple)):
        return "error: give a list of items"
    seen: Dict[str, str] = {}
    for item in items:
        key = " ".join(re.findall(r"[a-z0-9]+", str(item).lower()))
        if key and key not in seen:
            seen[key] = str(item).strip()
    return f"{len(seen)} distinct: " + "; ".join(seen.values())


def run_tool(name: str, arguments: str) -> str:
    """Run the tool a model asked for with its JSON arguments."""
    try:
        args = json.loads(arguments or "{}")
    except ValueError:
        return "error: arguments are not valid JSON"
    if not isinstance(args, dict):
        return "error: arguments must be an object"
    try:
        if name == "calculate":
            return calculate(args.get("expression", ""))
        if name == "days_between":
            return days_between(args.get("start", ""), args.get("end", ""))
        if name == "add_days":
            return add_days(args.get("date", ""), args.get("days"))
        if name == "count_distinct":
            return count_distinct(args.get("items"))
    except Exception as e:  # noqa: BLE001 — the model reads the failure
        return f"error: {type(e).__name__}"
    return f"error: no tool named {name!r}"


def answer_with_tools(client, model: str, question: str, memories: Sequence[Memory],
                      today: Optional[str] = None, max_rounds: int = 4,
                      max_tokens: int = 800, max_retries: int = 6,
                      request_kwargs: Optional[dict] = None) -> Dict:
    """Ask ``model`` (through an OpenAI-compatible ``client``) to answer
    ``question`` from ``memories``, letting it call the tools above.

    Returns ``{"answer", "reply", "requests", "tool_calls", "input_tokens",
    "output_tokens"}``: ``answer`` is ``final_answer`` of the last reply,
    ``tool_calls`` the ``(name, arguments, result)`` of every call. After
    ``max_rounds`` replies that asked for tools, one more is requested with
    tools switched off, so an answer always comes back.
    """
    messages: List[Dict] = answer_prompt(question, memories, today)
    messages[0] = {"role": "system", "content": ANSWER_SYSTEM + TOOLS_NOTE}
    stats = {"requests": 0, "tool_calls": [], "input_tokens": 0, "output_tokens": 0}
    reply = ""
    for round_number in range(max_rounds + 1):
        last = round_number == max_rounds
        kwargs = {"model": model, "messages": messages, "temperature": 0.0,
                  "max_tokens": max_tokens, "tools": TOOL_SPECS,
                  "tool_choice": "none" if last else "auto", **(request_kwargs or {})}
        response = call_with_retries(lambda: client.chat.completions.create(**kwargs),
                                     "answering with tools", max_retries, logger)
        stats["requests"] += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            stats["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            stats["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        message = response.choices[0].message
        reply = message.content or ""
        calls = getattr(message, "tool_calls", None) or []
        if not calls or last:
            break
        messages.append({"role": "assistant", "content": reply, "tool_calls": [
            {"id": call.id, "type": "function",
             "function": {"name": call.function.name, "arguments": call.function.arguments}}
            for call in calls]})
        for call in calls:
            result = run_tool(call.function.name, call.function.arguments)
            stats["tool_calls"].append((call.function.name, call.function.arguments, result))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    stats["reply"] = reply
    stats["answer"] = final_answer(reply) or "NO ANSWER"
    return stats
