"""Tools the agent can call.

A "tool" is just a Python function the LLM is allowed to invoke. LangChain's
`@tool` decorator turns a plain function into something the model can call,
using the function's name, type hints, and docstring as the schema the model
reads.

To add your own tool:
  1. Write a function with type hints + a clear docstring.
  2. Decorate it with `@tool`.
  3. Add it to the list returned by `get_tools()`.
The agent will automatically be able to use it — no other wiring needed.
"""

import ast
import operator
from datetime import datetime, timezone

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Safe calculator
# ---------------------------------------------------------------------------
# We never use Python's built-in eval() on model output (that would let the
# model run arbitrary code). Instead we parse the expression into an AST and
# only allow a small whitelist of math operators.

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted arithmetic AST node."""
    if isinstance(node, ast.Constant):  # numbers, e.g. 15
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp):  # e.g. 15 * 23
        op = _ALLOWED_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):  # e.g. -5
        op = _ALLOWED_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ValueError("Unsupported expression.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the result.

    Supports + - * / ** % and parentheses. Example: "15 * 23 + (4 / 2)".
    Use this whenever the user asks you to do math.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return f"{expression} = {result}"
    except Exception as exc:  # return the error so the model can react to it
        return f"Could not evaluate '{expression}': {exc}"


@tool
def current_time(timezone_name: str = "UTC") -> str:
    """Return the current date and time in UTC (ISO 8601).

    Use this whenever the user asks what time or date it is.
    The `timezone_name` argument is accepted for forward-compatibility but the
    starter kit always reports UTC to keep things dependency-free.
    """
    now = datetime.now(timezone.utc)
    return now.isoformat(timespec="seconds")


def get_tools() -> list:
    """Return the list of tools available to the agent.

    Register new tools here.
    """
    return [calculator, current_time]
