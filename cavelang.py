from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any, Callable

from textx import metamodel_from_file


RESERVED_WORDS = {
    "make", "say", "listen",
    "when", "else", "done",
    "repeat", "break", "continue",
    "add", "take", "smash", "split", "leftover",
    "to_num", "to_word",
    "same", "different", "bigger", "smaller", "at_least", "at_most",
    "and", "or",
}


class BreakException(Exception):
    """Internal signal used to exit a repeat loop."""


class ContinueException(Exception):
    """Internal signal used to skip to the next repeat-loop iteration."""


def strip_comments(source: str) -> str:
    """Remove // comments while preserving // inside strings."""
    return "\n".join(strip_comment_from_line(line) for line in source.splitlines())


def strip_comment_from_line(line: str) -> str:
    in_string = False
    escaped = False
    result: list[str] = []

    for i, ch in enumerate(line):
        if escaped:
            result.append(ch)
            escaped = False
        elif ch == "\\" and in_string:
            result.append(ch)
            escaped = True
        elif ch == '"':
            result.append(ch)
            in_string = not in_string
        elif ch == "/" and i + 1 < len(line) and line[i + 1] == "/" and not in_string:
            break
        else:
            result.append(ch)

    return "".join(result)


class Cavelang:
    """Interpreter for parsed Cavelang programs."""

    def __init__(self) -> None:
        self.vars: dict[str, Any] = {}

    def get_var(self, name: str) -> Any:
        if name not in self.vars:
            raise NameError(f"Variable '{name}' is not defined")
        return self.vars[name]

    def set_var(self, name: str, value: Any) -> None:
        if name in RESERVED_WORDS:
            raise NameError(f"'{name}' is a reserved word and cannot be used as a variable name")
        self.vars[name] = value

    def eval_expr(self, expr: Any) -> Any:
        node_type = expr.__class__.__name__

        if node_type == "Var":
            return self.get_var(expr.name)
        if node_type == "Number":
            return expr.value
        if node_type == "String":
            return self.eval_string(expr.value)
        if node_type == "PrefixOp":
            return self.eval_prefix_op(expr)
        if node_type == "HelperFunc":
            return self.eval_helper_func(expr)

        raise ValueError(f"Unknown expression type: {node_type}")

    def eval_string(self, value: str) -> str:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1] if len(value) >= 2 and value[0] == '"' and value[-1] == '"' else value

    def eval_prefix_op(self, expr: Any) -> Any:
        left = self.eval_expr(expr.left)
        right = self.eval_expr(expr.right)
        op = expr.op

        if op == "add":
            return self.add_values(left, right)
        if op == "take":
            return self.require_numbers(op, left, right, lambda a, b: a - b)
        if op == "smash":
            return self.require_numbers(op, left, right, lambda a, b: a * b)
        if op == "split":
            return self.require_numbers(op, left, right, self.safe_divide)
        if op == "leftover":
            return self.require_numbers(op, left, right, self.safe_modulo)

        raise ValueError(f"Unknown prefix operation: {op}")

    def add_values(self, left: Any, right: Any) -> Any:
        if type(left) is int and type(right) is int:
            return left + right
        return str(left) + str(right)

    def require_numbers(
        self,
        op: str,
        left: Any,
        right: Any,
        operation: Callable[[int, int], int],
    ) -> int:
        if type(left) is not int or type(right) is not int:
            raise TypeError(
                f"Operation '{op}' requires numbers, "
                f"got {type(left).__name__} and {type(right).__name__}"
            )
        return operation(left, right)

    @staticmethod
    def safe_divide(left: int, right: int) -> int:
        if right == 0:
            raise ValueError("Cannot split by zero")
        return left // right

    @staticmethod
    def safe_modulo(left: int, right: int) -> int:
        if right == 0:
            raise ValueError("Cannot take leftover by zero")
        return left % right

    def eval_helper_func(self, expr: Any) -> Any:
        value = self.eval_expr(expr.expr)

        if expr.name == "to_num":
            try:
                return int(value)
            except ValueError as exc:
                raise ValueError(f"Cannot convert {value!r} to a number") from exc

        if expr.name == "to_word":
            return str(value)

        raise ValueError(f"Unknown helper function: {expr.name}")

    def execute(self, stmt: Any) -> None:
        node_type = stmt.__class__.__name__

        if node_type == "Assignment":
            self.set_var(stmt.name, self.eval_expr(stmt.value))
        elif node_type == "Input":
            raw = input("> ")
            value = int(raw) if raw.lstrip("-").isdigit() else raw
            self.set_var(stmt.name, value)
        elif node_type == "Output":
            print(self.eval_expr(stmt.value))
        elif node_type == "Break":
            raise BreakException()
        elif node_type == "Continue":
            raise ContinueException()
        elif node_type == "IfStmt":
            self.execute_if(stmt)
        elif node_type == "WhileStmt":
            self.execute_while(stmt)
        else:
            raise ValueError(f"Unknown statement type: {node_type}")

    def execute_if(self, stmt: Any) -> None:
        if self.eval_condition(stmt.condition):
            self.execute_block(stmt.then_statements)
        else:
            self.execute_block(getattr(stmt, "else_statements", []))

    def execute_while(self, stmt: Any) -> None:
        while self.eval_condition(stmt.condition):
            try:
                self.execute_block(stmt.body_statements)
            except ContinueException:
                continue
            except BreakException:
                break

    def execute_block(self, statements: list[Any]) -> None:
        for stmt in statements:
            self.execute(stmt)

    def eval_condition(self, cond: Any) -> bool:
        left = self.eval_expr(cond.left)
        right = self.eval_expr(cond.right)
        result = self.compare_values(cond.op, left, right)

        bool_op = getattr(cond, "bool_op", None)
        right_cond = getattr(cond, "right_cond", None)

        if not bool_op or right_cond is None:
            return result

        if bool_op == "and":
            return result and self.eval_condition(right_cond)

        if bool_op == "or":
            return result or self.eval_condition(right_cond)

        raise ValueError(f"Unknown boolean operator: {bool_op}")

    def compare_values(self, op: str, left: Any, right: Any) -> bool:
        if op in ("same", "=="):
            return left == right
        if op in ("different", "!="):
            return left != right
        if op in ("bigger", ">"):
            return left > right
        if op in ("smaller", "<"):
            return left < right
        if op in ("at_least", ">="):
            return left >= right
        if op in ("at_most", "<="):
            return left <= right

        raise ValueError(f"Unknown condition operator: {op}")

    def interpret(self, model: Any) -> None:
        self.execute_block(model.statements)


def load_program_source(program_file: Path) -> str:
    return strip_comments(program_file.read_text(encoding="utf-8"))


def main() -> None:
    folder = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Run a Cavelang program.")
    parser.add_argument(
        "program",
        nargs="?",
        default=folder / "program1.ooga",
        type=Path,
        help="Path to a .ooga program file",
    )
    args = parser.parse_args()

    grammar_file = folder / "cavelang.tx"

    metamodel = metamodel_from_file(grammar_file)
    source = load_program_source(args.program)
    model = metamodel.model_from_str(source, file_name=str(args.program))

    Cavelang().interpret(model)


if __name__ == "__main__":
    main()