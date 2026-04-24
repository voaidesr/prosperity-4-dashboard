"""Human-readable diff for Prosperity algorithm submissions.

This tool compares two Python submission files and highlights changes that
matter during strategy review:

* hyperparameters and class constants
* method additions/removals/body changes
* state-management references
* order placement logic references
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STATE_KEYWORDS = {
    "traderData",
    "position",
    "own_trades",
    "market_trades",
    "order_depths",
    "observations",
    "timestamp",
    "json",
    "jsonpickle",
    "history",
    "hist",
    "anchor",
}

ORDER_KEYWORDS = {
    "Order",
    "orders",
    "buy_orders",
    "sell_orders",
    "best_bid",
    "best_ask",
    "limit",
    "LIMIT",
    "quantity",
}

RISK_KEYWORDS = {
    "POS_LIMITS",
    "position_limit",
    "limit",
    "conversions",
    "conversionObservations",
    "delta",
    "vega",
    "iv",
    "volatility",
    "hedge",
    "NormalDist",
    "cdf",
    "pdf",
    "broad_except",
}


@dataclass(frozen=True)
class FunctionSummary:
    """AST summary for one function or method."""

    name: str
    hash_value: str
    line: int
    returns: int
    order_calls: int
    state_refs: dict[str, int]
    order_refs: dict[str, int]


@dataclass(frozen=True)
class AlgorithmSummary:
    """Extracted logical summary of a Python submission."""

    path: Path
    constants: dict[str, Any]
    functions: dict[str, FunctionSummary]
    imports: list[str]
    product_literals: list[str]
    risk_refs: dict[str, int]
    broad_except_count: int


def stable_literal(node: ast.AST) -> Any:
    """Return a safe literal value or an AST expression string."""

    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return ast.unparse(node) if hasattr(ast, "unparse") else ast.dump(node)


def function_hash(node: ast.AST) -> str:
    """Hash a function body while ignoring line-number metadata."""

    normalized = ast.dump(node, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def collect_name_refs(node: ast.AST, keywords: set[str]) -> dict[str, int]:
    """Count references to strategy-relevant names or attributes."""

    counts = {keyword: 0 for keyword in keywords}
    for child in ast.walk(node):
        name = ""
        if isinstance(child, ast.Name):
            name = child.id
        elif isinstance(child, ast.Attribute):
            name = child.attr
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            for keyword in keywords:
                if keyword in child.value:
                    counts[keyword] += 1
            continue

        if name in counts:
            counts[name] += 1

    return {key: value for key, value in sorted(counts.items()) if value}


def count_order_calls(node: ast.AST) -> int:
    """Count explicit Order(...) constructor calls."""

    count = 0
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name) and func.id == "Order":
            count += 1
        elif isinstance(func, ast.Attribute) and func.attr == "Order":
            count += 1
    return count


def extract_imports(tree: ast.AST) -> list[str]:
    """Extract import statements in normalized text form."""

    imports = []
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node) if hasattr(ast, "unparse") else ast.dump(node))
    return imports


def extract_product_literals(tree: ast.AST) -> list[str]:
    """Extract product-like uppercase string literals."""

    products: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if value.isupper() and "_" in value and len(value) >= 4:
                products.add(value)
    return sorted(products)


def count_broad_excepts(tree: ast.AST) -> int:
    """Count bare ``except`` handlers that can hide strategy failures."""

    return sum(isinstance(node, ast.ExceptHandler) and node.type is None for node in ast.walk(tree))


def extract_constants(tree: ast.AST) -> dict[str, Any]:
    """Extract top-level and class-level constants/hyperparameters."""

    constants: dict[str, Any] = {}

    def visit_assignment(node: ast.AST, prefix: str = "") -> None:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            return
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        if value_node is None:
            return
        for target in targets:
            if isinstance(target, ast.Name):
                name = target.id
            elif isinstance(target, ast.Attribute):
                name = target.attr
            else:
                continue
            if name.isupper() or any(token in name.lower() for token in ("limit", "window", "threshold", "edge")):
                constants[f"{prefix}{name}"] = stable_literal(value_node)

    for node in tree.body if isinstance(tree, ast.Module) else []:
        visit_assignment(node)
        if isinstance(node, ast.ClassDef):
            for class_node in node.body:
                visit_assignment(class_node, prefix=f"{node.name}.")

    return dict(sorted(constants.items()))


def extract_functions(tree: ast.AST) -> dict[str, FunctionSummary]:
    """Extract function summaries from top-level functions and class methods."""

    functions: dict[str, FunctionSummary] = {}

    def add_function(node: ast.FunctionDef | ast.AsyncFunctionDef, prefix: str = "") -> None:
        name = f"{prefix}{node.name}"
        functions[name] = FunctionSummary(
            name=name,
            hash_value=function_hash(node),
            line=node.lineno,
            returns=sum(isinstance(child, ast.Return) for child in ast.walk(node)),
            order_calls=count_order_calls(node),
            state_refs=collect_name_refs(node, STATE_KEYWORDS),
            order_refs=collect_name_refs(node, ORDER_KEYWORDS),
        )

    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_function(node)
        elif isinstance(node, ast.ClassDef):
            for class_node in node.body:
                if isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add_function(class_node, prefix=f"{node.name}.")

    return dict(sorted(functions.items()))


def summarize(path: Path) -> AlgorithmSummary:
    """Parse a Python file and produce a logical summary."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    return AlgorithmSummary(
        path=path,
        constants=extract_constants(tree),
        functions=extract_functions(tree),
        imports=extract_imports(tree),
        product_literals=extract_product_literals(tree),
        risk_refs=collect_name_refs(tree, RISK_KEYWORDS),
        broad_except_count=count_broad_excepts(tree),
    )


def compare_constants(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Build human-readable constant changes."""

    rows = []
    old_keys = set(old)
    new_keys = set(new)
    for key in sorted(old_keys - new_keys):
        rows.append(f"- Removed hyperparameter `{key}` = {old[key]!r}")
    for key in sorted(new_keys - old_keys):
        rows.append(f"- Added hyperparameter `{key}` = {new[key]!r}")
    for key in sorted(old_keys & new_keys):
        if old[key] != new[key]:
            rows.append(f"- Changed `{key}`: {old[key]!r} -> {new[key]!r}")
    return rows


def compare_functions(old: dict[str, FunctionSummary], new: dict[str, FunctionSummary]) -> list[str]:
    """Build human-readable function changes."""

    rows = []
    old_keys = set(old)
    new_keys = set(new)
    for key in sorted(old_keys - new_keys):
        rows.append(f"- Removed function `{key}` at old line {old[key].line}")
    for key in sorted(new_keys - old_keys):
        summary = new[key]
        rows.append(
            f"- Added function `{key}` at new line {summary.line}; "
            f"Order calls: {summary.order_calls}; state refs: {format_refs(summary.state_refs)}"
        )
    for key in sorted(old_keys & new_keys):
        old_summary = old[key]
        new_summary = new[key]
        if old_summary.hash_value == new_summary.hash_value:
            continue
        rows.append(
            f"- Modified `{key}`; returns {old_summary.returns}->{new_summary.returns}, "
            f"Order calls {old_summary.order_calls}->{new_summary.order_calls}"
        )
    return rows


def compare_reference_focus(
    old: dict[str, FunctionSummary],
    new: dict[str, FunctionSummary],
    attr: str,
    label: str,
) -> list[str]:
    """Compare state or order reference counts for modified shared functions."""

    rows = []
    for key in sorted(set(old) & set(new)):
        old_refs = getattr(old[key], attr)
        new_refs = getattr(new[key], attr)
        if old_refs != new_refs:
            rows.append(f"- `{key}` {label}: {format_refs(old_refs)} -> {format_refs(new_refs)}")
    return rows


def compare_imports(old: list[str], new: list[str]) -> list[str]:
    """Compare import-level dependency shifts."""

    rows = []
    old_set = set(old)
    new_set = set(new)
    for item in sorted(old_set - new_set):
        rows.append(f"- Removed import `{item}`")
    for item in sorted(new_set - old_set):
        rows.append(f"- Added import `{item}`")
    return rows


def compare_risk_profile(old: AlgorithmSummary, new: AlgorithmSummary) -> list[str]:
    """Highlight review items that matter before Round 3 submission."""

    rows = []
    old_products = set(old.product_literals)
    new_products = set(new.product_literals)
    added_products = sorted(new_products - old_products)
    removed_products = sorted(old_products - new_products)
    if added_products:
        rows.append(f"- Added product/string universe literals: {', '.join(f'`{p}`' for p in added_products)}")
    if removed_products:
        rows.append(f"- Removed product/string universe literals: {', '.join(f'`{p}`' for p in removed_products)}")
    if old.broad_except_count != new.broad_except_count:
        rows.append(f"- Bare `except` handlers changed: {old.broad_except_count} -> {new.broad_except_count}")
    if old.risk_refs != new.risk_refs:
        rows.append(f"- Risk/math refs: {format_refs(old.risk_refs)} -> {format_refs(new.risk_refs)}")
    if new.broad_except_count:
        rows.append("- Review bare `except` blocks manually; they can hide broken product modules during live runs.")
    return rows


def format_refs(refs: dict[str, int]) -> str:
    """Format reference counts compactly."""

    if not refs:
        return "none"
    return ", ".join(f"{key}:{value}" for key, value in refs.items())


def render_markdown(old: AlgorithmSummary, new: AlgorithmSummary) -> str:
    """Render the comparison as Markdown."""

    sections = [
        ("Files", [f"- Old: `{old.path}`", f"- New: `{new.path}`"]),
        ("Hyperparameter Changes", compare_constants(old.constants, new.constants)),
        ("Logical Function Shifts", compare_functions(old.functions, new.functions)),
        (
            "State Management Shifts",
            compare_reference_focus(old.functions, new.functions, "state_refs", "state refs"),
        ),
        (
            "Order Logic Shifts",
            compare_reference_focus(old.functions, new.functions, "order_refs", "order refs"),
        ),
        ("Risk Review", compare_risk_profile(old, new)),
        ("Dependency Shifts", compare_imports(old.imports, new.imports)),
    ]

    lines: list[str] = []
    for title, rows in sections:
        lines.append(f"## {title}")
        if rows:
            lines.extend(rows)
        else:
            lines.append("- No material changes detected.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_json(old: AlgorithmSummary, new: AlgorithmSummary) -> str:
    """Render raw summary data as JSON."""

    def function_to_dict(summary: FunctionSummary) -> dict[str, Any]:
        return {
            "line": summary.line,
            "hash": summary.hash_value,
            "returns": summary.returns,
            "order_calls": summary.order_calls,
            "state_refs": summary.state_refs,
            "order_refs": summary.order_refs,
        }

    payload = {
        "old": {
            "path": str(old.path),
            "constants": old.constants,
            "functions": {key: function_to_dict(value) for key, value in old.functions.items()},
            "imports": old.imports,
            "product_literals": old.product_literals,
            "risk_refs": old.risk_refs,
            "broad_except_count": old.broad_except_count,
        },
        "new": {
            "path": str(new.path),
            "constants": new.constants,
            "functions": {key: function_to_dict(value) for key, value in new.functions.items()},
            "imports": new.imports,
            "product_literals": new.product_literals,
            "risk_refs": new.risk_refs,
            "broad_except_count": new.broad_except_count,
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description="Summarize logical shifts between two Prosperity algos.")
    parser.add_argument("old", type=Path, help="Old algorithm Python file.")
    parser.add_argument("new", type=Path, help="New algorithm Python file.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary JSON.")
    return parser


def main() -> None:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()

    old = summarize(args.old)
    new = summarize(args.new)
    if args.json:
        print(render_json(old, new))
    else:
        print(render_markdown(old, new))


if __name__ == "__main__":
    main()
