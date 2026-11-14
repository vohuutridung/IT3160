"""Human-readable tables for evaluation results."""

from __future__ import annotations

from .metrics import GROUPS

HEADLINE = ("nll", "top1", "top3", "top5")


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _with_ci(metrics: dict, group: str, key: str, digits: int) -> str:
    value = metrics.get(group, {}).get(key)
    ci = metrics.get("ci", {}).get(group, {}).get(key)
    text = _fmt(value, digits)
    if ci and ci.get("low") is not None:
        text += f" [{ci['low']:.{digits}f}, {ci['high']:.{digits}f}]"
    return text


def metrics_table(metrics: dict) -> str:
    """Markdown table: rows = overall + phases, columns = headline metrics."""
    lines = [
        "| Group | N | NLL | Top-1 % | Top-3 % | Top-5 % | Mean rank | ECE % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in GROUPS:
        row = metrics.get(group)
        if not row or not row.get("samples"):
            continue
        cells = [group, str(row["samples"])]
        cells += [_with_ci(metrics, group, key, 4 if key == "nll" else 2) for key in HEADLINE]
        cells += [_fmt(row.get("mean_rank")), _fmt(row.get("ece"))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def slices_table(slices: dict) -> str:
    lines = []
    for name, groups in slices.items():
        lines += [f"\n**{name}**\n",
                  "| Slice | N | NLL | Random NLL | Top-1 % | Random Top-1 % | Top-3 % | Top-5 % |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for label, row in groups.items():
            lines.append(
                f"| {label} | {row['samples']} | {_fmt(row.get('nll'), 4)} | {_fmt(row.get('random_nll'), 4)}"
                f" | {_fmt(row.get('top1'))} | {_fmt(row.get('random_top1'))}"
                f" | {_fmt(row.get('top3'))} | {_fmt(row.get('top5'))} |"
            )
    return "\n".join(lines)


def comparison_table(names: list[str], summaries: list[dict], group: str = "overall") -> str:
    lines = [
        f"| Model ({group}) | NLL | Top-1 % | Top-3 % | Top-5 % | Mean rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in zip(names, summaries):
        row = metrics.get(group, {})
        lines.append(
            f"| {name} | {_fmt(row.get('nll'), 4)} | {_fmt(row.get('top1'))} | {_fmt(row.get('top3'))}"
            f" | {_fmt(row.get('top5'))} | {_fmt(row.get('mean_rank'))} |"
        )
    return "\n".join(lines)


def paired_table(rows: list[dict]) -> str:
    lines = [
        "| Model vs reference | Group | ΔTop-1 (pp) [95% CI] | McNemar p | ΔNLL [95% CI] |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        d1, dn = row["delta_top1"], row["delta_nll"]
        ci1 = f" [{d1['low']:+.2f}, {d1['high']:+.2f}]" if d1["low"] is not None else ""
        cin = f" [{dn['low']:+.4f}, {dn['high']:+.4f}]" if dn["low"] is not None else ""
        lines.append(
            f"| {row['model']} vs {row['reference']} | {row['group']} | {d1['estimate']:+.2f}{ci1}"
            f" | {row['mcnemar']['p_value']:.3g} | {dn['estimate']:+.4f}{cin} |"
        )
    return "\n".join(lines)
