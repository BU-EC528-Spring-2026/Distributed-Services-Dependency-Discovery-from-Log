import os
import sys
import json
import asyncio
import argparse
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import graphviz
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

try:
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

from utils import extract_symptoms, link_symptoms_to_interactions, SYMPTOM_COLORS


# ---------------------------------------------------------------------------
# Interaction extraction
# ---------------------------------------------------------------------------

async def _run_agent(agent: AssistantAgent, task: str) -> str:
    reply = await agent.run(task=task)
    return reply.messages[-1].content if reply.messages else ""


def _repair_json_escapes(s: str) -> str:
    """Double any backslash that is not part of a valid JSON escape sequence."""
    _VALID = set('"\\\/bfnrt')
    result: list = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '\\' and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == 'u' and i + 5 < len(s):
                result.append(s[i:i + 6]); i += 6; continue
            if nxt in _VALID:
                result.append(s[i:i + 2]); i += 2; continue
            result.append('\\\\'); i += 1; continue
        result.append(ch); i += 1
    return ''.join(result)


def _parse_json_list(content: str) -> List[Dict[str, Any]]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if "```" in stripped:
            stripped = stripped.rsplit("```", 1)[0]
    stripped = stripped.strip()
    try:
        result = json.loads(stripped)
        return result if isinstance(result, list) else [result]
    except Exception:
        pass
    try:
        result = json.loads(_repair_json_escapes(stripped))
        return result if isinstance(result, list) else [result]
    except Exception as e:
        print(f"Error parsing LLM response: {e}\nRaw:\n{content}")
        return []


async def extract_interactions(agent: AssistantAgent, log_path: str) -> List[Dict[str, Any]]:
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    numbered = "".join(f"{i + 1}: {line}" for i, line in enumerate(lines))

    task = (
        "Analyze the following log file and identify ALL service interactions between distributed components "
        "(e.g. Spark, Hive, HDFS, YARN, etc.).\n\n"
        f"Log file:\n{numbered}\n\n"
        "For EACH interaction, return a JSON object:\n"
        "{\n"
        '  "service_a": {"value": "<name>", "regex": "<pattern>", "line": "<line_no>"},\n'
        '  "operation_a": {"value": "<op>", "regex": "<pattern>", "line": "<line_no>", "timestamp": "<ts>"},\n'
        '  "service_b": {"value": "<name>", "regex": "<pattern>", "line": "<line_no>"},\n'
        '  "operation_b": {"value": "<op>", "regex": "<pattern>", "line": "<line_no>", "timestamp": "<ts>"},\n'
        '  "resource": {"value": "<resource>", "regex": "<pattern>", "line": "<line_no>", "timestamp": "<ts>"}\n'
        "}\n\n"
        "- service_a is the caller, service_b is the callee.\n"
        "- regex fields must be valid Python regex patterns.\n"
        "- Return ONLY a valid JSON array, no extra text.\n"
    )
    raw = await _run_agent(agent, task)
    return _parse_json_list(raw)


# ---------------------------------------------------------------------------
# Timeline swimlane chart
# ---------------------------------------------------------------------------

def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Try multiple timestamp formats; return None on failure."""
    if not ts_str:
        return None
    for fmt in (
        "%y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%H:%M:%S",
    ):
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except ValueError:
            pass
    return None


def _short(name: str, max_chars: int = 10) -> str:
    return name if len(name) <= max_chars else name[:max_chars] + "..."


def build_timeline_chart(
    interactions: List[Dict[str, Any]],
    linked_symptoms: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Generate a horizontal swimlane timeline showing service operations and interactions."""

    # ── collect services (insertion order) ──────────────────────────────────
    # Use "value regex" as the unique key so sub-components (e.g. YARN ResourceManager
    # vs YARN RMAppManager) get their own rows even when value is the same.
    def _svc_key(entry: Dict[str, Any]) -> str:
        val   = (entry.get("value") or "").strip()
        regex = (entry.get("regex") or "").strip()
        # If regex is a URL-like pattern or longer than ~24 chars, just use value
        if not regex or regex == val or len(regex) > 24 or "/" in regex or "\\" in regex:
            return val
        return f"{val} {regex}"

    services: List[str] = []
    seen_svcs: set = set()
    for ix in interactions:
        for key in ("service_a", "service_b"):
            entry = ix.get(key) or {}
            svc = _svc_key(entry)
            if svc and svc not in seen_svcs:
                services.append(svc)
                seen_svcs.add(svc)

    if not services:
        print("No services found; skipping timeline chart.")
        return

    # ── collect operations and arrow links ───────────────────────────────────
    # ops_by_svc: svc → [(datetime, op_name)]
    ops_by_svc: Dict[str, List[Tuple[datetime, str]]] = {s: [] for s in services}
    raw_arrows: List[Tuple[str, datetime, str, datetime]] = []  # (svc_a, ts_a, svc_b, ts_b)

    for ix in interactions:
        svc_a = _svc_key(ix.get("service_a") or {})
        svc_b = _svc_key(ix.get("service_b") or {})
        op_a  = ((ix.get("operation_a") or {}).get("value") or "").strip()
        op_b  = ((ix.get("operation_b") or {}).get("value") or "").strip()
        ts_a  = _parse_timestamp(((ix.get("operation_a") or {}).get("timestamp") or ""))
        ts_b  = _parse_timestamp(((ix.get("operation_b") or {}).get("timestamp") or ""))

        if svc_a in ops_by_svc and ts_a:
            ops_by_svc[svc_a].append((ts_a, op_a))
        if svc_b in ops_by_svc and ts_b:
            ops_by_svc[svc_b].append((ts_b, op_b))
        if svc_a and svc_b and ts_a and ts_b:
            raw_arrows.append((svc_a, ts_a, svc_b, ts_b))

    # ── de-duplicate ops (same svc+ts) ──────────────────────────────────────
    for svc in services:
        unique: Dict[datetime, str] = {}
        for ts, op in ops_by_svc[svc]:
            if ts not in unique:
                unique[ts] = op
        ops_by_svc[svc] = list(unique.items())

    all_ts = [ts for ops in ops_by_svc.values() for (ts, _) in ops]
    if not all_ts:
        print("No timestamps found; skipping timeline chart.")
        return

    min_ts, max_ts = min(all_ts), max(all_ts)
    total_secs = max((max_ts - min_ts).total_seconds(), 1.0)

    def to_x(ts: datetime) -> float:
        return (ts - min_ts).total_seconds()

    # ── layout constants ─────────────────────────────────────────────────────
    ROW_H   = 1.0
    ROW_GAP = 0.65
    STEP    = ROW_H + ROW_GAP
    # Use a minimum scale so charts with very short time ranges stay readable.
    # All size-dependent constants derive from `scale`, not raw `total_secs`.
    scale     = max(total_secs, 12.0)
    OP_W      = scale * 0.12          # op block width in data units
    OP_H      = ROW_H * 0.48
    MARGIN    = scale * 0.10
    LABEL_PAD = scale * 0.42          # left space for multi-line service labels

    n = len(services)
    svc_y: Dict[str, float] = {svc: (n - 1 - i) * STEP for i, svc in enumerate(services)}

    x_lo = -MARGIN

    # ── pre-compute consequence positions so we know x_hi ────────────────────
    STYPE_COLORS = {
        "error_exception":         ("#c0392b", "#e74c3c"),
        "abnormal_execution_time":  ("#a05000", "#e8882a"),
        "excessive_resource_usage": ("#6c3483", "#9b59b6"),
    }

    def _resolve_svc(ls: Dict[str, Any]) -> str:
        """Find the service row that best matches this symptom entry.
        Supports both exact keys and substring matches (e.g. 'Executor' matches
        'Spark Executor') so that regex-based combined keys are handled."""
        candidates: List[str] = []
        for key in ("symptom_service", "linked_service_b", "linked_service_a"):
            v = (ls.get(key) or "").strip()
            if not v:
                continue
            if v in svc_y:                        # exact match
                return v
            for svc_key in svc_y:                 # substring match
                if v in svc_key or svc_key in v:
                    candidates.append(svc_key)
        return candidates[0] if candidates else ""

    seen_csq: set = set()
    csq_list: List[Tuple[str, str, str]] = []
    for ls in linked_symptoms:
        svc   = _resolve_svc(ls)
        stype = (ls.get("symptom_type") or "unknown").strip()
        event = (ls.get("symptom_event") or "").strip()
        if not svc or not event:
            continue
        key = (svc, event)
        if key in seen_csq:
            continue
        seen_csq.add(key)
        csq_list.append((svc, stype, event))

    c_w   = OP_W * 1.5
    c_gap = c_w * 0.25
    # simulate placement to find rightmost x
    csq_next_x: Dict[str, float] = {}
    csq_max_x = total_secs
    for svc, _, _ in csq_list:
        if svc not in csq_next_x:
            x_ops = [to_x(ts) for (ts, _) in ops_by_svc[svc]]
            csq_next_x[svc] = (max(x_ops) + OP_W + c_gap) if x_ops else scale * 0.6
        x_right = csq_next_x[svc] + c_w
        csq_max_x = max(csq_max_x, x_right)
        csq_next_x[svc] += c_w + c_gap
    csq_next_x.clear()   # reset for actual drawing pass

    x_hi = max(csq_max_x, scale) + MARGIN

    fig_w = min(max(16, (x_hi - x_lo + LABEL_PAD) / 4.5), 36)
    fig_h = n * STEP + 2.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("white")

    # ── swimlane backgrounds ─────────────────────────────────────────────────
    for svc in services:
        y = svc_y[svc]
        lane_y  = y + (ROW_H - ROW_H * 0.82) / 2
        lane_h  = ROW_H * 0.82
        bg = mpatches.FancyBboxPatch(
            (x_lo, lane_y), x_hi - x_lo, lane_h,
            boxstyle="square,pad=0",
            linewidth=0, facecolor="#cce7f5", alpha=0.55, zorder=1,
        )
        ax.add_patch(bg)
        # Wrap the service name at word boundaries, ~12 chars per line
        words = svc.split()
        lines: List[str] = []
        current = ""
        for w in words:
            candidate = (current + " " + w).strip()
            if len(candidate) <= 12 or not current:
                current = candidate
            else:
                lines.append(current)
                current = w
        if current:
            lines.append(current)
        label = "\n".join(lines)
        ax.text(
            x_lo - 0.3, y + ROW_H / 2, label,
            ha="right", va="center",
            fontsize=8.5, fontweight="bold", color="#1a3a5c",
            linespacing=1.25,
        )

    # ── draw operation blocks ────────────────────────────────────────────────
    # op_center[(svc, ts)] = (x, y)  — used for arrow anchoring
    op_center: Dict[Tuple[str, datetime], Tuple[float, float]] = {}
    # last_op_right[svc] = (x_right_edge, y_mid) of rightmost op on that row
    last_op_right: Dict[str, Tuple[float, float]] = {}

    for svc, ops in ops_by_svc.items():
        y_mid = svc_y[svc] + ROW_H / 2
        for ts, op_name in ops:
            x = to_x(ts)
            rect = mpatches.FancyBboxPatch(
                (x - OP_W / 2, y_mid - OP_H / 2), OP_W, OP_H,
                boxstyle="round,pad=0.06",
                linewidth=0.8, edgecolor="#1b6b1b", facecolor="#5cb85c", zorder=3,
            )
            ax.add_patch(rect)
            ax.text(x, y_mid, _short(op_name),
                    ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold", zorder=4)
            op_center[(svc, ts)] = (x, y_mid)
            # track rightmost op per service
            if svc not in last_op_right or x > last_op_right[svc][0]:
                last_op_right[svc] = (x + OP_W / 2, y_mid)

    # ── draw consequence blocks + dashed arrows from last op ─────────────────
    csq_tick_xs: List[float] = []          # x-centers of all consequence blocks

    for svc, stype, event in csq_list:
        y_mid = svc_y[svc] + ROW_H / 2
        if svc not in csq_next_x:
            x_ops = [to_x(ts) for (ts, _) in ops_by_svc[svc]]
            csq_next_x[svc] = (max(x_ops) + OP_W + c_gap) if x_ops else scale * 0.6

        x_c = csq_next_x[svc] + c_w / 2
        csq_next_x[svc] += c_w + c_gap
        csq_tick_xs.append(x_c)

        edge_c, face_c = STYPE_COLORS.get(stype, ("#555555", "#888888"))
        rect = mpatches.FancyBboxPatch(
            (x_c - c_w / 2, y_mid - OP_H / 2), c_w, OP_H,
            boxstyle="round,pad=0.06",
            linewidth=0.8, edgecolor=edge_c, facecolor=face_c, zorder=3,
        )
        ax.add_patch(rect)
        # two-line label: type on top, event name below
        type_short  = _short(stype.replace("_", " "), 10)
        event_short = _short(event, 10)
        ax.text(x_c, y_mid + OP_H * 0.15, type_short,
                ha="center", va="center",
                fontsize=5.5, color="white", fontweight="bold", zorder=4)
        ax.text(x_c, y_mid - OP_H * 0.22, event_short,
                ha="center", va="center",
                fontsize=5, color="white", zorder=4)

        # dashed arrow: last op right-edge → consequence left-edge
        if svc in last_op_right:
            x_src, y_src = last_op_right[svc]
            x_dst = x_c - c_w / 2
            csq_arrow = FancyArrowPatch(
                (x_src, y_src), (x_dst, y_mid),
                arrowstyle="-|>",
                color="#888888",
                linewidth=1.0,
                linestyle="dashed",
                connectionstyle="arc3,rad=0.0",
                mutation_scale=10,
                zorder=2,
            )
            ax.add_patch(csq_arrow)

    # ── draw interaction arrows ──────────────────────────────────────────────
    seen_arrow_keys: set = set()
    for svc_a, ts_a, svc_b, ts_b in raw_arrows:
        key = (svc_a, ts_a, svc_b, ts_b)
        if key in seen_arrow_keys:
            continue
        seen_arrow_keys.add(key)

        xa, ya = op_center.get((svc_a, ts_a), (to_x(ts_a), svc_y.get(svc_a, 0) + ROW_H / 2))
        xb, yb = op_center.get((svc_b, ts_b), (to_x(ts_b), svc_y.get(svc_b, 0) + ROW_H / 2))

        # curve direction alternates to reduce overlap
        rad = 0.28 if (ya > yb) else -0.28
        arrow = FancyArrowPatch(
            (xa, ya), (xb, yb),
            arrowstyle="<->",
            color="#1a4a8a",
            linewidth=1.4,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=13,
            zorder=5,
        )
        ax.add_patch(arrow)

    # ── axis / decoration ────────────────────────────────────────────────────
    ax.set_xlim(x_lo - LABEL_PAD - 0.3, x_hi + 0.5)
    ax.set_ylim(-ROW_GAP, n * STEP + ROW_GAP)

    # Real timestamps only cover the operation range [0, total_secs]
    # Operation-range ticks (real timestamps)
    n_op_ticks = min(6, max(2, int(total_secs / 5) + 1))
    op_tick_secs = [i * total_secs / (n_op_ticks - 1) for i in range(n_op_ticks)]

    # Consequence ticks — one per unique x-center, de-duplicated by rounding
    seen_csq_x: set = set()
    csq_tick_secs_dedup: List[float] = []
    for x in csq_tick_xs:
        key = round(x, 3)
        if key not in seen_csq_x:
            seen_csq_x.add(key)
            csq_tick_secs_dedup.append(x)

    all_tick_secs = sorted(set(
        [round(t, 6) for t in op_tick_secs] +
        [round(t, 6) for t in csq_tick_secs_dedup]
    ))
    all_tick_labels = [
        (min_ts + timedelta(seconds=s)).strftime("%H:%M:%S") for s in all_tick_secs
    ]
    ax.set_xticks(all_tick_secs)
    ax.set_xticklabels(all_tick_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks([])

    ax.set_xlabel("Time", fontsize=9, labelpad=6)
    ax.set_title("Service Interaction Timeline", fontsize=11, fontweight="bold", pad=12)

    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)

    # legend
    legend_handles = [
        mpatches.Patch(facecolor="#5cb85c", edgecolor="#1b6b1b", label="operation"),
        mpatches.Patch(facecolor="#e74c3c", edgecolor="#c0392b", label="error / exception"),
        mpatches.Patch(facecolor="#e8882a", edgecolor="#a05000", label="abnormal exec time"),
        mpatches.Patch(facecolor="#9b59b6", edgecolor="#6c3483", label="excessive resource"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7.5, framealpha=0.8)

    # "Process" label on the far left
    ax.text(x_lo - LABEL_PAD * 0.95, n * STEP / 2, "Process",
            ha="center", va="center", fontsize=9, color="#1a3a5c",
            rotation=90)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Timeline chart saved to: {output_path}")


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

def build_graph(
    interactions: List[Dict[str, Any]],
    linked_symptoms: List[Dict[str, Any]],
) -> graphviz.Digraph:
    dot = graphviz.Digraph(comment="Service Dependency Graph with Symptoms")
    dot.attr(rankdir="LR")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="lightblue")

    # --- service nodes & interaction edges ---
    seen_services: set = set()
    for ix in interactions:
        svc_a = ((ix.get("service_a") or {}).get("value") or "").strip()
        svc_b = ((ix.get("service_b") or {}).get("value") or "").strip()
        op_a  = (ix.get("operation_a") or {}).get("value") or ""
        op_b  = (ix.get("operation_b") or {}).get("value") or ""
        res   = (ix.get("resource")    or {}).get("value") or ""
        ts_a  = (ix.get("operation_a") or {}).get("timestamp") or ""
        ts_b  = (ix.get("operation_b") or {}).get("timestamp") or ""
        op_a, op_b, res, ts_a, ts_b = (
            op_a.strip(), op_b.strip(), res.strip(), ts_a.strip(), ts_b.strip()
        )

        for svc in (svc_a, svc_b):
            if svc and svc not in seen_services:
                dot.node(svc, svc)
                seen_services.add(svc)

        if svc_a and svc_b:
            parts = []
            if op_a:  parts.append(f"op_a: {op_a}")
            if ts_a:  parts.append(f"ts_a: {ts_a}")
            if op_b:  parts.append(f"op_b: {op_b}")
            if ts_b:  parts.append(f"ts_b: {ts_b}")
            if res:   parts.append(f"res: {res}")
            dot.edge(svc_a, svc_b, label="\n".join(parts))

    # --- symptom nodes & consequence edges ---
    seen_symptom_ids: set = set()
    for i, ls in enumerate(linked_symptoms):
        stype   = ls.get("symptom_type", "unknown")
        event   = ls.get("symptom_event", "")
        svc     = ls.get("symptom_service", "")
        svc_a   = ls.get("linked_service_a", "")
        svc_b   = ls.get("linked_service_b", "")
        reason  = ls.get("reason", "")

        # unique node id to allow duplicate events
        node_id = f"symptom_{i}"
        color   = SYMPTOM_COLORS.get(stype, "lightyellow")

        # short label: type + event (truncate long events)
        short_event = event if len(event) <= 40 else event[:37] + "..."
        node_label  = f"[{stype}]\n{short_event}"

        dot.node(node_id, node_label,
                 shape="ellipse", style="filled", fillcolor=color)

        # draw dashed "consequence" edge from the owning service
        # prefer symptom_service if it's already a node, else fall back to linked_service_a
        source = svc if svc in seen_services else (svc_a if svc_a in seen_services else svc_b)
        if source:
            dot.edge(source, node_id,
                     label=f"consequence\n({reason[:50]}...)" if len(reason) > 50 else f"consequence\n({reason})",
                     style="dashed", color="gray40")

        seen_symptom_ids.add(node_id)

    return dot


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

async def run_workflow(log_path: str) -> None:
    print(f"=== Service Dependency & Symptom Analysis ===")
    print(f"Analyzing: {log_path}")

    openai_model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = AssistantAgent(
        name="log_analyst",
        model_client=openai_model_client,
        system_message=(
            "You are a distributed systems log analysis expert for services like Spark, Hive, HDFS, YARN. "
            "You extract service interactions, detect symptom events, and reason about causal relationships. "
            "Always return strictly valid JSON as instructed."
        ),
    )

    try:
        log_name  = os.path.splitext(os.path.basename(log_path))[0]
        output_dir = os.path.normpath(
            os.path.join(os.path.dirname(log_path), "..", "output", log_name)
        )
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Extract interactions
        print("\n=== Step 1: Extracting service interactions ===")
        interactions = await extract_interactions(agent, log_path)
        print(f"Found {len(interactions)} interaction(s).")
        with open(os.path.join(output_dir, "interactions.json"), "w", encoding="utf-8") as f:
            json.dump(interactions, f, indent=2, ensure_ascii=False)

        # Step 2: Extract symptoms
        print("\n=== Step 2: Extracting symptoms ===")
        symptoms = await extract_symptoms(agent, log_path)
        print(f"Found {len(symptoms)} symptom(s).")
        with open(os.path.join(output_dir, "symptoms.json"), "w", encoding="utf-8") as f:
            json.dump(symptoms, f, indent=2, ensure_ascii=False)

        # Step 3: Link symptoms to interactions
        print("\n=== Step 3: Linking symptoms to interactions ===")
        linked_symptoms = await link_symptoms_to_interactions(agent, interactions, symptoms)
        print(f"Linked {len(linked_symptoms)} symptom(s) to interactions.")
        with open(os.path.join(output_dir, "linked_symptoms.json"), "w", encoding="utf-8") as f:
            json.dump(linked_symptoms, f, indent=2, ensure_ascii=False)

        # Step 4: Build and save enhanced graph
        print("\n=== Step 4: Building dependency graph with symptoms ===")
        dot = build_graph(interactions, linked_symptoms)

        graph_out = os.path.join(output_dir, "dependency_graph")
        dot.render(graph_out, format="png", cleanup=True)
        print(f"Graph (PNG) saved to: {graph_out}.png")

        with open(graph_out + ".dot", "w", encoding="utf-8") as f:
            f.write(dot.source)
        print(f"DOT source saved to: {graph_out}.dot")

        # Step 5: Build timeline swimlane chart
        print("\n=== Step 5: Building service interaction timeline chart ===")
        timeline_out = os.path.join(output_dir, "timeline_chart.png")
        build_timeline_chart(interactions, linked_symptoms, timeline_out)

    finally:
        try:
            usage = openai_model_client.total_usage()
            print(
                f"\nToken Usage - Prompt: {usage.prompt_tokens}, "
                f"Completion: {usage.completion_tokens}, "
                f"Total: {usage.prompt_tokens + usage.completion_tokens}"
            )
        except Exception:
            pass
        await openai_model_client.close()
        print("OpenAI model client closed.")


def main():
    parser = argparse.ArgumentParser(
        description="Service Dependency & Symptom Analysis from Logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", metavar="LOG_PATH", required=True, dest="log_path",
                        help="Path to the log file to analyze")
    args = parser.parse_args()

    if not os.path.isfile(args.log_path):
        print(f"Error: Log file not found: {args.log_path}")
        sys.exit(1)

    asyncio.run(run_workflow(args.log_path))


if __name__ == "__main__":
    main()
