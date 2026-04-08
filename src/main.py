import os
import sys
import json
import asyncio
import argparse
from typing import Any, Dict, List

import graphviz
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

try:
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from utils import extract_symptoms, link_symptoms_to_interactions, SYMPTOM_COLORS


# ---------------------------------------------------------------------------
# Interaction extraction
# ---------------------------------------------------------------------------

async def _run_agent(agent: AssistantAgent, task: str) -> str:
    reply = await agent.run(task=task)
    return reply.messages[-1].content if reply.messages else ""


def _parse_json_list(content: str) -> List[Dict[str, Any]]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if "```" in stripped:
            stripped = stripped.rsplit("```", 1)[0]
    try:
        result = json.loads(stripped.strip())
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
