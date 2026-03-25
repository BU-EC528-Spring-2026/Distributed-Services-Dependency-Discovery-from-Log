import os
import sys
import json
import asyncio
import argparse
from typing import Dict, Any, List

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

try:
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import graphviz


async def _run_agent(agent: AssistantAgent, task: str) -> str:
    reply = await agent.run(task=task)
    if not reply.messages:
        return "[No response]"
    return reply.messages[-1].content


def _parse_json_from_response(content: str) -> Any:
    """Strip markdown fences and parse JSON."""
    stripped = content.strip()
    if stripped.startswith("```"):
        # Remove opening fence (```json or ```)
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        # Remove closing fence
        if "```" in stripped:
            stripped = stripped.rsplit("```", 1)[0]
    return json.loads(stripped.strip())


async def extract_interactions(agent: AssistantAgent, log_path: str) -> List[Dict[str, Any]]:
    """Load log file and ask the LLM to extract all service interaction pairs."""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    numbered = "".join(f"{i + 1}: {line}" for i, line in enumerate(lines))

    task = (
        "Analyze the following log file and identify ALL service interactions between distributed components "
        "(e.g. Spark, Hive, HDFS, YARN, etc.).\n\n"
        f"Log file:\n{numbered}\n\n"
        "For EACH interaction, return a JSON object with this exact structure:\n"
        "{\n"
        '  "service_a": {"value": "<name>", "regex": "<pattern>", "line": "<line_no>"},\n'
        '  "operation_a": {"value": "<op>", "regex": "<pattern>", "line": "<line_no>", "timestamp": "<ts>"},\n'
        '  "service_b": {"value": "<name>", "regex": "<pattern>", "line": "<line_no>"},\n'
        '  "operation_b": {"value": "<op>", "regex": "<pattern>", "line": "<line_no>", "timestamp": "<ts>"},\n'
        '  "resource": {"value": "<resource>", "regex": "<pattern>", "line": "<line_no>", "timestamp": "<ts>"}\n'
        "}\n\n"
        "Rules:\n"
        "- service_a is the caller/requester, service_b is the callee/provider.\n"
        "- operation_a is what service_a does with the resource; operation_b is what service_b does.\n"
        "- resource is the shared object/data/endpoint mediating the interaction.\n"
        "- regex fields must be valid Python regex patterns matching the relevant log line.\n"
        "- line fields must be the exact line numbers (integers as strings) from the numbered log above.\n"
        "- Return ONLY a valid JSON array of such objects, with no additional text.\n"
    )

    raw = await _run_agent(agent, task)

    try:
        interactions = _parse_json_from_response(raw)
        if not isinstance(interactions, list):
            print(f"Warning: LLM returned non-list JSON, wrapping: {type(interactions)}")
            interactions = [interactions]
        return interactions
    except Exception as e:
        print(f"Error parsing LLM response as JSON: {e}")
        print(f"Raw response:\n{raw}")
        return []


def build_dependency_graph(interactions: List[Dict[str, Any]]) -> graphviz.Digraph:
    """Construct a directed dependency graph from extracted interactions."""
    dot = graphviz.Digraph(comment="Service Dependency Graph")
    dot.attr(rankdir="LR")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="lightblue")

    seen_nodes: set = set()

    for interaction in interactions:
        svc_a = interaction.get("service_a", {}).get("value", "").strip()
        svc_b = interaction.get("service_b", {}).get("value", "").strip()
        op_a = interaction.get("operation_a", {}).get("value", "").strip()
        op_b = interaction.get("operation_b", {}).get("value", "").strip()
        resource = interaction.get("resource", {}).get("value", "").strip()
        ts_a = interaction.get("operation_a", {}).get("timestamp", "").strip()
        ts_b = interaction.get("operation_b", {}).get("timestamp", "").strip()

        for svc in (svc_a, svc_b):
            if svc and svc not in seen_nodes:
                dot.node(svc, svc)
                seen_nodes.add(svc)

        if svc_a and svc_b:
            label_parts = []
            if op_a:
                label_parts.append(f"op_a: {op_a}")
            if ts_a:
                label_parts.append(f"ts_a: {ts_a}")
            if op_b:
                label_parts.append(f"op_b: {op_b}")
            if ts_b:
                label_parts.append(f"ts_b: {ts_b}")
            if resource:
                label_parts.append(f"res: {resource}")
            dot.edge(svc_a, svc_b, label="\n".join(label_parts))

    return dot


async def run_workflow(log_path: str) -> None:
    print(f"=== Service Dependency Discovery from Log ===")
    print(f"Analyzing: {log_path}")

    openai_model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = AssistantAgent(
        name="service_dependency_extractor",
        model_client=openai_model_client,
        system_message=(
            "You are a distributed systems log analysis expert. "
            "Your task is to analyze log files from distributed services (e.g. Spark, Hive, HDFS, YARN) "
            "and identify all inter-service interactions. "
            "For every interaction, extract the two services involved, their respective operations on a shared resource, "
            "and supporting evidence (regex patterns, line numbers, timestamps) from the log. "
            "Always return strictly valid JSON as instructed."
        ),
    )

    try:
        # Step 1: Extract interactions
        print("\n=== Step 1: Extracting service interactions via LLM ===")
        interactions = await extract_interactions(agent, log_path)

        if not interactions:
            print("No interactions found. Exiting.")
            return

        print(f"Found {len(interactions)} interaction(s).")
        print(json.dumps(interactions, indent=2, ensure_ascii=False))

        # Step 2: Build dependency graph
        print("\n=== Step 2: Building dependency graph ===")
        dot = build_dependency_graph(interactions)

        # Step 3: Save outputs
        log_name = os.path.splitext(os.path.basename(log_path))[0]
        output_dir = os.path.normpath(
            os.path.join(os.path.dirname(log_path), "..", "output")
        )
        os.makedirs(output_dir, exist_ok=True)

        json_out = os.path.join(output_dir, f"{log_name}_interactions.json")
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(interactions, f, indent=2, ensure_ascii=False)
        print(f"Interactions saved to: {json_out}")

        graph_out = os.path.join(output_dir, f"{log_name}_dependency_graph")
        dot.render(graph_out, format="png", cleanup=True)
        print(f"Dependency graph (PNG) saved to: {graph_out}.png")

        dot_out = graph_out + ".dot"
        with open(dot_out, "w", encoding="utf-8") as f:
            f.write(dot.source)
        print(f"DOT source saved to: {dot_out}")

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
        description="Distributed Services Dependency Discovery from Log",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i",
        metavar="LOG_PATH",
        required=True,
        dest="log_path",
        help="Path to the log file to analyze",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.log_path):
        print(f"Error: Log file not found: {args.log_path}")
        sys.exit(1)

    asyncio.run(run_workflow(args.log_path))


if __name__ == "__main__":
    main()
