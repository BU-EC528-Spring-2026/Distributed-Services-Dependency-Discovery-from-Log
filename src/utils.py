import json
from pathlib import Path
from typing import Any, Dict, List

from autogen_agentchat.agents import AssistantAgent

TEMPLATE_DIR = Path(__file__).parent / "template"

SYMPTOM_TYPES = ["error_exception", "abnormal_execution_time", "excessive_resource_usage"]

SYMPTOM_COLORS = {
    "error_exception": "lightcoral",
    "abnormal_execution_time": "lightyellow",
    "excessive_resource_usage": "orange",
}


def load_symptom_templates() -> Dict[str, Dict]:
    """Load all three symptom templates from the template directory."""
    templates = {}
    for name in SYMPTOM_TYPES:
        path = TEMPLATE_DIR / f"{name}.json"
        with open(path, "r", encoding="utf-8") as f:
            templates[name] = json.load(f)
    return templates


async def extract_symptoms(agent: AssistantAgent, log_path: str) -> List[Dict[str, Any]]:
    """Ask the LLM to detect symptom events in the log and fill in the templates."""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    numbered = "".join(f"{i + 1}: {line}" for i, line in enumerate(lines))

    templates = load_symptom_templates()
    templates_str = json.dumps(templates, indent=2, ensure_ascii=False)

    task = (
        "Analyze the following log file and identify ALL symptom events that match the three symptom types below.\n\n"
        f"=== Symptom Templates ===\n{templates_str}\n\n"
        f"=== Log File ===\n{numbered}\n\n"
        "For EACH symptom event found, return a filled JSON object based on its matching template. "
        "Fill in every UNKNOWN field with the actual value from the log. Keep symptom_type fixed.\n\n"
        "Rules:\n"
        "- error_exception: any ERROR log level, exception stack trace, or explicit failure message.\n"
        "- abnormal_execution_time: timeout, slow operation, long GC pause, task taking unexpectedly long.\n"
        "- excessive_resource_usage: memory/CPU/disk/thread limit reached or abnormally high usage reported.\n"
        "- regex must be a valid Python regex that uniquely matches the relevant log line.\n"
        "- line must be the exact line number from the numbered log above.\n"
        "- If no events of a type exist, omit that type entirely.\n"
        "- Return ONLY a valid JSON array of filled symptom objects, no extra text.\n"
    )

    raw = await agent.run(task=task)
    content = raw.messages[-1].content if raw.messages else ""
    return _parse_json_list(content)


async def link_symptoms_to_interactions(
    agent: AssistantAgent,
    interactions: List[Dict[str, Any]],
    symptoms: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Ask the LLM to associate each symptom with the most relevant interaction as its consequence."""
    interactions_str = json.dumps(interactions, indent=2, ensure_ascii=False)
    symptoms_str = json.dumps(symptoms, indent=2, ensure_ascii=False)

    task = (
        "Given the following service interactions and symptom events from the same log, "
        "determine which interaction each symptom is a consequence of.\n\n"
        f"=== Interactions ===\n{interactions_str}\n\n"
        f"=== Symptoms ===\n{symptoms_str}\n\n"
        "For EACH symptom, return a JSON object with:\n"
        "- symptom_type: the symptom_type field from the symptom\n"
        "- symptom_event: the event field from the symptom\n"
        "- symptom_service: the service field from the symptom\n"
        "- linked_service_a: service_a.value of the related interaction\n"
        "- linked_service_b: service_b.value of the related interaction\n"
        "- reason: one sentence explaining why this symptom is a consequence of that interaction\n\n"
        "Rules:\n"
        "- Every symptom must be linked to exactly one interaction (the most causally relevant one).\n"
        "- If a symptom is not directly caused by any interaction, link it to the interaction whose "
        "  services are most involved in the symptom.\n"
        "- Return ONLY a valid JSON array of such objects, no extra text.\n"
    )

    raw = await agent.run(task=task)
    content = raw.messages[-1].content if raw.messages else ""
    return _parse_json_list(content)


def _parse_json_list(content: str) -> List[Dict[str, Any]]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if "```" in stripped:
            stripped = stripped.rsplit("```", 1)[0]
    try:
        result = json.loads(stripped.strip())
        if not isinstance(result, list):
            result = [result]
        return result
    except Exception as e:
        print(f"Error parsing LLM response as JSON: {e}")
        print(f"Raw response:\n{content}")
        return []
