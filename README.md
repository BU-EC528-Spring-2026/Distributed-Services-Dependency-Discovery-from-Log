# Distributed-Services-Dependency-Discovery-from-Log

This project explores using LLM-based agents combined with temporal analysis to automatically discover, visualize, and reason about service dependencies from log data. Once the dependency graph is reconstructed, the system can further support incident analysis by tracing anomaly propagation patterns—identifying how failures cascade through the service topology and revealing the likely propagation paths from root cause to downstream impact.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your OpenAI API key in `.env`:

```
OPENAI_API_KEY=sk-...
```

## Run

```bash
bash run.sh
# or manually:
python src/main.py -i log/spark-21714.log
```

Output files are written to `output/`:
- `*_interactions.json` — extracted service interactions
- `*_dependency_graph.png` — visualized dependency graph
- `*_dependency_graph.dot` — Graphviz DOT source
