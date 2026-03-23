# Quantum Agent Trace

**Quantum Agent Trace** is a small project for experimenting with AI agents and task automation. It helps organize agents, tasks, and workflows in a simple way, making it easy to test ideas and build custom AI-powered agents for research or personal projects.

This project currently includes a text summarization agent and a trace reporter agent with Opic/Comet integration to monitor execution and usage metrics. Future updates will expand to real-time monitoring, reporting dashboards, and more advanced AI agent workflows.

#AI #AIAgents #TaskAutomation #AgentTracing #Optic #Comet #Groq #LiteLLM

## What is new: Opik Learning Lab

This project now includes a dedicated learning flow for Opik tracing.

The learning lab runs the same input through three different summarization scenarios:

- tweet-style
- executive-brief
- learning-notes

Each scenario is executed as a traced CrewAI run (with reviewer validation), so you can compare behavior and metrics in Opik.

## Why this is useful

- Learn how tracing captures multi-agent workflows, not just one prompt.
- Compare latency, token usage, and cost across prompt styles.
- Generate local artifacts for side-by-side analysis.

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Configure environment variables in `.env` (or shell):

```bash
GROQ_API_KEY=your_groq_key
OPIK_API_KEY=your_opik_key
OPIK_WORKSPACE=your_workspace_name
```

3. Put sample input files in `tests/`.

## Run

Run baseline demo:

```bash
uv run run_crew
```

Run Opik learning lab:

```bash
uv run run_with_trigger
```

## Outputs

After the learning lab run:

- `Report/opik_learning_lab_report.md` contains scenario comparisons and outputs.
- `Report/opik_learning_lab_results.json` contains structured metrics/output data.
- Opik UI contains full traces for each scenario run.
