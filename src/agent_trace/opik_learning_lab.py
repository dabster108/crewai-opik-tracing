import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from crewai import Agent, Crew, Task
from opik.integrations.crewai import track_crewai

logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

# Enables automatic CrewAI run tracing in Opik.
track_crewai()


@dataclass
class ScenarioResult:
    source_file: str
    scenario: str
    latency_seconds: float
    total_tokens: int | None
    cost: float | None
    model: str
    output: str


SCENARIOS = [
    {
        "name": "tweet-style",
        "instruction": "Write a tight summary in at most 280 characters.",
    },
    {
        "name": "executive-brief",
        "instruction": "Write a professional summary for leadership in 3 bullet points.",
    },
    {
        "name": "learning-notes",
        "instruction": "Write a beginner-friendly summary with 3 key takeaways.",
    },
]


def _extract_usage(crew: Crew) -> tuple[int | None, float | None]:
    usage = getattr(crew, "usage_metrics", None)
    if not usage:
        return None, None

    tokens = getattr(usage, "total_tokens", None)
    cost = getattr(usage, "cost", None)
    return tokens, cost


def _run_scenario(source_file: str, input_text: str, scenario_name: str, instruction: str) -> ScenarioResult:
    start = time.time()

    summarizer = Agent(
        role="Scenario Summarizer",
        goal="Create summaries in requested style while preserving facts.",
        backstory="Specialist in adapting one source into different summary styles.",
        llm="groq/llama-3.3-70b-versatile",
        verbose=True,
    )

    reviewer = Agent(
        role="Quality Reviewer",
        goal="Review and improve the summary for clarity and accuracy.",
        backstory="Expert editor who fixes unclear or missing points.",
        llm="groq/llama-3.3-70b-versatile",
        verbose=True,
    )

    summarize_task = Task(
        description=(
            "You are running scenario: "
            f"{scenario_name}.\n"
            f"Instruction: {instruction}\n\n"
            "Text to summarize:\n"
            f"{input_text}"
        ),
        expected_output="A summary that follows the scenario instruction exactly.",
        agent=summarizer,
    )

    review_task = Task(
        description=(
            "Review the produced summary. Improve clarity and factual correctness. "
            "Keep the same style and constraints from the scenario instruction."
        ),
        expected_output="Final polished summary.",
        agent=reviewer,
        context=[summarize_task],
    )

    crew = Crew(
        agents=[summarizer, reviewer],
        tasks=[summarize_task, review_task],
        verbose=True,
        tracing=True,
    )

    output = crew.kickoff()
    latency = round(time.time() - start, 2)
    tokens, cost = _extract_usage(crew)

    model = getattr(summarizer, "llm", "groq/llama-3.3-70b-versatile")

    return ScenarioResult(
        source_file=source_file,
        scenario=scenario_name,
        latency_seconds=latency,
        total_tokens=tokens,
        cost=cost,
        model=str(model),
        output=str(output),
    )


def run_opik_learning_lab(input_folder: Path) -> list[ScenarioResult]:
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    text_files = [file for file in input_folder.iterdir() if file.is_file()]
    all_results: list[ScenarioResult] = []

    for file in text_files:
        input_text = file.read_text(encoding="utf-8", errors="ignore")
        for scenario in SCENARIOS:
            result = _run_scenario(
                source_file=file.name,
                input_text=input_text,
                scenario_name=scenario["name"],
                instruction=scenario["instruction"],
            )
            all_results.append(result)

    return all_results


def write_learning_lab_report(results: list[ScenarioResult], report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = report_dir / "opik_learning_lab_report.md"
    json_path = report_dir / "opik_learning_lab_results.json"

    grouped: dict[str, list[ScenarioResult]] = {}
    for result in results:
        grouped.setdefault(result.source_file, []).append(result)

    lines = ["# Opik Learning Lab Report", ""]
    lines.append("This report compares multiple traced CrewAI scenarios per input file.")
    lines.append("")

    for source_file, items in grouped.items():
        lines.append(f"## Source: {source_file}")
        lines.append("")
        lines.append("| Scenario | Model | Latency (s) | Tokens | Cost |")
        lines.append("|---|---|---:|---:|---:|")
        for item in items:
            token_text = str(item.total_tokens) if item.total_tokens is not None else "Unknown"
            cost_text = f"{item.cost:.6f}" if item.cost is not None else "Unknown"
            lines.append(
                f"| {item.scenario} | {item.model} | {item.latency_seconds} | {token_text} | {cost_text} |"
            )

        lines.append("")
        for item in items:
            lines.append(f"### Output: {item.scenario}")
            lines.append("")
            lines.append(item.output)
            lines.append("")

    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
    return markdown_path, json_path
