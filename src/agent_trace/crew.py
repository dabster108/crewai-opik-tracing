import time
import logging
from crewai import Agent, Task, Crew
from opik.integrations.crewai import track_crewai

logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

track_crewai()


def run_task(prompt: str):

    start_time = time.time()

    summarizer = Agent(
        role="Text Summarizer",
        goal="Summarize text clearly",
        backstory="Expert at turning long text into short summaries.",
        llm="groq/llama-3.3-70b-versatile",
        verbose=True
    )

    summarize_task = Task(
        description=f"Summarize the following text:\n\n{prompt}",
        expected_output="A short summary in 2-3 sentences.",
        agent=summarizer
    )

    crew = Crew(
        agents=[summarizer],
        tasks=[summarize_task],
        verbose=True,
        tracing=True
    )

    result = crew.kickoff()

    latency = round(time.time() - start_time, 2)

    usage = crew.usage_metrics

    tokens = usage.total_tokens if usage else "Unknown"

    model = "llama-3.3-70b-versatile"

    cost = "Approx $0.0003"

    execution_report = f"""
Execution Report

Agent: Text Summarizer
Model: {model}
Tokens Used: {tokens}
Latency: {latency}s
Cost: {cost}

Task Status: Completed
"""

    return result, execution_report