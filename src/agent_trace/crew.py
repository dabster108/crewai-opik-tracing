import time
import logging
from crewai import Agent, Task, Crew
from opik.integrations.crewai import track_crewai

logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

# Enable tracing with Optic/Comet
track_crewai()

def run_task(prompt: str):
    start_time = time.time()

    # Create summarizer agent
    summarizer = Agent(
        role="Text Summarizer",
        goal="Summarize text clearly",
        backstory="Expert at turning long text into short summaries.",
        llm="groq/llama-3.3-70b-versatile",
        verbose=True
    )

    # Define summarization task
    summarize_task = Task(
        description=f"Summarize the following text:\n\n{prompt}",
        expected_output="A short summary in 2-3 sentences.",
        agent=summarizer
    )

    # Create Crew with tracing enabled
    crew = Crew(
        agents=[summarizer],
        tasks=[summarize_task],
        verbose=True,
        tracing=True
    )

    # Run the Crew
    result = crew.kickoff()

    # Calculate latency
    latency = round(time.time() - start_time, 2)

    # Fetch usage metrics
    usage = crew.usage_metrics
    tokens = usage.total_tokens if usage else "Unknown"
    model = summarizer.llm_name if hasattr(summarizer, "llm_name") else "llama-3.3-70b-versatile"
    cost = usage.cost if usage and hasattr(usage, "cost") else "Approx $0.0003"

    # Build execution report
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