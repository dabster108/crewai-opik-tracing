import logging
from crewai import Agent, Task, Crew
from opik.integrations.crewai import track_crewai
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
track_crewai()


def run_task(task_name: str, prompt: str) -> str:
    agent = Agent(
        role="Text Summarizer",
        goal="Summarize the given text clearly and concisely",
        backstory="You are an expert at distilling long content into short, clear summaries.",
        llm="groq/llama-3.3-70b-versatile",
        verbose=True,
    )

    task = Task(
        description=f"Summarize the following text:\n\n{prompt}",
        expected_output="A concise summary in 2-3 sentences.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    return str(result)