from pathlib import Path
from .crew import run_task


def run():
    print("CrewAI Agent Trace Demo")

    file_path = Path("/Users/dikshanta/Documents/crewai-opik-tracing/agent_trace/tests")

    if not file_path.exists():
        print("tests directory not found")
        return

    for file in file_path.iterdir():
        if file.is_file():
            print(f"\nProcessing: {file.name}")

            text = file.read_text()

            summary, report = run_task(text)

            print("\nSummary:\n")
            print(summary)

            print("\nExecution Report:\n")
            print(report)


if __name__ == "__main__":
    run()