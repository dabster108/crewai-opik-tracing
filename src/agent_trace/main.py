from pathlib import Path
from .crew import run_task


def run():
    print("CrewAI Agent Trace Demo")

    file_path = Path("tests/dummy.txt")

    if not file_path.exists():
        print("tests/dummy.txt not found")
        return

    text = file_path.read_text()

    summary, report = run_task(text)

    print("\nSummary:\n")
    print(summary)

    print("\nExecution Report:\n")
    print(report)


if __name__ == "__main__":
    run()