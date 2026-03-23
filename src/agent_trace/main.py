from pathlib import Path
from .crew import run_task
from .opik_learning_lab import run_opik_learning_lab, write_learning_lab_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = PROJECT_ROOT / "tests"
REPORT_DIR = PROJECT_ROOT / "Report"


def run():
    print("CrewAI Agent Trace Demo")

    file_path = TESTS_DIR

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


def run_with_trigger():
    print("Opik Learning Lab")

    if not TESTS_DIR.exists():
        print("tests directory not found")
        return

    results = run_opik_learning_lab(TESTS_DIR)
    markdown_path, json_path = write_learning_lab_report(results, REPORT_DIR)

    print("\nLearning lab finished.")
    print(f"Markdown report: {markdown_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    run()