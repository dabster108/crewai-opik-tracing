from .crew import run_task

def run():
    print("CrewAI Opik Tracing")
    while True:
        prompt = input("\nEnter text to summarize (or 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        output = run_task("summarize_text", prompt)
        print("\n--- Output ---")
        print(output)

if __name__ == "__main__":
    run()