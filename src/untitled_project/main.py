#!/usr/bin/env python
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import agentops
import os
from untitled_project.crew import UntitledProjectCrew

# This main file is intended to be a way for your to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")
agentops.init(api_key=AGENTOPS_API_KEY, default_tags=['crewai'])


def run():
    """
    Run the crew.
    """
    # Interactive prompt for use case input
    use_case_description = input("Enter use case description: ")
    industry = input("Enter industry: ")
    inputs = {
        "use_case_description": use_case_description,
        "industry": industry,
    }
    crew = UntitledProjectCrew().crew()
    crew.kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'use_case_description': 'sample_value',
        'industry': 'sample_value',
        'company_size': 'sample_value',
        'tech_stack': 'sample_value',
        'budget_range': 'sample_value',
        'timeline': 'sample_value',
        'strategic_priority': 'sample_value'
    }
    try:
        UntitledProjectCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        UntitledProjectCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        'use_case_description': 'sample_value',
        'industry': 'sample_value',
        'company_size': 'sample_value',
        'tech_stack': 'sample_value',
        'budget_range': 'sample_value',
        'timeline': 'sample_value',
        'strategic_priority': 'sample_value'
    }
    try:
        UntitledProjectCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py <command> [<args>]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "run":
        run()
    elif command == "train":
        train()
    elif command == "replay":
        replay()
    elif command == "test":
        test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
