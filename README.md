# UntitledProject Crew

Welcome to the UntitledProject Crew project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```
### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/untitled_project/config/agents.yaml` to define your agents
- Modify `src/untitled_project/config/tasks.yaml` to define your tasks
- Modify `src/untitled_project/crew.py` to add your own logic, tools and specific args
- Modify `src/untitled_project/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```

This command initializes the untitled_project Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

### Testing the Notion MCP server independently

If you want to confirm the Docker-based Notion MCP server is working outside CrewAI, you can run the standalone tester (depends on the `mcp` package, install once via `uv pip install mcp`):

```bash
export INTERNAL_INTEGRATION_TOKEN=ntn_...
uv run python scripts/notion_mcp_test.py
```

The script starts the `mcp/notion` container, lists available tools, and (when available) calls `notion.list_workspaces`. Use this to validate your token or Docker setup before running the full crew.

### Tracking API usage

Every time you run `crewai run`, the script now prints a usage summary (successful requests and token counts) and stores a timestamped JSON report inside the `reports/` directory. You can inspect these files to monitor how many Gemini API calls the crew makes and how many tokens were consumed per run.

## Understanding Your Crew

The untitled_project Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

## Support

For support, questions, or feedback regarding the UntitledProject Crew or crewAI.
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.
