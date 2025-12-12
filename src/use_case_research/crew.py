import os

from dotenv import load_dotenv
from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.crews.crew_output import CrewOutput
from crewai.project import CrewBase, agent, crew, task
from crewai.mcp import MCPServerHTTP, MCPServerStdio
from crewai_tools import (
	ScrapeWebsiteTool,
    FileWriterTool,
)

load_dotenv()
MCP_GOOGLE_SEARCH_KEY = os.getenv("MCP_GOOGLE_SEARCH_KEY")
GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

file_writer_tool = FileWriterTool()

@CrewBase
class UseCaseResearchCrew:
    """Use Case Research Crew"""

    def _build_llm(self, temperature: float) -> LLM:
        model = os.getenv("MODEL")
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not model:
            raise RuntimeError(
                "GEMINI_MODEL environment variable is not set. Please set it to the Gemini model you want to use (e.g., 'gemini-1.5-flash')."
            )
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. Please export a valid Gemini API key."
            )
        return LLM(model=model, api_key=api_key, temperature=temperature, base_url=base_url)

    
    
    @agent
    def use_case_decomposition_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["use_case_decomposition_analyst"],
            mcps=[
                MCPServerHTTP(
                    url="https://kon-mcp-google-search-805102662749.us-central1.run.app/mcp",
                    headers={"Authorization": f"{MCP_GOOGLE_SEARCH_KEY}"},
                    streamable=True,
                    cache_tools_list=True,
                )
            ],
            reasoning=True,
            max_reasoning_attempts=3,
            inject_date=True,
            allow_delegation=False,
            max_iter=10,
            max_rpm=None,
            max_execution_time=None,
            llm=self._build_llm(temperature=0.5),
        )
    
    @agent
    def decision_framework_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["decision_framework_agent"],
            mcps=[
                MCPServerHTTP(
                    url="https://kon-mcp-google-search-805102662749.us-central1.run.app/mcp",
                    headers={"Authorization": f"{MCP_GOOGLE_SEARCH_KEY}"},
                    streamable=True,
                    cache_tools_list=True,
                )
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=10,
            max_rpm=None,
            max_execution_time=None,
            llm=self._build_llm(temperature=0.7),
        )

    @agent
    def github_agent(self) -> Agent:
        if not GITHUB_PERSONAL_ACCESS_TOKEN:
            raise RuntimeError(
                "GITHUB_PERSONAL_ACCESS_TOKEN environment variable is not set. "
                "Please export a valid GitHub token to enable the GitHub MCP server."
            )

        return Agent(
            config=self.agents_config["github_agent"],
            mcps=[
                MCPServerHTTP(
                    url="https://kon-mcp-google-search-805102662749.us-central1.run.app/mcp",
                    headers={"Authorization": f"{MCP_GOOGLE_SEARCH_KEY}"},
                    streamable=True,
                    cache_tools_list=True,
                )
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=10,
            max_rpm=None,
            max_execution_time=None,
            llm=self._build_llm(temperature=0.3),
        )

    
    @agent
    def report_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["report_agent"],
            tools=[
                file_writer_tool,
            ],
            reasoning=True,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=10,
            max_rpm=None,
            max_execution_time=None,
            llm=self._build_llm(temperature=0.2),
        )
    

    @task
    def decomposition_task(self) -> Task:
        return Task(
            config=self.tasks_config["decomposition_task"],
            markdown=False,
        )
    
    @task
    def decision_framework_task(self) -> Task:
        return Task(
            config=self.tasks_config["decision_framework_task"],
            markdown=False,
        )

    @task
    def github_repo_research_task(self) -> Task:
        return Task(
            config=self.tasks_config["github_repo_research_task"],
            markdown=False,
        )

    
    @task
    def report_task(self) -> Task:
        return Task(
            config=self.tasks_config["report_task"],
            markdown=False,
            context=[self.decomposition_task(), self.decision_framework_task(), self.github_repo_research_task()],
            tools=[file_writer_tool]
        )
    
    

    @crew
    def crew(self) -> Crew:
        """Creates the Use Case Research crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            #after_kickoff_callbacks=[self._tag_output_with_source],
        )

    def _load_response_format(self, name):
        with open(os.path.join(self.base_directory, "config", f"{name}.json")) as f:
            json_schema = json.loads(f.read())

        return SchemaConverter.build(json_schema)
