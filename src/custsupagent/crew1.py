from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = LLM(model="gemini/gemini-2.0-flash", temperature=0)

google_embedder = {
    "provider": "google",
    "config": {
        "model": "models/text-embedding-004",
        "api_key": api_key,
    }
}

content = "Buns and Burger restaurant is a fast food restaurant.It is an online restaurant."

menu_data = JSONKnowledgeSource(
    file_paths=["menudata.json"]
)

restaurants_info = TextFileKnowledgeSource(
    file_path="restaurant_data.txt"
)
# print(restaurants_info)


@CrewBase
class RestaurantCrew():
    """RestaurantCrew crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def Customer_Support_Agent(self) -> Agent:
        return Agent(
            config=self.agents_config['Customer_Support_Agent'],
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

    @task
    def Customer_Support_Task(self) -> Task:
        return Task(
            config=self.tasks_config['Customer_Support_Task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the RestaurantCrew crew"""

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            knowledge_sources=[menu_data, restaurants_info],
            embedder=google_embedder,
        )
