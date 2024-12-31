# agents/agent_setup.py

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os


def load_instructions(file_path):
    """Load instructions from a file, ensuring it's a list of clean strings."""
    with open(file_path, "r") as file:
        return [line.strip() for line in file if line.strip()]


def create_agent(name, role, tools, instructions):
    """Creates an Agent with specified parameters."""
    return Agent(
        name=name,
        role=role,
        # Use OpenAI's GPT-3.5 Turbo model
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
        tools=tools,
        instructions=instructions,
        show_tools_calls=True,
        markdown=True,
    )


def setup_agents(instruction_path):
    # Load instructions
    web_search_instructions = load_instructions(
        os.path.join(instruction_path, "web_search_instructions.txt"))
    financial_data_instructions = load_instructions(
        os.path.join(instruction_path, "financial_data_instructions.txt"))
    multi_agent_instructions = load_instructions(
        os.path.join(instruction_path, "multi_agent_instructions.txt"))

    # Create Agents
    web_search_agent = create_agent(
        name="Web Search Agent",
        role="I am an AI assistant that can help you with web search and sources.",
        tools=[DuckDuckGo()],
        instructions=web_search_instructions,
    )

    finance_agent = create_agent(
        name="Financial Data Agent",
        role="I am an AI assistant that can help you with financial information.",
        tools=[
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                company_news=True,
            )
        ],
        instructions=financial_data_instructions,
    )

    # Initialize Multi-modal Agent
    multi_ai_agent = Agent(
        team=[web_search_agent, finance_agent],
        instructions=multi_agent_instructions,
        show_tools_calls=True,
        markdown=True,
    )

    return multi_ai_agent
