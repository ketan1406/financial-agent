# agents/agent_setup.py

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
import logging


def load_instructions(file_path):
    """Load instructions from a file as a single string."""
    with open(file_path, "r") as file:
        return file.read()


def create_agent(name, role, tools, instructions):
    """Creates an Agent with specified parameters."""
    logging.info(f"Creating agent: {name}")
    try:
        agent = Agent(
            name=name,
            role=role,
            model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
            tools=tools,
            instructions=instructions,
            show_tools_calls=False,  # Disable tool call display for cleaner responses
            markdown=False,  # Disable markdown to prevent escape codes
        )
        logging.info(f"Agent {name} created successfully.")
        return agent
    except Exception as e:
        logging.error(f"Failed to create agent {name}: {e}")
        raise


def setup_agents(instruction_path):
    # Load instructions
    web_search_instructions = load_instructions(
        os.path.join(instruction_path, "web_search_instructions.txt"))
    financial_data_instructions = load_instructions(
        os.path.join(instruction_path, "financial_data_instructions.txt"))
    multi_agent_instructions = load_instructions(
        os.path.join(instruction_path, "multi_agent_instructions.txt"))

    # Create Tools with both search and news enabled
    duck_tool = DuckDuckGo(search=True, news=True)
    finance_tool = YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_news=True,
    )

    # Create Agents
    web_search_agent = create_agent(
        name="Web Search Agent",
        role="I am an AI assistant that can help you with web search and sources.",
        tools=[duck_tool],
        instructions=web_search_instructions,
    )

    finance_agent = create_agent(
        name="Financial Data Agent",
        role="I am an AI assistant that can help you with financial information.",
        tools=[finance_tool],
        instructions=financial_data_instructions,
    )

    # Initialize Multi-modal Agent
    multi_ai_agent = Agent(
        team=[web_search_agent, finance_agent],
        instructions=multi_agent_instructions,
        show_tools_calls=False,
        markdown=False,
    )

    return multi_ai_agent
