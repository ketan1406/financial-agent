from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Web Search Agent Configuration

web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the required information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always Include Sources"],
    show_tools_calls=True,
    markdown=True,
)

# Financial Data Agent Configuration
finance_agent = Agent(
    name="Financial Data Agent",
    role="",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True,
                         stock_fundamentals=True, company_news=True)],
    instructions=["Use tables to display data"],
    show_tools_calls=True,
    markdown=True,
)

# Multi-modal agent combines the web search and financial data agents
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    instructions=["Always Include Sources", "Use tables to display data"],
    show_tools_calls=True,
    markdown=True,
)

multi_ai_agent.print_response(
    "Summarize the analyst response and share the latest news for NVIDIA", stream=True)
