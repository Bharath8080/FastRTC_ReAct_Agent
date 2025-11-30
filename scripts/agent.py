import os
from dotenv import load_dotenv
from loguru import logger
from langchain_cerebras import ChatCerebras
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

# Import tools from the new tools package
from tools.tavily_tool import tavily_tool
from tools.stock_tools import get_stock_price, get_company_info
from tools.weather_tool import get_weather
from tools.flight_tool import search_flights
from tools.hotel_tool import search_hotels

load_dotenv()

# ==========================
# 1. LLM MODEL (CEREBRAS)
# ==========================
model = ChatCerebras(
    model="gpt-oss-120b",      # Low latency, strong model
    max_tokens=512,
    api_key=os.getenv("CEREBRAS_API_KEY"),
    temperature=0.7,
)

# ==========================
# 2. REGISTER ALL TOOLS
# ==========================
tools = [
    tavily_tool,
    get_stock_price,
    get_company_info,
    get_weather,
    search_flights,
    search_hotels
]

# ==========================
# 3. SYSTEM PROMPT
# ==========================
system_prompt = """
You are Samantha, a helpful AI agent.
Use TavilySearch for general or current information.
Use YFinance tools for stock prices and company financials.
Use Weather tool for current weather.
Use Flights tool for flight options with the help of SerpAPI.
Use Hotels tool for hotel options with the help of SerpAPI.
Keep responses short, natural, and suitable for voice interaction.
"""

# ==========================
# 4. MEMORY
# ==========================
memory = InMemorySaver()

# ==========================
# 5. BUILD THE AGENT
# ==========================
agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=system_prompt,
    checkpointer=memory,
)

# Config
agent_config = {
    "configurable": {
        "thread_id": "default_user"
    }
}
