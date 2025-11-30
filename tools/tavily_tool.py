import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

# ==========================
# TAVILY SEARCH TOOL
# ==========================
tavily_tool = TavilySearch(
    max_results=2,
    topic="general",
    api_key=os.getenv("TAVILY_API_KEY")
)
