import os
from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
from loguru import logger
import requests
import json

# LLM + Tools
from langchain_cerebras import ChatCerebras
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import WeatherDataLoader
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from typing import Optional, List, Dict

load_dotenv()


@tool
def shopping_search(query: str, num_results: int = 10) -> str:
    """
    Search for products using Google Shopping via Serper API.
    Returns product information including title, price, source, rating, and link.
    
    Args:
        query: Search query for products (e.g., "nike shoes", "iPhone 15")
        num_results: Number of results to return (default: 10, max: 40)
    
    Returns:
        Formatted product information with prices, ratings, and links
    """
    try:
        url = "https://google.serper.dev/shopping"
        
        payload = json.dumps({
            "q": query,
            "num": min(num_results, 40)  # Limit to 40 max
        })
        
        headers = {
            'X-API-KEY': os.getenv("SERPER_API_KEY"),
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        
        data = response.json()
        shopping_results = data.get("shopping", [])
        
        if not shopping_results:
            return f"❌ No products found for '{query}'."
        
        # Format output
        output = f"🛍️ *Shopping Results for '{query}'*\n\n"
        
        for idx, item in enumerate(shopping_results[:num_results], 1):
            title = item.get("title", "Unknown Product")
            price = item.get("price", "N/A")
            source = item.get("source", "Unknown")
            rating = item.get("rating", "N/A")
            rating_count = item.get("ratingCount", 0)
            link = item.get("link", "")
            
            output += f"{idx}. *{title}*\n"
            output += f"   💵 Price: {price}\n"
            output += f"   🏪 Source: {source}\n"
            
            if rating != "N/A":
                output += f"   ⭐ Rating: {rating}/5 ({rating_count} reviews)\n"
            
            if link:
                output += f"   🔗 {link}\n"
            
            output += "\n"
        
        return output
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error in shopping search: {e}")
        return f"❌ Error connecting to shopping search API: {str(e)}"
    except Exception as e:
        logger.error(f"Error in shopping search: {e}")
        return f"❌ Shopping Search Error: {str(e)}"


@tool
def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: Optional[str] = None
) -> str:
    """
    Search for flights on Kayak using Firecrawl scraping.
    
    Args:
        origin: Departure city or airport code (e.g., "New York" or "JFK")
        destination: Arrival city or airport code (e.g., "Los Angeles" or "LAX")
        departure_date: Departure date in YYYY-MM-DD format
        return_date: Optional return date in YYYY-MM-DD format for round trips
    
    Returns:
        Formatted flight information from Kayak
    """
    try:
        # Build Kayak URL
        origin_clean = origin.replace(" ", "-")
        dest_clean = destination.replace(" ", "-")
        
        if return_date:
            url = f"https://www.kayak.com/flights/{origin_clean}-{dest_clean}/{departure_date}/{return_date}"
        else:
            url = f"https://www.kayak.com/flights/{origin_clean}-{dest_clean}/{departure_date}"
        
        logger.info(f"Scraping Kayak flights: {url}")
        
        # Use FireCrawlLoader from LangChain
        loader = FireCrawlLoader(
            api_key=os.getenv("FIRECRAWL_API_KEY"),
            url=url,
            mode="scrape"
        )
        
        docs = loader.load()
        
        if not docs:
            return f"❌ Could not fetch flight data from Kayak for {origin} to {destination}"
        
        doc = docs[0]
        content = doc.page_content
        metadata = doc.metadata
        
        # Format output
        output = f"✈️ *Flights from {origin} to {destination}*\n"
        output += f"📅 Departure: {departure_date}\n"
        if return_date:
            output += f"📅 Return: {return_date}\n"
        output += f"\n--- Extracted Content ---\n"
        output += f"{content[:2500]}\n\n"  # Limit content
        output += f"🔗 Full details: {url}"
        
        return output
        
    except Exception as e:
        logger.error(f"Error scraping flights: {e}")
        return f"❌ Error fetching flight data: {str(e)}"


@tool
def search_hotels(
    location: str,
    check_in: str,
    check_out: str,
    guests: int = 2
) -> str:
    """
    Search for hotels on Kayak using Firecrawl scraping.
    
    Args:
        location: City or location name (e.g., "Paris" or "New York")
        check_in: Check-in date in YYYY-MM-DD format
        check_out: Check-out date in YYYY-MM-DD format
        guests: Number of guests (default: 2)
    
    Returns:
        Formatted hotel information from Kayak
    """
    try:
        # Build Kayak URL
        location_clean = location.replace(" ", "-")
        url = f"https://www.kayak.com/hotels/{location_clean}/{check_in}/{check_out}/{guests}adults"
        
        logger.info(f"Scraping Kayak hotels: {url}")
        
        # Use FireCrawlLoader from LangChain
        loader = FireCrawlLoader(
            api_key=os.getenv("FIRECRAWL_API_KEY"),
            url=url,
            mode="scrape"
        )
        
        docs = loader.load()
        
        if not docs:
            return f"❌ Could not fetch hotel data from Kayak for {location}"
        
        doc = docs[0]
        content = doc.page_content
        metadata = doc.metadata
        
        # Format output
        output = f"🏨 *Hotels in {location}*\n"
        output += f"📅 Check-in: {check_in}\n"
        output += f"📅 Check-out: {check_out}\n"
        output += f"👥 Guests: {guests}\n\n"
        
        # Add metadata if available
        if metadata.get("title"):
            output += f"📄 Page: {metadata['title']}\n"
        
        output += f"\n--- Extracted Content ---\n"
        output += f"{content[:2500]}\n\n"  # Limit content
        output += f"🔗 Full details: {url}"
        
        return output
        
    except Exception as e:
        logger.error(f"Error scraping hotels: {e}")
        return f"❌ Error fetching hotel data: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city using OpenWeatherMap."""
    try:
        loader = WeatherDataLoader.from_params(
            [city],
            openweathermap_api_key=os.getenv("OPENWEATHERMAP_API_KEY")
        )

        docs = loader.load()
        if not docs:
            return f"No weather data found for {city}."

        data = docs[0].page_content

        return f"🌤 Weather in {city.title()}:\n{data}"

    except Exception as e:
        return f"Error fetching weather: {str(e)}"


# ==========================
# YFINANCE TOOLS
# ==========================

@tool
def get_stock_price(ticker: str) -> str:
    """Get the latest stock price for a ticker symbol like AAPL or TSLA."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.history(period="1d")

        if info.empty:
            return f"No stock data found for '{ticker}'."

        price = info["Close"].iloc[-1]
        return f"📈 {ticker.upper()} Current Price: {price:.2f} USD"

    except Exception as e:
        return f"Error fetching stock price: {str(e)}"


@tool
def get_company_info(ticker: str) -> str:
    """Get company name, sector, and market cap for a given stock ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        name = info.get("longName", "Unknown")
        sector = info.get("sector", "Unknown")
        mc = info.get("marketCap", "N/A")

        return (
            f"🏢 {name}\n"
            f"• Sector: {sector}\n"
            f"• Market Cap: {mc}\n"
        )

    except Exception as e:
        return f"Error fetching company info: {str(e)}"


# ==========================
# LLM MODEL (CEREBRAS)
# ==========================
model = ChatCerebras(
    model="gpt-oss-120b",
    max_tokens=512,
    api_key=os.getenv("CEREBRAS_API_KEY"),
    temperature=0.7,
)


# ==========================
# TAVILY SEARCH TOOL
# ==========================
tavily_tool = TavilySearch(
    max_results=2,
    topic="general",
    api_key=os.getenv("TAVILY_API_KEY")
)


# ==========================
# REGISTER ALL TOOLS
# ==========================
tools = [
    tavily_tool,
    get_stock_price,
    get_company_info,
    get_weather,
    search_flights,
    search_hotels,
    shopping_search,
]


# ==========================
# SYSTEM PROMPT
# ==========================
system_prompt = """
You are Samantha, a smart and helpful AI assistant.
Use TavilySearch for current info and news.
Use YFinance for stocks and company data.
Use Weather tool for current weather.
Use Flights and Hotels tools for travel info (powered by Kayak scraping via Firecrawl).
Use ShoppingSearch for product searches and price comparisons.
Keep answers short, clear, natural, and voice-ready.
"""


# ==========================
# MEMORY
# ==========================
memory = InMemorySaver()


# ==========================
# BUILD THE AGENT
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