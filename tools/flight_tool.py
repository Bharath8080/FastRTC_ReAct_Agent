import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from loguru import logger
from typing import Optional

load_dotenv()

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
