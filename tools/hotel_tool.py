import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from loguru import logger

load_dotenv()

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
