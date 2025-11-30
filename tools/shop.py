import os
import json
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from loguru import logger

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
