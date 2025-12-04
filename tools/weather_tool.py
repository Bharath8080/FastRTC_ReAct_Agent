import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WeatherDataLoader
from langchain.tools import tool

load_dotenv()

# ==========================
# WEATHER TOOL
# ==========================

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city using OpenWeatherMap."""
    try:
        loader = WeatherDataLoader.from_params(
            [city],
            openweathermap_api_key=os.getenv("OPENWEATHERMAP_API_KEY")
        )

        docs = loader.load()  # returns list of Document objects
        if not docs:
            return f"No weather data found for {city}."

        data = docs[0].page_content  # raw text

        return f"🌤 Weather in {city.title()}:\n{data}"

    except Exception as e:
        return f"Error fetching weather: {str(e)}"
