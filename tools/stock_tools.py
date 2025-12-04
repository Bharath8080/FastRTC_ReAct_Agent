import yfinance as yf
from langchain.tools import tool

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
