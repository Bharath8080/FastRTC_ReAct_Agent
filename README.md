# 🎙️ FastRTC Voice Agent

A high-performance, ultra-low latency voice assistant built with **FastRTC**, **LangGraph**, and **Cerebras**. This agent supports real-time voice interaction with advanced capabilities including web search, flight & hotel booking, stock market analysis, and shopping comparisons.

## 🚀 Features

- **Ultra-Low Latency**: Powered by FastRTC for real-time voice streaming.
- **Dual Backend Support**:
  - **Cartesia**: High-quality Sonic 3 TTS and native STT (`app.py`).
  - **Groq**: Fast Whisper STT and local Kokoro TTS (`main.py`).
- **Intelligent Agent**: Built with **LangGraph** and **Cerebras (GPT-OSS-120B)** for smart, context-aware responses.
- **Rich Toolset**:
  - � **Web Search**: Real-time information via **Tavily**.
  - ✈️ **Travel**: Flight and Hotel search using **Kayak** (via **Firecrawl**).
  - 🛍️ **Shopping**: Product search and price comparison via **Serper** (Google Shopping).
  - 📈 **Finance**: Real-time stock data and company info via **YFinance**.
  - �️ **Weather**: Current weather updates via **OpenWeatherMap**.
- **Multi-Interface**:
  - 🌈 **Gradio UI**: Beautiful web interface for desktop use.
  - 📞 **Phone Interface**: Connect via a temporary phone number.

## 🛠️ Tech Stack

- **Framework**: [FastRTC](https://github.com/fastrtc/fastrtc)
- **LLM**: [Cerebras](https://cerebras.net/) (gpt-oss-120b)
- **Agent Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/)
- **Speech-to-Text (STT)**: Cartesia / Groq Whisper
- **Text-to-Speech (TTS)**: Cartesia Sonic 3 / FastRTC Kokoro
- **Tools**: Tavily, Firecrawl, Serper, YFinance, OpenWeatherMap

## 📋 Prerequisites

- Python 3.10 or higher
- API Keys for the following services:
  - [Cerebras](https://cerebras.net/)
  - [Cartesia](https://cartesia.ai/) (for `app.py`)
  - [Groq](https://groq.com/) (for `main.py`)
  - [Tavily](https://tavily.com/)
  - [Firecrawl](https://firecrawl.dev/)
  - [Serper](https://serper.dev/)
  - [OpenWeatherMap](https://openweathermap.org/)

## � Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd fastrtc-groq-voice-agent01
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuration

Create a `.env` file in the root directory and add your API keys:

```ini
# Core
CEREBRAS_API_KEY=your_cerebras_key

# Voice Services
CARTESIA_API_KEY=your_cartesia_key
GROQ_API_KEY=your_groq_key

# Tools
TAVILY_API_KEY=your_tavily_key
FIRECRAWL_API_KEY=your_firecrawl_key
SERPER_API_KEY=your_serper_key
OPENWEATHERMAP_API_KEY=your_openweathermap_key
```

## � Usage

### Option 1: Cartesia Backend (Recommended for Quality)
Uses Cartesia for both STT and TTS (Sonic 3).

```bash
python app.py
```

### Option 2: Groq Backend (Recommended for Speed/Cost)
Uses Groq Whisper for STT and local Kokoro TTS.

```bash
python main.py
```

### Phone Interface
To launch with a temporary phone number interface, add the `--phone` flag:

```bash
python app.py --phone
# or
python main.py --phone
```

## 📂 Project Structure

```
├── app.py                 # Main entry point (Cartesia backend)
├── main.py                # Alternative entry point (Groq backend)
├── websearch_agent.py     # LangGraph agent definition and tools
├── tools/                 # Tool implementations
│   ├── flight_tool.py     # Kayak flight search
│   ├── hotel_tool.py      # Kayak hotel search
│   ├── shop.py            # Google Shopping search
│   ├── stock_tools.py     # YFinance stock tools
│   ├── tavily_tool.py     # Tavily web search
│   └── weather_tool.py    # OpenWeatherMap tool
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## � License

This project is licensed under the MIT License.
