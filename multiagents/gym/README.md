# 🏋️ AI Gym Assistant

Multi-agentic AI fitness system built with CrewAI

<img width="2556" height="1375" alt="Screenshot from 2025-07-14 17-57-15" src="https://github.com/user-attachments/assets/1ce0f691-ed4d-41e4-b81b-c3aaa139bd69" />

## Features

- **5 Specialized AI Agents:**
  - Interviewer: Conducts fitness assessments
  - Plan Generator: Creates personalized workout plans
  - Nutrition Agent: Offers dietary guidance

- **LLM Support:**
  - Local qwen3:8b via Ollama
  - Google Gemini APIs (free tier)

- **Web Interface:**
  - Modern, responsive chat interface
  - Real-time conversation with AI agents

## Quick Start

1. **Setup:**
   ```bash
   python setup.py
   ```

2. **Run:**
   ```bash
   python main.py
   ```

3. **Visit:** http://localhost:8000

## Configuration

Edit `config.py` or set environment variables:

```bash
# For local LLM (default)
export LLM_PROVIDER=ollama
export LLM_MODEL=qwen3:8b

# For Google AI
export LLM_PROVIDER=google
export LLM_MODEL=gemini-2.5-flash
export GOOGLE_API_KEY=your_key_here

```

## Project Structure

```
gym/
├── main.py              # FastAPI web server
├── gym_crew.py          # CrewAI agents and coordination
├── database.py          # SQLite database handler
├── llm_config.py        # LLM provider configuration
├── config.py            # Environment configuration
├── requirements.txt     # Python dependencies
├── setup.py            # Installation script
├── templates/
│   └── index.html      # Web interface
└── static/             # CSS/JS assets
```

## How It Works

1. **Interview Phase**: AI interviews user about fitness level, goals, equipment, and limitations
2. **Plan Generation**: Creates personalized workout and nutrition plans

## Future Enhancements

- LangGraph integration as alternative to CrewAI
- Mobile app interface
- Advanced progress analytics
- Social features and challenges 
