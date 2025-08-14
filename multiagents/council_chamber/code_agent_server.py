#!/usr/bin/env python3
"""
ACP Server with Smolagents CodeAgent - Practice Implementation
Based on L6 tutorial but using ollama/qwen3:4b
"""

from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server
from smolagents import CodeAgent, LiteLLMModel
import nest_asyncio
import os

# Clear proxy settings as per GLOBAL_RULES.md
def clear_proxy_settings():
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]

clear_proxy_settings()
nest_asyncio.apply()

server = Server()

# Using ollama with qwen3:4b instead of OpenAI
model = LiteLLMModel(
    model_id="ollama/qwen3:4b",
    max_tokens=2048
)

@server.agent()
async def code_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    A CodeAgent that writes and executes Python code to solve programming tasks.
    This agent generates complete, functional code solutions.
    """
    agent = CodeAgent(
        tools=[],  # Empty tools list - CodeAgent can still write and execute code
        model=model,
        max_steps=5,  # Limited to 5 steps max
        additional_authorized_imports=["pandas", "numpy", "matplotlib", "requests"],
        verbosity_level=1
    )
    
    prompt = input[0].parts[0].content
    response = agent.run(prompt)
    
    yield Message(parts=[MessagePart(content=str(response))])

if __name__ == "__main__":
    print("🚀 Starting ACP Code Agent Server on port 8000...")
    server.run(port=8000)