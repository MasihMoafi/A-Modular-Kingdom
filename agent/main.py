
# interactive_agent.py
import os
import sys
import asyncio
import nest_asyncio
import traceback
import json

# --- Initial Setup ---
def clear_proxy_settings():
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]
clear_proxy_settings()

import ollama
from mcp import ClientSession, stdio_client, StdioServerParameters

# --- Get the absolute path to the host.py script ---
# This makes the agent runnable from any directory.
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
HOST_PATH = os.path.join(AGENT_DIR, "host.py")

nest_asyncio.apply()
LLM_MODEL = 'qwen3:8b'

async def main():
    print("--- Intelligent Agent ---")
    params = StdioServerParameters(command=sys.executable, args=["-u", HOST_PATH])
    
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # Wait for the host to initialize
            await session.initialize()
            print("\n✅ Agent is ready. Type 'exit' to quit.")
            
            while True:
                try:
                    user_input = await asyncio.to_thread(input, "\n> ")
                    if user_input.lower() == 'exit':
                        break
                    if not user_input.strip():
                        continue

                    if user_input.lower() == '/memory':
                        print("🔍 Inspecting memory...")
                        mems = await session.call_tool('list_all_memories')
                        try:
                            all_memories = json.loads(mems.content[0].text)
                            print("\n--- AGENT'S CURRENT MEMORY ---")
                            if all_memories:
                                for i, mem in enumerate(all_memories):
                                    print(f"{i+1}. (ID: {mem.get('id', 'N/A')[:8]}) - {mem.get('content', 'N/A')}")
                            else:
                                print("Memory is empty.")
                            print("---------------------------------")
                        except (json.JSONDecodeError, IndexError, TypeError) as e:
                            print(f"Could not parse memory response: {e}")
                        continue
                    
                    # 1. SEARCH: Get relevant memories.
                    print("🧠 Searching memories...")
                    search_result = await session.call_tool('search_memories', {'query': user_input, 'k': 3})
                    
                    try:
                        memories = json.loads(search_result.content[0].text)
                    except (json.JSONDecodeError, IndexError, TypeError):
                        memories = []

                    # Build the context for the LLM
                    memory_context = "--- Relevant Memories ---\n"
                    if memories and isinstance(memories, list):
                        # Filter out potential errors and format valid memories
                        valid_mems = [mem.get('content') for mem in memories if mem and 'content' in mem]
                        if valid_mems:
                            memory_context += "\n".join([f"- {mem}" for mem in valid_mems])
                        else:
                            memory_context += "No relevant memories found."
                    else:
                        memory_context += "No relevant memories found."

                    # 2. CHAT: Formulate the prompt and get the assistant's response.
                    prompt = f"""You are a hyper-intelligent assistant with a perfect, persistent memory. Your single most important duty is to maintain factual accuracy based on your memory.

Below is a list of facts you know to be true. This is your source of truth.

--- MEMORY ---
{memory_context}
---

A user is interacting with you. Your task is to respond to them, following these strict rules:
1.  If the user's message contradicts a fact in your memory, you MUST correct them. State the fact from your memory clearly.
2.  If the user's message asks a question, answer it using the facts from your memory.
3.  NEVER, under any circumstances, repeat or validate information from the user that you know to be false based on your memory.

User: {user_input}
Assistant:"""
                    
                    print("🤔 Thinking...")
                    response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}])
                    assistant_output = response['message']['content']
                    print(f"\nAssistant: {assistant_output}")

                    # 3. SAVE: Save the conversation turn to be processed by the memory system.
                    print("📝 Saving conversation to memory...")
                    conversation_turn = f"User: {user_input}\nAssistant: {assistant_output}"
                    await session.call_tool(
                        'save_fact',
                        arguments={'fact_data': {'content': conversation_turn}}
                    )

                except Exception as e:
                    print(f"\n--- An Error Occurred in the Loop ---", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception:
        print("\n--- A FATAL ERROR OCCURRED ---")
        traceback.print_exc()

