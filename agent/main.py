
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

                    # --- Step 1: Search Internal Memory ---
                    print("🧠 Searching memories...")
                    search_result = await session.call_tool('search_memories', {'query': user_input, 'k': 3})
                    try:
                        memories = json.loads(search_result.content[0].text)
                    except (json.JSONDecodeError, IndexError, TypeError):
                        memories = []
                    
                    memory_context = "--- Relevant Memories ---\n"
                    if memories and isinstance(memories, list):
                        valid_mems = [mem.get('content') for mem in memories if mem and 'content' in mem]
                        memory_context += "\n".join([f"- {mem}" for mem in valid_mems]) if valid_mems else "No relevant memories found."
                    else:
                        memory_context += "No relevant memories found."

                    # --- Step 2: Decide if RAG is needed ---
                    print("🤔 Analyzing query for knowledge base...")
                    decision_prompt = f"""You are a topic classifier. Your only job is to determine if the user's query is related to "organic chemistry".
                    Respond with a single JSON object with a boolean key "use_rag".
                    
                    User Query: "{user_input}"
                    
                    JSON Response:"""
                    
                    decision_response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': decision_prompt}], format='json')
                    knowledge_context = ""
                    try:
                        decision = json.loads(decision_response['message']['content'])
                        if decision.get("use_rag", False):
                            print(f"📚 Querying knowledge base for: '{user_input}'...")
                            rag_result = await session.call_tool('query_knowledge_base', {'query': user_input})
                            knowledge = json.loads(rag_result.content[0].text)
                            knowledge_context = f"\n--- External Knowledge ---\n{knowledge.get('result', 'No result found.')}"
                    except (json.JSONDecodeError, IndexError, TypeError) as e:
                        print(f"Could not parse RAG decision response: {e}")

                    # --- Step 3: Synthesize and Respond ---
                    final_prompt = f"""You are a hyper-intelligent assistant. Your single most important duty is to maintain factual accuracy.
                    You have two sources of information: your own memory and an external knowledge base for specific topics.

                    Your primary source of truth is your memory. If the user contradicts it, you MUST correct them.
                    If the user asks about a topic you have external knowledge on, use that to answer.

                    --- MEMORY ---
                    {memory_context}
                    ---
                    {knowledge_context}
                    ---

                    User: {user_input}
                    Assistant:"""
                    
                    print("💡 Synthesizing final response...")
                    response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': final_prompt}])
                    assistant_output = response['message']['content']
                    print(f"\nAssistant: {assistant_output}")

                    # --- Step 4: Save to Memory ---
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

