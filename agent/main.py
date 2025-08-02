#!/usr/bin/env python
# coding: utf-8

import os
import sys
import asyncio
import nest_asyncio
import traceback
import json
from typing import List, Optional
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

# --- Initial Setup ---
def clear_proxy_settings():
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]
clear_proxy_settings()

import ollama
from mcp import ClientSession, stdio_client, StdioServerParameters

# --- Get the absolute path to the host.py script ---
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
HOST_PATH = os.path.join(AGENT_DIR, "host.py")

nest_asyncio.apply()
LLM_MODEL = 'qwen3:8b'

class DocumentCompleter(Completer):
    def __init__(self):
        self.resources = []
        self.commands = ['/memory', '/help', '/tools', '/files']
    
    def update_resources(self, resources: List[str]):
        self.resources = resources
    
    def get_completions(self, document, complete_event):
        text_before_cursor = document.text_before_cursor
        
        # Handle slash commands
        if text_before_cursor.startswith('/'):
            prefix = text_before_cursor[1:]
            for cmd in self.commands:
                if cmd[1:].startswith(prefix.lower()):
                    yield Completion(
                        cmd[1:],
                        start_position=-len(prefix),
                        display=cmd,
                        display_meta="Command",
                    )
            return
        
        # Handle @ mentions
        if "@" in text_before_cursor:
            last_at_pos = text_before_cursor.rfind("@")
            prefix = text_before_cursor[last_at_pos + 1:]
            
            for resource_id in self.resources:
                if resource_id.lower().startswith(prefix.lower()):
                    yield Completion(
                        resource_id,
                        start_position=-len(prefix),
                        display=resource_id,
                        display_meta="Document",
                    )

async def main():
    print("--- Intelligent Agent ---")
    params = StdioServerParameters(command=sys.executable, args=["-u", HOST_PATH])
    
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Setup prompt_toolkit with dropdown
            completer = DocumentCompleter()
            kb = KeyBindings()
            
            @kb.add("@")
            def _(event):
                buffer = event.app.current_buffer
                buffer.insert_text("@")
                if buffer.document.is_cursor_at_the_end:
                    buffer.start_completion(select_first=False)
                    
            @kb.add("/")
            def _(event):
                buffer = event.app.current_buffer
                if buffer.document.is_cursor_at_the_end and not buffer.text:
                    buffer.insert_text("/")
                    buffer.start_completion(select_first=False)
                else:
                    buffer.insert_text("/")
            
            prompt_session = PromptSession(
                completer=completer,
                key_bindings=kb,
                style=Style.from_dict({
                    "prompt": "#aaaaaa",
                    "completion-menu.completion": "bg:#222222 #ffffff",
                    "completion-menu.completion.current": "bg:#444444 #ffffff",
                }),
                complete_while_typing=True,
                complete_in_thread=True,
            )
            
            # Load available documents for dropdown
            try:
                available_docs = await session.read_resource("docs://documents")
                doc_list = json.loads(available_docs.contents[0].text)
                completer.update_resources(doc_list)
                print(f"\nLoaded {len(doc_list)} documents for @ completion")
            except Exception as e:
                print(f"Could not load document list: {e}")
            
            print("\nAgent is ready. Type 'exit' to quit. Use @ to see document dropdown.")
            
            while True:
                try:
                    user_input = await prompt_session.prompt_async("\n> ")
                    if user_input.lower() == 'exit':
                        break
                    if not user_input.strip():
                        continue

                    # Handle # for direct memory saving
                    if user_input.startswith('#'):
                        content_to_save = user_input[1:].strip()
                        if content_to_save:
                            print("📝 Saving directly to memory...")
                            await session.call_tool('save_direct_memory', {'content': content_to_save})
                            print("✅ Saved to memory!")
                        continue

                    # Handle slash commands
                    if user_input.startswith('/'):
                        command = user_input[1:].lower()
                        
                        if command == 'help':
                            print("""Available commands:
- /help - Show this help
- /tools - List all available tools  
- /memory - List and manage memories
- /files - List available files
- @filename - Access file content (e.g., @Napoleon.pdf)
- #message - Save message directly to memory""")
                            continue
                            
                        elif command == 'tools':
                            print("""Available MCP Tools:
1. query_knowledge_base(query: str) - Search the RAG knowledge base
2. search_memories(query: str, top_k: int = 3) - Search memory database
3. save_direct_memory(content: str) - Save content directly to memory
4. delete_memory(memory_id: str) - Delete memory by ID
5. list_all_memories() - List all memories in database
6. web_search(query: str) - Perform web search""")
                            continue
                            
                        elif command == 'memory':
                            print("🔍 Inspecting memory...")
                            mems = await session.call_tool('list_all_memories')
                            try:
                                all_memories = json.loads(mems.content[0].text)
                                print("\n--- AGENT'S CURRENT MEMORY ---")
                                if all_memories:
                                    for i, mem in enumerate(all_memories):
                                        print(f"{i+1}. (ID: {mem.get('id', 'N/A')[:8]}) - {mem.get('content', 'N/A')}")
                                    
                                    # Interactive deletion
                                    delete_input = await prompt_session.prompt_async(
                                        "\nEnter memory ID to delete (or press Enter to skip): "
                                    )
                                    
                                    if delete_input.strip():
                                        # Find matching memory
                                        matching_mem = None
                                        for mem in all_memories:
                                            if mem.get('id', '').startswith(delete_input.strip()):
                                                matching_mem = mem
                                                break
                                        
                                        if matching_mem:
                                            confirm = await prompt_session.prompt_async(
                                                f"Delete '{matching_mem.get('content', '')[:50]}...'? (y/N): "
                                            )
                                            if confirm.lower() == 'y':
                                                result = await session.call_tool('delete_memory', {'memory_id': matching_mem['id']})
                                                print("Memory deleted successfully!")
                                        else:
                                            print("Memory ID not found.")
                                else:
                                    print("Memory is empty.")
                                print("---------------------------------")
                            except (json.JSONDecodeError, IndexError, TypeError) as e:
                                print(f"Could not parse memory response: {e}")
                            continue
                            
                        elif command == 'files':
                            print("📁 Available files:")
                            try:
                                result = await session.read_resource("docs://documents")
                                files = json.loads(result.contents[0].text)
                                if files:
                                    for i, file in enumerate(files, 1):
                                        print(f"{i}. {file}")
                                else:
                                    print("No files found.")
                            except Exception as e:
                                print(f"Error listing files: {e}")
                            continue
                        else:
                            print(f"Unknown command: /{command}. Type /help for available commands.")
                            continue

                    # --- Step 1: Process @ mentions for document references ---
                    document_context = ""
                    mentions = [word[1:] for word in user_input.split() if word.startswith("@")]
                    
                    if mentions:
                        print(f"Found document mentions: {mentions}")
                        try:
                            # Get list of available documents
                            available_docs = await session.read_resource("docs://documents")
                            doc_list = json.loads(available_docs.contents[0].text)
                            
                            for mention in mentions:
                                if mention in doc_list:
                                    print(f"Fetching content for: {mention}")
                                    doc_resource = await session.read_resource(f"docs://documents/{mention}")
                                    content = doc_resource.contents[0].text
                                    document_context += f'\n<document id="{mention}">\n{content}\n</document>\n'
                        except Exception as e:
                            print(f"Could not fetch document resources: {e}")

                    # --- Step 2: Search Internal Memory ---
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

                    # --- Step 3: Decide which tool to use (if any) ---
                    # Skip tool decision if we already have document context from @ mentions
                    external_context = ""
                    if not document_context:
                        print("🤔 Analyzing query for tool use...")
                        decision_prompt = f"""You are a tool-use decision engine. Your job is to decide which tool, if any, is appropriate for the user's query. You have three choices:
1.  `rag`: If the query is about documents, knowledge from files, or general information that might be in our knowledge base.
2.  `web_search`: If the query requires up-to-date information, current events, or real-time data.
3.  `none`: If the query is personal, conversational, or can be answered from memory alone.

Respond with a single JSON object with one key: "tool", and the value as one of ["rag", "web_search", "none"].

User Query: "{user_input}"

JSON Response:"""
                        
                        decision_response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': decision_prompt}], format='json')
                        try:
                            decision = json.loads(decision_response['message']['content'])
                            tool_to_use = decision.get("tool")

                            if tool_to_use == "rag":
                                print(f"📚 Querying knowledge base for: '{user_input}'...")
                                rag_result = await session.call_tool('query_knowledge_base', {'query': user_input})
                                knowledge = json.loads(rag_result.content[0].text)
                                external_context = f"\n--- External Knowledge (Books) ---\n{knowledge.get('result', 'No result found.')}"
                            
                            elif tool_to_use == "web_search":
                                print(f"🌐 Performing web search for: '{user_input}'...")
                                search_result = await session.call_tool('web_search', {'query': user_input})
                                results = json.loads(search_result.content[0].text)
                                external_context = f"\n--- External Knowledge (Web) ---\n{results.get('results', 'No results found.')}"

                        except (json.JSONDecodeError, IndexError, TypeError) as e:
                            print(f"Could not parse tool decision response: {e}")
                    else:
                        print("Using document context from @ mentions, skipping tool selection.")

                    # --- Step 4: Synthesize and Respond ---
                    final_prompt = f"""You are a hyper-intelligent assistant. Your single most important duty is to maintain factual accuracy.
                    You have access to your personal memory, external knowledge base, web search, and can reference specific documents.

                    Your primary source of truth is your memory. If the user contradicts it, you MUST correct them.
                    Use the provided information sources to answer questions when appropriate.

                    --- MEMORY ---
                    {memory_context}
                    ---
                    {document_context if document_context else external_context}
                    ---

                    Note: If the user's query contains references to documents like "@Napoleon.pdf", the "@" is only a way of mentioning the doc. 
                    The actual document content (if available) is provided above. Answer directly and concisely using the provided information.

                    User: {user_input}
                    Assistant:"""
                    
                    print("💡 Synthesizing final response...")
                    response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': final_prompt}])
                    assistant_output = response['message']['content']
                    print(f"\nAssistant: {assistant_output}")
                    

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