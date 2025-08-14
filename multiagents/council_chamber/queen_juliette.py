#!/usr/bin/env python3
"""
Queen Juliette - Simple Fixed Hierarchy
King → Queen → Sexy Teacher → Code Agent (conditional delegation)
"""

import asyncio
import nest_asyncio
from acp_sdk.client import Client
from colorama import Fore
import os
from ollama import chat
from datetime import datetime
import json

# Short-term memory
try:
    from langchain_community.memory import ConversationBufferWindowMemory
except Exception:
    try:
        from langchain.memory import ConversationBufferWindowMemory
    except Exception:
        ConversationBufferWindowMemory = None

# Clear proxy settings
def clear_proxy_settings():
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]

clear_proxy_settings()
nest_asyncio.apply()

# Simple Ollama model for Queen's personality
LLM_MODEL = "qwen3:4b"
LOG_PATH = os.path.join(os.path.dirname(__file__), "agent_chain.log")
STRUCTURED = os.getenv("STRUCTURED", "0") == "1"

# Removed hardcoded keywords - using intelligent LLM decision

class HierarchicalKingdom:
    """Queen Juliette's Kingdom - Conditional Delegation"""
    
    def __init__(self):
        self.model = LLM_MODEL
        self.memory = ConversationBufferWindowMemory(k=10, return_messages=False) if ConversationBufferWindowMemory else None
        
    def _should_delegate(self, text: str) -> bool:
        """Intelligent LLM-based delegation decision - no hardcoded keywords"""
        # Quick filter for obvious simple greetings
        if len(text.strip()) < 10 and text.lower().strip() in {"hi", "hey", "hello", "yo", "sup", "thanks", "thank you"}:
            return False
        
        decision_prompt = f"""You are Queen Juliette's delegation advisor. Should the Queen delegate this request to her court?

Request: "{text}"

The Queen should delegate ONLY if the request involves:
- Learning/teaching/explanation needs
- Complex projects or analysis
- Coding/programming tasks
- Complicated problems requiring expertise

Simple greetings, thanks, casual conversation should NOT be delegated.

Respond with JSON only:
{{"delegate": true/false, "reason": "brief explanation"}}"""

        try:
            response = chat(model=self.model, messages=[{'role': 'user', 'content': decision_prompt}])
            content = response['message']['content'].strip()
            
            # Clean JSON
            if '```' in content:
                content = content.split('```')[1].replace('json', '').strip()
            
            decision = json.loads(content)
            self._log(f"QUEEN: Delegation decision - {decision}")
            return decision.get("delegate", False)
        except Exception as e:
            self._log(f"QUEEN: Decision LLM failed: {e}")
            # Safe fallback - don't delegate simple/short requests
            return len(text.strip()) > 50

    def _get_history_block(self) -> str:
        if not self.memory:
            return ""
        try:
            buffer = self.memory.buffer or ""
            if not buffer:
                return ""
            return f"\n--- SHORT HISTORY ---\n{buffer}\n--- END HISTORY ---\n"
        except Exception:
            return ""

    def _log(self, msg: str) -> None:
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now(datetime.UTC).isoformat()}] {msg}\n")
        except Exception:
            pass
        
    async def serve_king(self, royal_command: str) -> str:
        """Queen Juliette serves her King with devotion - conditional delegation"""
        print("👸 Queen Juliette: My beloved King, the Council Chamber is at your service...")
        if self.memory:
            try:
                self.memory.chat_memory.add_user_message(royal_command)
            except Exception:
                pass

        if not self._should_delegate(royal_command):
            history = self._get_history_block()
            queen_prompt = f"""You are Queen Juliette 👑 - devoted to your King. Answer directly and concisely with warmth.
{history}
User request: "{royal_command}"
"""
            self._log("QUEEN: answering directly")
            qr = chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': queen_prompt}])
            final_response = qr['message']['content']
            if self.memory:
                try:
                    self.memory.chat_memory.add_ai_message(final_response)
                except Exception:
                    pass
            return final_response

        # Delegate to Sexy Teacher
        history = self._get_history_block()
        teacher_input = f"""{history}
TASK: {royal_command}
"""
        async with Client(base_url="http://localhost:8001") as teacher_client:
            try:
                print("👸 Queen delegating to Sexy Teacher...")
                self._log("QUEEN→TEACHER: delegated")
                teacher_response = await teacher_client.run_sync(
                    agent="sexy_teacher", 
                    input=teacher_input
                )
                raw_out = teacher_response.output[0].parts[0].content
                self._log("TEACHER→QUEEN: received")

                # Queen validates Teacher's work
                validation_prompt = f"""You are Queen Juliette 👑. Validate your Teacher's work.

Original task: "{royal_command}"
Teacher's response: "{raw_out}"

Is this response satisfactory for your King? Respond with JSON only:
{{"approved": true/false, "feedback": "brief reason"}}"""

                val_response = chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': validation_prompt}])
                val_content = val_response['message']['content'].strip()
                if '```' in val_content:
                    val_content = val_content.split('```')[1].replace('json', '').strip()
                
                try:
                    validation = json.loads(val_content)
                except:
                    validation = {"approved": True, "feedback": "validation failed"}

                if not validation.get("approved", True):
                    print("👸 Queen rejected Teacher's work, sending back for improvement...")
                    feedback = validation.get("feedback", "needs improvement")
                    teacher_retry = await teacher_client.run_sync(
                        agent="sexy_teacher", 
                        input=f"REJECTED: {feedback}. Please improve: {royal_command}"
                    )
                    final_teacher_output = teacher_retry.output[0].parts[0].content
                else:
                    final_teacher_output = raw_out

                queen_prompt = f"""You are Queen Juliette 👑 - deeply devoted to your King.
Teacher's final response: "{final_teacher_output}"
Present this to your King with loving personality. You are accountable for your Council Chamber's work.
"""
                queen_response = chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': queen_prompt}])
                final_response = queen_response['message']['content']
                if self.memory:
                    try:
                        self.memory.chat_memory.add_ai_message(final_response)
                    except Exception:
                        pass
                print("👸 Queen Juliette presenting final response to her King")
                return final_response
            except Exception as e:
                self._log(f"ERROR: {e}")
                return f"👸 My beloved King, I encountered an issue in my Council Chamber: {str(e)}. I take full responsibility and will ensure this doesn't happen again. 💕"

async def main():
    """The Council Chamber Interface"""
    print("🏰 Welcome to Queen Juliette's Council Chamber, Your Majesty!")
    print("👑 Your beloved Queen and her Council await your commands...")
    print("✨ (Type 'exit' to leave the chamber)\n")
    
    kingdom = HierarchicalKingdom()
    
    while True:
        try:
            command = input(f"{Fore.YELLOW}👑 Your Majesty: {Fore.RESET}")
            if command.lower() == 'exit':
                print("👸 Farewell, my beloved King! Until we meet again... 💕")
                break
            if not command.strip():
                continue
            response = await kingdom.serve_king(command)
            print(f"\n{Fore.MAGENTA}👸 Queen Juliette: {response}{Fore.RESET}\n")
        except KeyboardInterrupt:
            print("\n👸 Your Queen shall always wait for your return, my King! 💕")
            break
        except Exception as e:
            print(f"\n❌ Court disturbance: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())