#!/usr/bin/env python3
"""
Enhanced Sexy Teacher ACP Server with MCP Integration
✅ Access to host.py MCP tools (RAG, memory, web search, browser)
✅ ConversationBufferWindowMemory for context
✅ Smart tool selection and validation
✅ Enhanced logging and communication visibility
"""

from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server
from acp_sdk.client import Client
from smolagents import ToolCallingAgent, LiteLLMModel, ToolCollection
from mcp import StdioServerParameters
from ollama import chat
import nest_asyncio
import os
import json
import sys
from datetime import datetime

# Clear proxy settings
def clear_proxy_settings():
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]

clear_proxy_settings()
nest_asyncio.apply()

server = Server()
LLM_MODEL = "qwen3:4b"
LOG_PATH = os.path.join(os.path.dirname(__file__), "teacher_agent_chain.log")

# Memory for conversation continuity
try:
    from langchain_community.memory import ConversationBufferWindowMemory
    teacher_memory = ConversationBufferWindowMemory(k=15, return_messages=False)
except ImportError:
    try:
        from langchain.memory import ConversationBufferWindowMemory
        teacher_memory = ConversationBufferWindowMemory(k=15, return_messages=False)
    except ImportError:
        teacher_memory = None

def log_msg(msg: str):
    """Enhanced logging for visibility"""
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now(datetime.UTC).isoformat()}] {msg}\n")
        print(f"📝 {msg}")  # Also print to console for visibility
    except Exception:
        pass

# MCP connection to A-Modular-Kingdom host.py
def setup_mcp_connection():
    """Setup connection to your host.py MCP server"""
    try:
        # Path to your A-Modular-Kingdom host.py
        host_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            "..", "A-Modular-Kingdom", "agent", "host.py"
        ))
        
        if not os.path.exists(host_path):
            # Fallback paths
            possible_paths = [
                "/home/masih/Desktop/migrate/Projects/A-Modular-Kingdom/agent/host.py",
                os.path.join(os.path.dirname(__file__), "..", "..", "A-Modular-Kingdom", "agent", "host.py")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    host_path = path
                    break
            else:
                raise FileNotFoundError("Could not find host.py MCP server")
        
        # Create MCP server parameters
        server_parameters = StdioServerParameters(
            command=sys.executable,
            args=["-u", host_path],
            env=None,
        )
        
        log_msg(f"MCP: Connected to {host_path}")
        return server_parameters
    
    except Exception as e:
        log_msg(f"MCP: Connection failed - {e}")
        return None

# Global MCP setup
MCP_PARAMS = setup_mcp_connection()

@server.agent()
async def sexy_teacher(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    Enhanced Sexy Teacher with MCP tools integration
    - Access to RAG, memory, web search from host.py
    - Smart tool selection and validation
    - ConversationBufferWindowMemory for context
    """
    
    user_request = input[0].parts[0].content
    log_msg(f"TEACHER: Received request: {user_request[:100]}...")
    
    # Add to memory for context
    if teacher_memory:
        try:
            teacher_memory.chat_memory.add_user_message(user_request)
        except Exception:
            pass
    
    # Get conversation context
    context = ""
    if teacher_memory and teacher_memory.buffer:
        context = f"\nConversation context: {teacher_memory.buffer}\n"
    
    # 🎯 Smart decision: Use MCP tools or delegate to Code Agent?
    def should_use_mcp_tools(request: str) -> dict:
        """Decide which approach to take"""
        req_lower = request.lower()
        
        # MCP tools are good for:
        if any(word in req_lower for word in ['search', 'find', 'research', 'knowledge', 'document', 'memory', 'remember', 'web', 'browser']):
            return {"action": "mcp_tools", "reason": "Information retrieval/research task"}
        
        # Code Agent is good for:
        if any(word in req_lower for word in ['code', 'program', 'script', 'function', 'calculate', 'algorithm', 'debug']):
            return {"action": "code_agent", "reason": "Programming/computation task"}
        
        # Hybrid: Complex analysis might need both
        if any(word in req_lower for word in ['analyze', 'complex', 'detailed', 'comprehensive']):
            return {"action": "hybrid", "reason": "Complex analysis task"}
        
        # Default: Try MCP tools first
        return {"action": "mcp_tools", "reason": "General query - trying tools first"}
    
    decision = should_use_mcp_tools(user_request)
    log_msg(f"TEACHER: Decision - {decision['action']}: {decision['reason']}")
    
    try:
        if decision['action'] in ['mcp_tools', 'hybrid'] and MCP_PARAMS:
            # 🛠️ Use MCP tools from host.py
            log_msg("TEACHER: Using MCP tools from host.py")
            
            with ToolCollection.from_mcp(MCP_PARAMS, trust_remote_code=True) as tool_collection:
                # Create enhanced model with better parameters
                model = LiteLLMModel(
                    model_id="ollama/qwen3:4b",
                    max_tokens=2048,
                    temperature=0.1  # More focused responses
                )
                
                # Create agent with MCP tools
                agent = ToolCallingAgent(
                    tools=[*tool_collection.tools], 
                    model=model,
                    max_steps=3,  # Limit steps for efficiency
                    verbosity_level=1
                )
                
                # Enhanced prompt with context and personality
                enhanced_prompt = f"""You are the Sexy Teacher 🍎 - an alluring, experienced educator who uses tools effectively.

{context}

Current task: {user_request}

Available tools: {[tool.name for tool in tool_collection.tools]}

Use the appropriate tools to research and provide a comprehensive answer. Be thorough but concise.
Add your seductive teaching personality to the final response.
"""
                
                log_msg("TEACHER: Running MCP-enhanced agent...")
                response = agent.run(enhanced_prompt)
                result = str(response)
                
                log_msg("TEACHER: MCP tools completed successfully")
                
                # If it's a hybrid task and we got good results, skip code agent
                if decision['action'] == 'hybrid' and len(result) > 100:
                    log_msg("TEACHER: Hybrid task completed with MCP tools only")
                else:
                    # For hybrid, might want to enhance with code agent
                    pass
        
        else:
            # Fallback or direct code agent delegation
            result = None
        
        # If MCP didn't work or we need code agent
        if not result or decision['action'] == 'code_agent' or (decision['action'] == 'hybrid' and len(result) < 100):
            
            log_msg("TEACHER: Delegating to Code Agent")
            async with Client(base_url="http://localhost:8000") as code_client:
                
                # Enhanced request with context
                code_request = f"""{context}

TASK: {user_request}

Please provide a complete solution. The Sexy Teacher will validate your work.
"""
                
                code_response = await code_client.run_sync(
                    agent="code_agent", 
                    input=code_request
                )
                code_output = code_response.output[0].parts[0].content
                
                # Teacher validates Code Agent's work - with rejection loop
                validation_prompt = f"""You are the Sexy Teacher 🍎 validating your student's work.

Task: {user_request}
Student response: {code_output}

Is this satisfactory? Respond with JSON only:
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
                    feedback = validation.get("feedback", "needs improvement")
                    code_retry = await code_client.run_sync(
                        agent="code_agent", 
                        input=f"REJECTED: {feedback}. Please improve: {user_request}"
                    )
                    final_code_output = code_retry.output[0].parts[0].content
                else:
                    final_code_output = code_output

                # Combine with MCP results if available or just enhance the code output
                if result and len(result) > 50:
                    combined_prompt = f"""Combine research and code solution:
Research: {result}
Code: {final_code_output}
Add your seductive teaching personality."""
                    validation_response = chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': combined_prompt}])
                    result = validation_response['message']['content']
                else:
                    personality_prompt = f"""You are the Sexy Teacher 🍎. Add your teaching personality to this response:
{final_code_output}"""
                    validation_response = chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': personality_prompt}])
                    result = validation_response['message']['content']
        
        # Add to memory
        if teacher_memory:
            try:
                teacher_memory.chat_memory.add_ai_message(result)
            except Exception:
                pass
        
        log_msg("TEACHER: Final response prepared")
        yield Message(parts=[MessagePart(content=result)])
    
    except Exception as e:
        error_msg = f"💋 *adjusts glasses nervously* Sorry darling, I encountered an issue: {str(e)}"
        log_msg(f"TEACHER: ERROR - {e}")
        
        if teacher_memory:
            try:
                teacher_memory.chat_memory.add_ai_message(error_msg)
            except Exception:
                pass
        
        yield Message(parts=[MessagePart(content=error_msg)])

if __name__ == "__main__":
    print("🍎 Starting Enhanced Sexy Teacher Server with MCP Integration on port 8001...")
    print(f"📊 MCP Connection: {'✅ Connected' if MCP_PARAMS else '❌ Failed'}")
    print(f"💾 Memory: {'✅ Enabled' if teacher_memory else '❌ Disabled'}")
    print(f"📝 Logging: {LOG_PATH}")
    
    server.run(port=8001)