#!/usr/bin/env python3
"""
Example usage of universal_debugger package
"""

from universal_debugger import DebuggerClient, TraceBuilder, trace_agent, trace_tool

# Example 1: Using DebuggerClient directly
def example_direct_client():
    """Example: Using DebuggerClient directly"""
    # Reads API key from DEBUGGER_API_KEY env var, uses default base_url
    client = DebuggerClient()
    
    # Or pass API key directly
    # client = DebuggerClient(api_key="your-api-key")
    
    # Create trace
    trace = TraceBuilder("example-agent")
    
    # Add steps
    trace.add_llm_step(
        name="Query LLM",
        prompt="What is Python?",
        response="Python is a programming language...",
        model="gpt-4",
        provider="openai",
        tokens=50
    )
    
    # Send trace
    client.send_trace(trace, status="success")


# Example 2: Using trace_agent context manager
def example_context_manager():
    """Example: Using trace_agent context manager"""
    # Reads API key from DEBUGGER_API_KEY env var
    with trace_agent("my-agent") as trace:
        # Any code here is automatically tracked via HTTP interception
        # Works with LangChain, AutoGPT, custom agents, anything!
        result = some_function_that_calls_llm()
        
        # You can also manually add steps
        trace.add_tool_step(
            name="Custom Processing",
            tool_name="process",
            input_args={"data": "example"},
            output_result="processed"
        )


# Example 3: Using trace_tool decorator
@trace_tool("search")
def search_web(query: str):
    """Example tool that gets tracked automatically"""
    return f"Results for: {query}"


def example_with_tool():
    """Example: Using trace_tool decorator"""
    with trace_agent("agent-with-tools"):
        result = search_web("Python tutorials")
        # Tool automatically tracked! ✅

