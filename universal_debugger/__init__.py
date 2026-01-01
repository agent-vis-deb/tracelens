"""
TraceLens - Framework Agnostic AI Agent Tracing & Visualization

Works with ANY framework:
- LangChain
- AutoGPT
- CrewAI
- LlamaIndex
- Custom agents
- Direct API calls (OpenAI, Anthropic, Google, etc.)

Usage (1 line integration):
    from universal_debugger import trace_agent
    
    with trace_agent("my-agent", api_key="..."):
        # ANY code here - automatically tracked!
        result = my_agent.run(query)
        # Works with LangChain, AutoGPT, custom code, anything!
"""

# CRITICAL: Early httpx patching - store originals BEFORE any SDKs create clients
# The actual wrapper will be installed when HTTPInterceptor.install() is called
# This just ensures we can access the original methods later
def _early_httpx_patch():
    """Early patch httpx to ensure we catch all HTTP calls"""
    try:
        import httpx
        # Store original if not already stored
        if not hasattr(httpx.Client, '_universal_debugger_original_request'):
            httpx.Client._universal_debugger_original_request = httpx.Client.request
        if not hasattr(httpx.AsyncClient, '_universal_debugger_original_request'):
            httpx.AsyncClient._universal_debugger_original_request = httpx.AsyncClient.request
    except ImportError:
        pass  # httpx not available yet

# Try to patch early, but it's OK if it fails (will patch later when interceptor installs)
try:
    _early_httpx_patch()
except:
    pass

from .core import TraceBuilder
from .client import (
    DebuggerClient,
    trace_agent,
    trace_tool,
    get_trace,
)

__all__ = [
    "DebuggerClient",
    "TraceBuilder",
    "trace_agent",
    "trace_tool",
    "get_trace",
]

__version__ = "0.1.0"

