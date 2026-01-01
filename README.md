# TraceLens Python SDK

**Visual debugging and observability for AI agents**

TraceLens automatically captures and visualizes your AI agent runs—LLM calls, tool invocations, latency, tokens, and errors—as an interactive timeline graph. Works with any framework: LangChain, AutoGPT, CrewAI, LlamaIndex, or custom agents.

## Installation

Early access via GitHub:

```bash
pip install git+https://github.com/<your-username>/tracelens.git
```

## Quickstart

```python
from tracelens import trace_agent

# Set your API key (get it from https://www.tracelensapp.com/settings)
import os
os.environ["DEBUGGER_API_KEY"] = "your-api-key-here"

# Wrap your agent code - that's it!
with trace_agent("my-agent"):
    result = my_agent.run(query)
    # Automatically tracked! ✅
```

View your traces at: **https://www.tracelensapp.com/app**

## How It Works

**Thin SDK, Smart Backend**

The TraceLens SDK is intentionally lightweight—it collects raw events (HTTP requests, function calls, logs) and sends them to the TraceLens backend. All intelligence—LLM detection, provider classification, token counting, graph visualization—happens server-side.

This means:
- ✅ **Minimal overhead** - Just event collection, no heavy processing
- ✅ **Always up-to-date** - New features appear automatically
- ✅ **Framework agnostic** - Works with any Python agent code

### What Gets Tracked

- **LLM calls** - Automatic interception of OpenAI, Anthropic, Google, and other providers
- **Tool invocations** - Function calls decorated with `@trace_tool`
- **HTTP requests** - All HTTP traffic via `requests` and `httpx`
- **Logs** - Captured print statements and logging output
- **Errors** - Exceptions and failures

## Advanced Usage

### Manual Event Tracking

```python
from tracelens import TraceLensClient, TraceBuilder

client = TraceLensClient(api_key="your-api-key")
trace = TraceBuilder("my-agent")

# Add HTTP event
trace.add_http_event(
    method="POST",
    url="https://api.openai.com/v1/chat/completions",
    request_body={"model": "gpt-4", "messages": [...]},
    response_status=200,
    response_body={"choices": [...]},
)

# Send trace
client.send_trace(trace, status="success")
```

### Tool Decorator

```python
from tracelens import trace_agent, trace_tool

@trace_tool("search_web")
def search_web(query: str):
    return f"Results for: {query}"

with trace_agent("my-agent"):
    result = search_web("Python tutorials")
    # Tool automatically tracked! ✅
```

## Configuration

### Environment Variables

- `DEBUGGER_API_KEY` - Your API key (required)
- `DEBUGGER_BASE_URL` - Backend URL (defaults to `https://www.tracelensapp.com`)
- `DEBUGGER_DISABLE_FUNCTION_TRACING` - Set to `"1"` to disable automatic function tracing
- `DEBUGGER_DISABLE_LOG_CAPTURE` - Set to `"1"` to disable log capture

### Get Your API Key

1. Sign up at [https://www.tracelensapp.com](https://www.tracelensapp.com)
2. Go to Settings → API Keys
3. Generate a new ingest API key
4. Copy it immediately (you won't see it again!)

## Documentation

- **Web UI**: [https://www.tracelensapp.com/app](https://www.tracelensapp.com/app)
- **Documentation**: [https://www.tracelensapp.com/docs](https://www.tracelensapp.com/docs)

## License

MIT
