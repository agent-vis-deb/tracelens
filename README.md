# TraceLens Python SDK

> **Trace, observe, and visualize AI agent runs** — See inside your LLM workflows with an interactive timeline graph.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

The official Python SDK for [TraceLens](https://www.tracelensapp.com) - a framework-agnostic observability tool for AI agents. Works with **any** framework or custom code.

## ✨ Features

- 🎯 **1-Line Integration** - Just wrap your code with `trace_agent()`
- 🔄 **Automatic Tracking** - Automatically detects LLM calls, tool executions, and errors
- 🌐 **Universal Framework Support** - Works with LangChain, AutoGPT, CrewAI, LlamaIndex, custom agents, and direct API calls
- 📊 **Visual Debugging** - View traces as interactive graphs in the TraceLens dashboard
- 🚀 **Zero Code Changes** - No manual instrumentation needed for most use cases
- 🔒 **Secure** - API key authentication, encrypted data transmission

## 📦 Installation

### From PyPI (Coming Soon)

```bash
pip install tracelens
```

### From Source

```bash
git clone https://github.com/your-username/tracelens-python-sdk.git
cd tracelens-python-sdk
pip install -e .
```

## 🚀 Quick Start

### 1. Get Your API Key

1. Sign up at [TraceLens](https://www.tracelensapp.com)
2. Go to Settings → Ingest API Keys
3. Generate a new API key
4. Copy it (you won't see it again!)

### 2. Set Environment Variables

```bash
export DEBUGGER_API_KEY="your-api-key-here"
export DEBUGGER_URL="https://www.tracelensapp.com"  # Optional, defaults to this
```

Or create a `.env` file:

```bash
DEBUGGER_API_KEY=your-api-key-here
DEBUGGER_URL=https://www.tracelensapp.com
```

### 3. Use in Your Code

**Automatic tracking (recommended):**

```python
from universal_debugger import trace_agent
from openai import OpenAI
import os

# Set API key (or use environment variable)
os.environ["DEBUGGER_API_KEY"] = "your-api-key-here"

# Wrap your agent code - everything is automatically tracked!
with trace_agent("my-agent"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2+2?"}]
    )
    
    print(response.choices[0].message.content)

# Trace is automatically sent when the 'with' block ends!
```

**Manual tracking (more control):**

```python
from universal_debugger import DebuggerClient, TraceBuilder

client = DebuggerClient(api_key="your-api-key-here")
trace = TraceBuilder("my-agent")

# Add steps manually
trace.add_http_event(
    method="POST",
    url="https://api.openai.com/v1/chat/completions",
    request_body={"model": "gpt-4o-mini", "messages": [...]},
    response_status=200,
    response_body={"choices": [...]},
    started_at=start_time,
    ended_at=end_time
)

# Send trace
client.send_trace(trace, status="success")
```

## 📖 Documentation

### Basic Usage

```python
from universal_debugger import trace_agent, trace_tool

# Automatic tracking
with trace_agent("my-agent"):
    # All LLM calls, tool calls, and errors are automatically tracked
    result = my_agent.run(query)
```

### Tool Decorator

```python
from universal_debugger import trace_tool

@trace_tool("search_web")
def search_web(query: str):
    """Search the web"""
    return f"Results for: {query}"

with trace_agent("my-agent"):
    results = search_web("Python tutorials")
    # Tool automatically tracked! ✅
```

### Direct Client Usage

```python
from universal_debugger import DebuggerClient, TraceBuilder

client = DebuggerClient(
    api_key="your-api-key",
    base_url="https://www.tracelensapp.com"  # Optional
)

trace = TraceBuilder("my-agent", environment="production")

# Add HTTP events
trace.add_http_event(
    method="POST",
    url="https://api.openai.com/v1/chat/completions",
    request_body={...},
    response_status=200,
    response_body={...},
    started_at=start_time,
    ended_at=end_time
)

# Add function events
trace.add_function_event(
    function_name="process_data",
    args={"input": "data"},
    result="processed",
    started_at=start_time,
    ended_at=end_time
)

# Add logs
trace.add_log("info", "Processing started")
trace.add_log("error", "Something went wrong", metadata={"code": 500})

# Send trace
trace_id = client.send_trace(trace, status="success")
```

## 🔧 Configuration

### Environment Variables

- `DEBUGGER_API_KEY` (required) - Your TraceLens API key
- `DEBUGGER_URL` (optional) - Base URL of TraceLens service (defaults to `https://www.tracelensapp.com`)
- `DEBUGGER_ENV` (optional) - Environment name (defaults to `production`)
- `DEBUGGER_DEBUG` (optional) - Enable debug logging (`1` to enable)
- `DEBUGGER_DISABLE_FUNCTION_TRACING` (optional) - Disable automatic function tracing (`1` to disable)
- `DEBUGGER_DISABLE_LOG_CAPTURE` (optional) - Disable automatic log capture (`1` to disable)

## 🎯 Supported Frameworks

The SDK automatically tracks:

- ✅ **OpenAI SDK** - All `chat.completions.create()` calls
- ✅ **Anthropic SDK** - Claude API calls (via HTTP interception)
- ✅ **Google AI SDK** - Gemini API calls (via HTTP interception)
- ✅ **LangChain** - Agent executions and tool calls
- ✅ **Custom HTTP clients** - Any `requests` or `httpx` calls
- ✅ **Function calls** - Automatic function call tracking
- ✅ **Logs** - Automatic `print()` and `logging` capture

## 📚 Examples

See [`example_usage.py`](example_usage.py) for complete examples.

### Example: LangChain Agent

```python
from universal_debugger import trace_agent
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI

with trace_agent("langchain-agent"):
    llm = ChatOpenAI(model="gpt-4o-mini")
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    
    result = executor.invoke({"input": "What is the weather?"})
    # All LLM calls and tool executions are automatically tracked!
```

### Example: Custom Agent with Tools

```python
from universal_debugger import trace_agent, trace_tool

@trace_tool("search")
def search_knowledge_base(query: str):
    return f"Results for: {query}"

@trace_tool("calculate")
def calculate(expression: str):
    return eval(expression)

with trace_agent("custom-agent"):
    # Your agent code here
    results = search_knowledge_base("Python")
    answer = calculate("2 + 2")
    # All tool calls are automatically tracked!
```

## 🔍 Viewing Traces

After sending a trace:

1. Go to [TraceLens Dashboard](https://www.tracelensapp.com/app)
2. Find your trace by app name (e.g., "my-agent")
3. Click to view the interactive graph
4. Explore steps, timing, tokens, and logs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Website**: [https://www.tracelensapp.com](https://www.tracelensapp.com)
- **Documentation**: [https://www.tracelensapp.com/docs](https://www.tracelensapp.com/docs)
- **Support**: support@tracelensapp.com

## 🙏 Acknowledgments

Built for developers who want to understand what their AI agents are doing.

