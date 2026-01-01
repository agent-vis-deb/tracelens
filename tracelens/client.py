"""
Client for TraceLens - THIN SDK VERSION
Just collects raw events, all processing happens on backend
"""

import os
import sys
import json
import requests
import functools
import inspect
import builtins
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable
from contextlib import contextmanager
from threading import local

from .core import TraceBuilder

# Thread-local storage for trace context
_thread_local = local()

# Debug flag
_DEBUG_MODE = os.getenv("DEBUGGER_DEBUG", "0") == "1"

def _debug_print(*args, **kwargs):
    """Helper to conditionally print debug messages"""
    if _DEBUG_MODE:
        print(*args, **kwargs)

_OPENAI_SDK_PATCHED = False

def _install_openai_sdk_adapter() -> None:
    """
    Lightweight OpenAI SDK adapter.

    - Patches OpenAI SDK resource `.create()` methods at the CLASS level (no per-instance patching).
    - Records events ONLY when inside `trace_agent()` (i.e., when `get_trace()` is non-None).
    - Ensures streaming requests include usage tokens by injecting `stream_options={"include_usage": True}`
      (no extra network calls).
    """
    global _OPENAI_SDK_PATCHED
    if _OPENAI_SDK_PATCHED:
        return

    try:
        import openai  # noqa: F401
    except Exception:
        return

    # Helpers ---------------------------------------------------------------
    def _safe_messages_preview(messages: Any, max_items: int = 25, max_chars: int = 8000) -> Any:
        """Keep prompts lightweight: include only a limited number of messages and truncate long content."""
        try:
            if not isinstance(messages, list):
                return messages
            out = []
            for m in messages[:max_items]:
                if not isinstance(m, dict):
                    out.append(m)
                    continue
                role = m.get("role")
                content = m.get("content")
                if isinstance(content, str) and len(content) > max_chars:
                    content = content[:max_chars] + "..."
                out.append({"role": role, "content": content})
            return out
        except Exception:
            return messages

    def _extract_usage_dict(resp_obj: Any) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(resp_obj, "usage") and resp_obj.usage is not None:
                usage = resp_obj.usage
                if isinstance(usage, dict):
                    total = usage.get("total_tokens") or usage.get("totalTokens")
                    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("inputTokens")
                    completion = usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("outputTokens")
                else:
                    total = getattr(usage, "total_tokens", None)
                    prompt = getattr(usage, "prompt_tokens", None)
                    completion = getattr(usage, "completion_tokens", None)
                return {
                    "total_tokens": total,
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    # Compatibility aliases
                    "totalTokens": total,
                    "input_tokens": prompt,
                    "output_tokens": completion,
                }
        except Exception:
            return None
        return None

    def _infer_base_url(resource_self: Any) -> str:
        # Best-effort: OpenAI resources often have `_client.base_url`
        try:
            client = getattr(resource_self, "_client", None) or getattr(resource_self, "client", None)
            base_url = getattr(client, "base_url", None) if client is not None else None
            if base_url is None:
                return "https://api.openai.com"
            return str(base_url).rstrip("/")
        except Exception:
            return "https://api.openai.com"

    def _wrap_create(original_create: Callable[..., Any], path: str) -> Callable[..., Any]:
        @functools.wraps(original_create)
        def wrapped(resource_self, *args, **kwargs):
            # Only capture when tracing is active
            trace = get_trace()
            if trace is None:
                return original_create(resource_self, *args, **kwargs)

            started_at = datetime.now(timezone.utc)

            # Inject streaming usage tokens (no extra calls)
            try:
                if kwargs.get("stream") is True and "stream_options" not in kwargs:
                    kwargs["stream_options"] = {"include_usage": True}
            except Exception:
                pass

            # Build minimal request body (avoid heavy tool schemas / large payloads)
            model = kwargs.get("model", "unknown")
            messages = kwargs.get("messages")
            request_body: Dict[str, Any] = {"model": model}
            if messages is not None:
                request_body["messages"] = _safe_messages_preview(messages)
            if "prompt" in kwargs:
                request_body["prompt"] = kwargs.get("prompt")
            if "input" in kwargs:
                request_body["input"] = kwargs.get("input")
            if kwargs.get("stream") is True:
                request_body["stream"] = True
                if "stream_options" in kwargs:
                    request_body["stream_options"] = kwargs.get("stream_options")

            try:
                response = original_create(resource_self, *args, **kwargs)
            except Exception as e:
                ended_at = datetime.now(timezone.utc)
                trace.add_http_event(
                    method="POST",
                    url=_infer_base_url(resource_self) + path,
                    request_headers={"Content-Type": "application/json"},
                    request_body=request_body,
                    response_status=None,
                    response_headers={},
                    response_body={"error": str(e)[:1000]},
                    started_at=started_at,
                    ended_at=ended_at,
                )
                raise

            # Streaming-like: iterable and no `choices` attribute (OpenAI Stream)
            is_streaming = hasattr(response, "__iter__") and not hasattr(response, "choices")

            if _DEBUG_MODE:
                _debug_print(f"[DEBUGGER] OpenAI response type: {type(response)}")
                _debug_print(f"[DEBUGGER] OpenAI response is_streaming_like: {is_streaming}")

            if is_streaming:
                # Wrap stream so we can capture final usage + output preview once the stream finishes
                active_stream = response
                original_iter = iter(active_stream)
                text_preview: List[str] = []
                max_preview_chars = 10000
                last_usage: Optional[Dict[str, Any]] = None

                class _StreamWrapper:
                    def __enter__(self_inner):
                        nonlocal active_stream, original_iter
                        if hasattr(response, "__enter__") and hasattr(response, "__exit__"):
                            try:
                                active_stream = response.__enter__()
                                original_iter = iter(active_stream)
                            except Exception:
                                active_stream = response
                                original_iter = iter(active_stream)
                        return self_inner

                    def __exit__(self_inner, exc_type, exc, tb):
                        if hasattr(response, "__exit__"):
                            try:
                                return response.__exit__(exc_type, exc, tb)
                            except Exception:
                                return False
                        return False

                    def __iter__(self_inner):
                        nonlocal last_usage, text_preview
                        try:
                            for chunk in original_iter:
                                try:
                                    u = _extract_usage_dict(chunk)
                                    if u:
                                        last_usage = u
                                    # Capture delta preview if present
                                    if hasattr(chunk, "choices") and chunk.choices:
                                        c0 = chunk.choices[0]
                                        delta = getattr(c0, "delta", None)
                                        delta_content = getattr(delta, "content", None) if delta is not None else None
                                        if delta_content:
                                            current = "".join(text_preview)
                                            if len(current) < max_preview_chars:
                                                text_preview.append(str(delta_content))
                                except Exception:
                                    pass
                                yield chunk
                        finally:
                            ended_at = datetime.now(timezone.utc)
                            response_body: Dict[str, Any] = {
                                "model": getattr(response, "model", None),
                                "choices": [{"message": {"content": ("".join(text_preview))[:max_preview_chars]}}] if text_preview else [],
                            }
                            if last_usage:
                                response_body["usage"] = last_usage
                            trace.add_http_event(
                                method="POST",
                                url=_infer_base_url(resource_self) + path,
                                request_headers={"Content-Type": "application/json"},
                                request_body=request_body,
                                response_status=200,
                                response_headers={"Content-Type": "application/json"},
                                response_body=response_body,
                                started_at=started_at,
                                ended_at=ended_at,
                            )
                            if _DEBUG_MODE:
                                _debug_print(f"[DEBUGGER] OpenAI SDK stream captured: {model}")
                                _debug_print(f"[DEBUGGER] Stream usage: {last_usage}")

                    def __getattr__(self_inner, name):
                        return getattr(response, name)

                return _StreamWrapper()

            # Non-streaming: capture immediately
            ended_at = datetime.now(timezone.utc)
            usage_data = _extract_usage_dict(response)

            content = None
            try:
                if hasattr(response, "choices") and response.choices:
                    choice0 = response.choices[0]
                    msg = getattr(choice0, "message", None)
                    content = getattr(msg, "content", None) if msg is not None else None
            except Exception:
                content = None

            response_body2: Dict[str, Any] = {
                "model": getattr(response, "model", None),
                "choices": [{"message": {"content": content}}] if content is not None else [],
            }
            if usage_data:
                response_body2["usage"] = usage_data

            trace.add_http_event(
                method="POST",
                url=_infer_base_url(resource_self) + path,
                request_headers={"Content-Type": "application/json"},
                request_body=request_body,
                response_status=200,
                response_headers={"Content-Type": "application/json"},
                response_body=response_body2,
                started_at=started_at,
                ended_at=ended_at,
            )
            if _DEBUG_MODE:
                _debug_print(f"[DEBUGGER] OpenAI SDK call captured: {model}")
                _debug_print(f"[DEBUGGER] Usage: {usage_data}")
            return response

        return wrapped

    # Patch targets ----------------------------------------------------------
    patched_any = False
    candidates: List[tuple] = []
    try:
        from openai.resources.chat.completions import Completions as ChatCompletions  # type: ignore
        candidates.append((ChatCompletions, "/v1/chat/completions"))
    except Exception:
        pass
    try:
        from openai.resources.completions import Completions as LegacyCompletions  # type: ignore
        candidates.append((LegacyCompletions, "/v1/completions"))
    except Exception:
        pass
    try:
        from openai.resources.responses import Responses  # type: ignore
        candidates.append((Responses, "/v1/responses"))
    except Exception:
        pass

    for cls, path in candidates:
        try:
            if hasattr(cls, "create") and not hasattr(getattr(cls, "create"), "_tracelens_patched"):
                original = getattr(cls, "create")
                wrapped = _wrap_create(original, path)
                wrapped._tracelens_patched = True  # type: ignore
                setattr(cls, "create", wrapped)
                patched_any = True
        except Exception:
            continue

    _OPENAI_SDK_PATCHED = True
    if _DEBUG_MODE and patched_any:
        _debug_print("[DEBUGGER] OpenAI SDK adapter installed (class-level patch)")


class HTTPInterceptor:
    """
    Thin HTTP interceptor - just captures raw HTTP request/response pairs.
    No classification, no provider detection, no parsing - backend handles all that.
    """
    
    def __init__(self, trace_builder: TraceBuilder):
        self.trace = trace_builder
        self._original_request = None
        self._original_httpx_request = None
        self._original_httpx_async_request = None
        self._active = False
    
    def install(self):
        """Install HTTP interceptor for requests and httpx - captures ALL HTTP calls"""
        if self._active:
            return
        
        # Intercept requests library
        try:
            import requests as requests_module
            
            if not hasattr(requests_module.Session, '_original_request'):
                self._original_request = requests_module.Session.request
                
                @functools.wraps(self._original_request)
                def wrapped_request(self_session, method, url, **kwargs):
                    started_at = datetime.now(timezone.utc)
                    
                    # Get request data
                    request_headers = dict(kwargs.get('headers', {}))
                    request_body = kwargs.get('data') or kwargs.get('json')
                    
                    # Call original
                    try:
                        response = self._original_request(self_session, method, url, **kwargs)
                        ended_at = datetime.now(timezone.utc)
                        
                        # Capture raw HTTP event
                        self.trace.add_http_event(
                            method=method,
                            url=str(url),
                            request_headers=request_headers,
                            request_body=request_body,
                            response_status=response.status_code,
                            response_headers=dict(response.headers),
                            response_body=response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text[:10000],
                            started_at=started_at,
                            ended_at=ended_at,
                        )
                        
                        return response
                    except Exception as e:
                        ended_at = datetime.now(timezone.utc)
                        # Still capture event even if request failed
                        self.trace.add_http_event(
                            method=method,
                            url=str(url),
                            request_headers=request_headers,
                            request_body=request_body,
                            response_status=None,
                            response_headers={},
                            response_body={"error": str(e)[:1000]},
                            started_at=started_at,
                            ended_at=ended_at,
                        )
                        raise
                
                requests_module.Session.request = wrapped_request
                requests_module.Session.request._original = self._original_request
                _debug_print("[DEBUGGER] Intercepted requests library")
        except ImportError:
            _debug_print("[DEBUGGER] requests library not found, skipping interception")
        
        # Intercept httpx.Client (sync)
        # CRITICAL: Patch both the class method AND ensure new instances use it
        try:
            import httpx
            
            # Check if already patched (either by us or by early patch)
            if hasattr(httpx.Client, '_tracelens_original_request'):
                self._original_httpx_request = httpx.Client._tracelens_original_request
            elif not hasattr(httpx.Client, '_original_request'):
                self._original_httpx_request = httpx.Client.request
                httpx.Client._tracelens_original_request = httpx.Client.request
            else:
                self._original_httpx_request = httpx.Client._original_request
            
            # Only patch if not already patched
            if not hasattr(httpx.Client.request, '_tracelens_patched'):
                
                @functools.wraps(self._original_httpx_request)
                def wrapped_httpx_request(self_client, method, url, **kwargs):
                    started_at = datetime.now(timezone.utc)
                    
                    url_str = str(url)
                    request_headers = dict(kwargs.get('headers', {}))
                    request_body = kwargs.get('json') or kwargs.get('content')
                    
                    # Debug: Log all POST requests to help diagnose
                    if _DEBUG_MODE and method.upper() == 'POST':
                        _debug_print(f"[DEBUGGER] httpx.Client POST: {url_str[:100]}")
                        if request_body and isinstance(request_body, dict):
                            _debug_print(f"[DEBUGGER] httpx.Client request_body keys: {list(request_body.keys())[:5]}")
                    
                    try:
                        response = self._original_httpx_request(self_client, method, url, **kwargs)
                        ended_at = datetime.now(timezone.utc)
                        
                        # Capture raw HTTP event
                        try:
                            if response.headers.get('content-type', '').startswith('application/json'):
                                response_body = response.json()
                            else:
                                response_body = response.text[:10000]
                        except:
                            response_body = response.text[:10000] if hasattr(response, 'text') else None
                        
                        self.trace.add_http_event(
                            method=method,
                            url=url_str,
                            request_headers=request_headers,
                            request_body=request_body,
                            response_status=response.status_code,
                            response_headers=dict(response.headers),
                            response_body=response_body,
                            started_at=started_at,
                            ended_at=ended_at,
                        )
                        
                        return response
                    except Exception as e:
                        ended_at = datetime.now(timezone.utc)
                        self.trace.add_http_event(
                            method=method,
                            url=url_str,
                            request_headers=request_headers,
                            request_body=request_body,
                            response_status=None,
                            response_headers={},
                            response_body={"error": str(e)[:1000]},
                            started_at=started_at,
                            ended_at=ended_at,
                        )
                        raise
                
                # Patch the class method - this affects all existing and future instances
                httpx.Client.request = wrapped_httpx_request
                httpx.Client.request._original = self._original_httpx_request
                httpx.Client.request._tracelens_patched = True
                
                # Also try to patch at the module level if httpx uses a different structure
                # Some libraries create clients via httpx._client or httpx._transports
                try:
                    if hasattr(httpx, '_client'):
                        import httpx._client as httpx_client
                        if hasattr(httpx_client, 'Client') and hasattr(httpx_client.Client, 'request'):
                            if not hasattr(httpx_client.Client.request, '_original'):
                                httpx_client.Client.request = wrapped_httpx_request
                                _debug_print("[DEBUGGER] Also patched httpx._client.Client")
                except:
                    pass  # Not critical if this fails
                
                _debug_print("[DEBUGGER] Intercepted httpx.Client")
            
            # Intercept httpx.AsyncClient (async)
            # Check if already patched
            if hasattr(httpx.AsyncClient, '_tracelens_original_request'):
                self._original_httpx_async_request = httpx.AsyncClient._tracelens_original_request
            elif not hasattr(httpx.AsyncClient, '_original_request'):
                self._original_httpx_async_request = httpx.AsyncClient.request
                httpx.AsyncClient._tracelens_original_request = httpx.AsyncClient.request
            else:
                self._original_httpx_async_request = httpx.AsyncClient._original_request
            
            # Only patch if not already patched
            if not hasattr(httpx.AsyncClient.request, '_tracelens_patched'):
                
                @functools.wraps(self._original_httpx_async_request)
                async def wrapped_httpx_async_request(self_client, method, url, **kwargs):
                    started_at = datetime.now(timezone.utc)
                    
                    url_str = str(url)
                    request_headers = dict(kwargs.get('headers', {}))
                    request_body = kwargs.get('json') or kwargs.get('content')
                    
                    # Debug: Log all POST requests to help diagnose
                    if _DEBUG_MODE and method.upper() == 'POST':
                        _debug_print(f"[DEBUGGER] httpx.AsyncClient POST: {url_str[:100]}")
                        if request_body and isinstance(request_body, dict):
                            _debug_print(f"[DEBUGGER] httpx.AsyncClient request_body keys: {list(request_body.keys())[:5]}")
                    
                    try:
                        response = await self._original_httpx_async_request(self_client, method, url, **kwargs)
                        ended_at = datetime.now(timezone.utc)
                        
                        # Capture raw HTTP event
                        try:
                            if response.headers.get('content-type', '').startswith('application/json'):
                                response_body = await response.json() if hasattr(response, 'json') and inspect.iscoroutinefunction(response.json) else response.json()
                            else:
                                response_text = await response.aread() if hasattr(response, 'aread') else response.text
                                response_body = (response_text.decode() if isinstance(response_text, bytes) else str(response_text))[:10000]
                        except:
                            try:
                                response_text = await response.aread() if hasattr(response, 'aread') else response.text
                                response_body = (response_text.decode() if isinstance(response_text, bytes) else str(response_text))[:10000]
                            except:
                                response_body = None
                        
                        self.trace.add_http_event(
                            method=method,
                            url=url_str,
                            request_headers=request_headers,
                            request_body=request_body,
                            response_status=response.status_code,
                            response_headers=dict(response.headers),
                            response_body=response_body,
                            started_at=started_at,
                            ended_at=ended_at,
                        )
                        
                        return response
                    except Exception as e:
                        ended_at = datetime.now(timezone.utc)
                        self.trace.add_http_event(
                            method=method,
                            url=url_str,
                            request_headers=request_headers,
                            request_body=request_body,
                            response_status=None,
                            response_headers={},
                            response_body={"error": str(e)[:1000]},
                            started_at=started_at,
                            ended_at=ended_at,
                        )
                        raise
                
                httpx.AsyncClient.request = wrapped_httpx_async_request
                httpx.AsyncClient.request._original = self._original_httpx_async_request
                httpx.AsyncClient.request._tracelens_patched = True
                _debug_print("[DEBUGGER] Intercepted httpx.AsyncClient")
                
        except ImportError:
            _debug_print("[DEBUGGER] httpx library not found, skipping interception")
        
        # OpenAI SDK adapter (patched once, records only inside trace_agent)
        _install_openai_sdk_adapter()
        
        self._active = True
    
    def uninstall(self):
        """Uninstall HTTP interceptor"""
        if not self._active:
            return
        
        try:
            import requests as requests_module
            if hasattr(requests_module.Session, '_original_request'):
                requests_module.Session.request = self._original_request
                delattr(requests_module.Session, '_original_request')
        except ImportError:
            pass
        
        try:
            import httpx
            if hasattr(httpx.Client, '_original_request'):
                httpx.Client.request = self._original_httpx_request
                delattr(httpx.Client, '_original_request')
            if hasattr(httpx.AsyncClient, '_original_request'):
                httpx.AsyncClient.request = self._original_httpx_async_request
                delattr(httpx.AsyncClient, '_original_request')
        except ImportError:
            pass
        
        self._active = False


class TraceContext:
    """
    Universal trace context - automatically tracks ANY agent framework
    THIN VERSION: Just collects raw events, no processing
    """
    
    def __init__(self, app_name: str, client: 'DebuggerClient', environment: Optional[str] = None):
        self.app_name = app_name
        self.client = client
        self.environment = environment
        self.trace = TraceBuilder(app_name, environment)
        self.http_interceptor = None
        self.start_time = None
        self.original_trace = None
        self.tracked_functions = {}  # Track function call start times
        self._original_print = None
        self._logging_patched = False
        
        # Store the directory where the script was called from
        if sys.argv and len(sys.argv) > 0:
            try:
                script_path = os.path.abspath(sys.argv[0])
                self.script_dir = os.path.dirname(script_path)
            except:
                self.script_dir = os.getcwd()
        else:
            self.script_dir = os.getcwd()
    
    def __enter__(self):
        _debug_print(f"[DEBUGGER] Starting trace for app: {self.app_name}")
        
        # Clear any stale thread-local state
        for attr in ['trace', 'trace_context', 'last_llm_step_id', 'last_tool_step_id', 'inside_agent_executor']:
            if hasattr(_thread_local, attr):
                delattr(_thread_local, attr)
        
        # Store in thread-local
        _thread_local.trace = self.trace
        _thread_local.trace_context = self
        
        self.start_time = datetime.now(timezone.utc)
        
        # Install HTTP interceptor (captures all HTTP calls)
        self.http_interceptor = HTTPInterceptor(self.trace)
        self.http_interceptor.install()
        _debug_print("[DEBUGGER] HTTP interceptor installed")
        
        # Install automatic function call tracking
        self._install_function_tracer()
        
        # Install automatic log capture
        self._install_log_capture()
        
        _debug_print(f"[DEBUGGER] Trace context ready. Events captured: {len(self.trace.events)}")
        return self.trace
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now(timezone.utc)
        status = "success"
        
        if exc_type is not None:
            status = "error"
        
        # Uninstall function tracer
        if self.original_trace is not None:
            sys.settrace(self.original_trace)
            self.original_trace = None
        
        # Uninstall interceptor
        if self.http_interceptor:
            self.http_interceptor.uninstall()
        
        # Uninstall log capture
        self._uninstall_log_capture()
        
        # Clear thread-local state
        for attr in ['trace', 'trace_context', 'last_llm_step_id', 'last_tool_step_id', 'inside_agent_executor']:
            if hasattr(_thread_local, attr):
                delattr(_thread_local, attr)
        
        # Send trace
        _debug_print(f"[DEBUGGER] Sending trace with {len(self.trace.events)} events, status: {status}")
        self.client.send_trace(self.trace, status=status)
    
    def _install_function_tracer(self):
        """Install automatic function call tracking - captures raw function calls"""
        if os.getenv("DEBUGGER_DISABLE_FUNCTION_TRACING", "0") == "1":
            return
        
        def trace_function(frame, event, arg):
            if event == 'call':
                # Only track functions in user's project directory
                filename = frame.f_code.co_filename
                if not filename.startswith(self.script_dir):
                    return None
                
                # Skip debugger's own functions
                if 'tracelens' in filename or 'debugger' in filename.lower():
                    return None
                
                # Skip built-ins and stdlib
                if filename.startswith('<') or 'site-packages' in filename:
                    return None
                
                # Get function info
                func_name = frame.f_code.co_name
                module_name = frame.f_code.co_filename
                
                # Skip private/magic methods
                if func_name.startswith('_'):
                    return None
                
                # Track start time
                call_id = f"{func_name}_{id(frame)}"
                self.tracked_functions[call_id] = {
                    'started_at': datetime.now(timezone.utc),
                    'frame': frame,
                }
                
                return trace_function
            
            elif event == 'return':
                # Find matching call
                call_id = f"{frame.f_code.co_name}_{id(frame)}"
                if call_id in self.tracked_functions:
                    func_info = self.tracked_functions.pop(call_id)
                    started_at = func_info['started_at']
                    ended_at = datetime.now(timezone.utc)
                    
                    # Capture raw function event
                    self.trace.add_function_event(
                        function_name=frame.f_code.co_name,
                        module_name=frame.f_code.co_filename,
                        args=None,  # Args are complex to extract, backend can handle this
                        kwargs=None,
                        result=str(arg)[:5000] if arg is not None else None,
                        exception=None,
                        started_at=started_at,
                        ended_at=ended_at,
                    )
                
                return trace_function
            
            return None
        
        self.original_trace = sys.gettrace()
        sys.settrace(trace_function)
        _debug_print("[DEBUGGER] Function tracer installed")
    
    def _install_log_capture(self):
        """Install automatic log capture (captures print() and logging module)"""
        if os.getenv("DEBUGGER_DISABLE_LOG_CAPTURE", "0") == "1":
            return
        
        # Capture print()
        self._original_print = builtins.print
        original_print = builtins.print  # Capture for closure
        
        def wrapped_print(*args, **kwargs):
            # Call original
            original_print(*args, **kwargs)
            
            # Capture as log
            try:
                message = ' '.join(str(arg) for arg in args)
                # Filter out very long messages
                if len(message) > 500:
                    return
                # Filter common response patterns
                skip_patterns = ['---', 'Generated response:', 'Final answer:', 'Reply to customer']
                if any(pattern in message for pattern in skip_patterns):
                    return
                
                self.trace.add_log('info', message)
            except:
                pass
        
        # Monkey-patch print
        builtins.print = wrapped_print
        
        # Capture logging module
        try:
            import logging
            original_log = logging.Logger._log
            
            def wrapped_log(self_logger, level, msg, args, **kwargs):
                # Call original
                original_log(self_logger, level, msg, args, **kwargs)
                
                # Capture as log
                try:
                    message = str(msg) % args if args else str(msg)
                    if len(message) > 500:
                        return
                    
                    level_map = {
                        logging.DEBUG: 'debug',
                        logging.INFO: 'info',
                        logging.WARNING: 'warning',
                        logging.ERROR: 'error',
                    }
                    log_level = level_map.get(level, 'info')
                    self.trace.add_log(log_level, message)
                except:
                    pass
            
            logging.Logger._log = wrapped_log
            self._logging_patched = True
        except:
            pass
        
        _debug_print("[DEBUGGER] Log capture installed")
    
    def _uninstall_log_capture(self):
        """Restore original print and logging functions"""
        if self._original_print:
            builtins.print = self._original_print
        
        if self._logging_patched:
            try:
                import logging
                # Can't easily restore, but that's OK - it's a one-time patch
                pass
            except:
                pass


class DebuggerClient:
    """Client for sending traces to the debugger API"""
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = base_url or os.getenv("DEBUGGER_BASE_URL", "https://www.tracelensapp.com")
        self.api_key = api_key or os.getenv("DEBUGGER_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key required. Set DEBUGGER_API_KEY env var or pass api_key parameter")
    
    def send_trace(self, trace: TraceBuilder, status: str = "success") -> Optional[str]:
        """Send trace to the backend - sends raw events, not processed steps"""
        try:
            # Build payload with raw events
            payload = {
                "trace": {
                    "appName": trace.app_name,
                    "status": status,
                    "environment": trace.environment,
                },
            }
            
            # Always include events (even if empty) - backend validation requires either events or steps
            payload["events"] = trace.events
            
            # Only include logs if there are any
            if trace.logs:
                payload["logs"] = trace.logs
            
            # Send to backend
            response = requests.post(
                f"{self.base_url}/api/traces",
                json=payload,
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
            
            # Accept 2xx status codes as success (200 OK, 201 Created, etc.)
            if 200 <= response.status_code < 300:
                result = response.json()
                trace_id = result.get("traceId") or result.get("id")
                _debug_print(f"✅ Trace sent successfully! ({len(trace.events)} events)")
                if trace_id:
                    print(f"\n📊 Trace sent successfully! ({len(trace.events)} events)")
                    print(f"View at: {self.base_url}/app")
                    print(f"Trace ID: {trace_id}")
                return trace_id
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg += f" error: {error_data['error']}"
                    if "details" in error_data:
                        details = error_data['details']
                        if isinstance(details, list) and len(details) > 0:
                            error_msg += f"\n  Details: {details[0].get('message', '')}"
                        elif isinstance(details, str):
                            error_msg += f"\n  Details: {details}"
                except:
                    pass
                print(f"\n❌ Failed to send trace: {error_msg}")
                if response.status_code == 400:
                    _debug_print(f"[DEBUGGER] Payload had {len(trace.events)} events, {len(trace.logs)} logs")
                return None
                
        except Exception as e:
            error_msg = str(e)[:200]
            print(f"\n❌ Failed to send trace: {error_msg}")
            return None


@contextmanager
def trace_agent(app_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None, environment: Optional[str] = None):
    """
    Context manager for tracing an AI agent.
    
    Usage:
        with trace_agent("my-agent"):
            result = my_agent.run(query)
    """
    client = DebuggerClient(base_url=base_url, api_key=api_key)
    context = TraceContext(app_name, client, environment=environment)
    yield context.__enter__()
    context.__exit__(None, None, None)


def trace_tool(tool_name: Optional[str] = None):
    """
    Decorator to mark a function as a tool (will be tracked as function event).
    
    Usage:
        @trace_tool("my_tool")
        def my_tool(x: int) -> int:
            return x * 2
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            trace = getattr(_thread_local, 'trace', None)
            if not trace:
                return func(*args, **kwargs)
            
            started_at = datetime.now(timezone.utc)
            try:
                result = func(*args, **kwargs)
                ended_at = datetime.now(timezone.utc)
                
                # Capture as function event
                trace.add_function_event(
                    function_name=tool_name or func.__name__,
                    module_name=func.__module__ or None,
                    args={"args": [str(arg)[:1000] for arg in args]},
                    kwargs={k: str(v)[:1000] for k, v in kwargs.items()},
                    result=str(result)[:5000] if result is not None else None,
                    exception=None,
                    started_at=started_at,
                    ended_at=ended_at,
                )
                
                return result
            except Exception as e:
                ended_at = datetime.now(timezone.utc)
                trace.add_function_event(
                    function_name=tool_name or func.__name__,
                    module_name=func.__module__ or None,
                    args={"args": [str(arg)[:1000] for arg in args]},
                    kwargs={k: str(v)[:1000] for k, v in kwargs.items()},
                    result=None,
                    exception=str(e)[:5000],
                    started_at=started_at,
                    ended_at=ended_at,
                )
                raise
        
        return wrapper
    return decorator


def get_trace() -> Optional[TraceBuilder]:
    """Get current trace builder from thread-local storage"""
    return getattr(_thread_local, 'trace', None)
