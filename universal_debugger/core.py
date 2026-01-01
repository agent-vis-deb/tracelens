"""
Core trace building functionality - THIN SDK VERSION
Just collects raw events, no processing or classification
"""

import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def format_datetime_for_api(dt: datetime) -> str:
    """
    Format datetime as ISO-8601 string compatible with Zod datetime validator.
    Ensures consistent format: YYYY-MM-DDTHH:MM:SS.sssZ for UTC
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    
    iso_str = dt.isoformat()
    
    if iso_str.endswith('+00:00'):
        iso_str = iso_str[:-6] + 'Z'
    elif iso_str.endswith('-00:00'):
        iso_str = iso_str[:-6] + 'Z'
    elif not iso_str.endswith('Z') and dt.tzinfo == timezone.utc:
        if '+' not in iso_str and '-' not in iso_str[-6:]:
            iso_str += 'Z'
    
    return iso_str


class TraceBuilder:
    """
    Thin event collector - just captures raw events with minimal metadata.
    All processing (classification, provider detection, model extraction, token computation)
    happens on the backend.
    """
    
    def __init__(self, app_name: str, environment: Optional[str] = None):
        self.app_name = app_name
        self.environment = environment or os.getenv("DEBUGGER_ENV", "production")
        self.events: List[Dict[str, Any]] = []  # Raw events, not processed steps
        self.event_counter = 0
        self.logs: List[Dict[str, Any]] = []  # Logs for this trace
    
    def _next_event_id(self) -> str:
        self.event_counter += 1
        return f"event_{self.event_counter}"
    
    def add_http_event(
        self,
        method: str,
        url: str,
        request_headers: Optional[Dict[str, str]] = None,
        request_body: Optional[Any] = None,
        response_status: Optional[int] = None,
        response_headers: Optional[Dict[str, str]] = None,
        response_body: Optional[Any] = None,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
    ) -> str:
        """
        Add raw HTTP request/response event.
        No classification or processing - backend will handle that.
        """
        event_id = self._next_event_id()
        
        if started_at is None:
            started_at = datetime.now(timezone.utc)
        if ended_at is None:
            ended_at = datetime.now(timezone.utc)
        
        # Serialize request/response bodies if needed
        def serialize_body(body):
            if body is None:
                return None
            if isinstance(body, (str, int, float, bool, type(None))):
                return body
            if isinstance(body, (dict, list)):
                return body  # Already JSON-serializable
            try:
                return json.loads(body) if isinstance(body, str) else body
            except:
                return str(body)[:10000]  # Fallback to string, limit size
        
        event = {
            "id": event_id,
            "type": "http",
            "method": method.upper(),
            "url": str(url)[:2000],  # Limit URL length
            "started_at": format_datetime_for_api(started_at),
            "ended_at": format_datetime_for_api(ended_at),
        }
        
        # Only include optional fields if they have values (avoid null)
        if request_headers:
            event["request_headers"] = {k: str(v)[:500] for k, v in request_headers.items()}
        if request_body is not None:
            event["request_body"] = serialize_body(request_body)
        if response_status is not None:
            event["response_status"] = response_status
        if response_headers:
            event["response_headers"] = {k: str(v)[:500] for k, v in response_headers.items()}
        if response_body is not None:
            event["response_body"] = serialize_body(response_body)
        
        self.events.append(event)
        return event_id
    
    def add_function_event(
        self,
        function_name: str,
        module_name: Optional[str] = None,
        args: Optional[Any] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        exception: Optional[Any] = None,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
    ) -> str:
        """
        Add raw function call event.
        No classification or processing - backend will determine if it's a tool.
        """
        event_id = self._next_event_id()
        
        if started_at is None:
            started_at = datetime.now(timezone.utc)
        if ended_at is None:
            ended_at = datetime.now(timezone.utc)
        
        # Serialize args/kwargs/result if needed
        def serialize_value(val):
            if val is None or isinstance(val, (str, int, float, bool)):
                return val
            if isinstance(val, (dict, list)):
                try:
                    # Try to keep it as-is if JSON-serializable
                    json.dumps(val)
                    return val
                except:
                    return str(val)[:5000]  # Fallback, limit size
            return str(val)[:5000]  # Fallback for other types
        
        event = {
            "id": event_id,
            "type": "function",
            "function_name": str(function_name)[:500],
            "started_at": format_datetime_for_api(started_at),
            "ended_at": format_datetime_for_api(ended_at),
        }
        
        # Only include optional fields if they have values (avoid null)
        if module_name:
            event["module_name"] = str(module_name)[:500]
        if args is not None:
            event["args"] = serialize_value(args)
        if kwargs is not None:
            event["kwargs"] = serialize_value(kwargs)
        if result is not None:
            event["result"] = serialize_value(result)
        if exception:
            event["exception"] = str(exception)[:5000]
        
        self.events.append(event)
        return event_id
    
    def add_log(
        self,
        level: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Add a log entry to the trace.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        log_entry = {
            "level": level,
            "message": str(message)[:10000],
            "timestamp": format_datetime_for_api(timestamp),
            "metadata": metadata or {}
        }
        
        self.logs.append(log_entry)
