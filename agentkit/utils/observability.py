import os
from functools import wraps
from typing import Callable
from loguru import logger

try:
    from langfuse.decorators import observe
    from langfuse import Langfuse
    _HAS_LANGFUSE = True
    
    if os.environ.get("LANGFUSE_SECRET_KEY") and os.environ.get("LANGFUSE_PUBLIC_KEY"):
        # Doğrulamak için instance yaratıyoruz
        _langfuse_client = Langfuse()
    else:
        _HAS_LANGFUSE = False
except ImportError:
    _HAS_LANGFUSE = False


def trace_agent(name: str = "agent_run"):
    """
    Eğer Langfuse kurulu ve environment ayarları doğruysa LLM/Agent generation'ı izler.
    Aksi takdirde fonksiyonu etkilemeden çalışmasını sağlar.
    """
    def decorator(func: Callable) -> Callable:
        if _HAS_LANGFUSE:
            return observe(name=name, as_type="generation")(func)
        return func
    return decorator


def trace_tool(name: str = "tool_execution"):
    """
    Araçların (tools) çalışma süresini ve parametrelerini Langfuse üzerinde bir Span olarak kaydeder.
    """
    def decorator(func: Callable) -> Callable:
        if _HAS_LANGFUSE:
            return observe(name=name, as_type="span")(func)
        return func
    return decorator
