import gc
import platform
from rich import print as rprint


def release_runtime_memory(tag: str = ""):
    """
    Release Python and torch runtime caches after heavy tasks.
    This does not guarantee RSS drops to zero, but helps reclaim reusable memory.
    """
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        # Keep cleanup best-effort and non-blocking.
        pass

    # Linux allocator hint.
    if platform.system().lower() == "linux":
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            if hasattr(libc, "malloc_trim"):
                libc.malloc_trim(0)
        except Exception:
            pass

    if tag:
        rprint(f"[cyan]🧹 Runtime memory cleanup completed: {tag}[/cyan]")

