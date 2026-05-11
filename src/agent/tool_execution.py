import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any, Callable


def call_tool_safely(func: Callable[..., Any], logger: Callable[[str], None], *args, **kwargs):
    """Run tool code with stdout/stderr capture so MCP transport stays clean."""
    out_buf = StringIO()
    err_buf = StringIO()
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            result = func(*args, **kwargs)
    except Exception as exc:
        logger(f"[HOST] Tool {func.__name__} crashed: {exc}\n")
        return json.dumps({"error": f"{func.__name__} failed: {str(exc)}"})

    leaked_stdout = out_buf.getvalue().strip()
    if leaked_stdout:
        logger(f"[HOST] Suppressed stdout leak ({len(leaked_stdout)} chars)\n")
    leaked_stderr = err_buf.getvalue().strip()
    if leaked_stderr:
        logger(f"[HOST] Suppressed stderr leak ({len(leaked_stderr)} chars)\n")
    return result


def call_in_subprocess(module_name: str, function_name: str, payload: dict, project_root: str, timeout: int = 120) -> str:
    """Isolate tool execution in a subprocess to protect the host runtime seam."""
    runner = (
        "import json, importlib, inspect, asyncio, io\n"
        "from contextlib import redirect_stdout, redirect_stderr\n"
        f"module = importlib.import_module({module_name!r})\n"
        f"func = getattr(module, {function_name!r})\n"
        f"payload = {repr(payload)}\n"
        "out_buf = io.StringIO()\n"
        "err_buf = io.StringIO()\n"
        "with redirect_stdout(out_buf), redirect_stderr(err_buf):\n"
        "    result = func(**payload)\n"
        "    if inspect.iscoroutine(result):\n"
        "        result = asyncio.run(result)\n"
        "print(result if isinstance(result, str) else json.dumps(result))\n"
    )
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{existing_pp}" if existing_pp else project_root

    proc = subprocess.run(
        [sys.executable, "-c", runner],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        return json.dumps(
            {
                "success": False,
                "error": f"{module_name}.{function_name} failed",
                "stderr": proc.stderr[-1000:],
            }
        )
    return proc.stdout.strip() or json.dumps({"success": False, "error": "No output from subprocess"})
