"""Pure request routing for the loopback learning-lab API."""

from collections.abc import Callable

from runtime.execution import detect_capabilities, run_provider


def handle_api_request(
    method: str,
    path: str,
    body: dict[str, object] | None,
    *,
    capabilities: Callable[[], dict[str, object]] = detect_capabilities,
    runner: Callable[..., dict[str, object]] = run_provider,
) -> tuple[int, dict[str, object]]:
    if method == "GET" and path == "/api/capabilities":
        return 200, capabilities()
    if method == "POST" and path == "/api/run":
        if not isinstance(body, dict):
            return 400, {"error": "request body must be a JSON object"}
        provider = body.get("provider")
        group_size = body.get("group_size", 10)
        confirmed = body.get("confirmed", False)
        if provider not in {"apple-mlx", "modal-triton"}:
            return 400, {"error": "unknown execution provider"}
        if not isinstance(group_size, int) or group_size <= 0 or group_size > 1024:
            return 400, {"error": "group_size must be an integer from 1 to 1024"}
        if not isinstance(confirmed, bool):
            return 400, {"error": "confirmed must be a boolean"}
        try:
            return 200, runner(provider, confirmed=confirmed, group_size=group_size)
        except PermissionError as error:
            return 403, {"error": str(error)}
        except (RuntimeError, ValueError):
            return 422, {"error": "GPU execution failed. Check the companion-server terminal for details."}
    return 404, {"error": "not found"}
