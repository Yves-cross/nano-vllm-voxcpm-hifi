from __future__ import annotations

from typing import Any, cast

from fastapi import HTTPException, Request


def get_server(request: Request) -> Any:
    server = getattr(request.app.state, "server", None)
    if server is None:
        raise HTTPException(status_code=503, detail="Model server not ready")
    # app.state is dynamically typed; normalize for type checkers.
    return cast(Any, server)
