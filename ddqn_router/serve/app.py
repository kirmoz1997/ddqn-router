"""Optional FastAPI server for routing inference (requires 'serve' extras)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI as _FastAPI

_router_instance = None
_app_instance = None


def create_app(artifacts_path: str) -> _FastAPI:
    """Create and return a FastAPI app that serves routing inference.

    Requires the ``serve`` extras: ``pip install ddqn-router[serve]``
    """
    try:
        from fastapi import FastAPI
    except ImportError:
        raise ImportError(
            "FastAPI is required for the serve module. "
            "Install with: pip install ddqn-router[serve]"
        ) from None

    from pydantic import BaseModel as PydanticBaseModel

    from ddqn_router.inference.router import DDQNRouter

    global _router_instance, _app_instance

    _router_instance = DDQNRouter.load(artifacts_path)
    app = FastAPI(title="ddqn-router", version="0.1.0")

    class RouteRequest(PydanticBaseModel):
        query: str

    class RouteResponse(PydanticBaseModel):
        agents: list[int]
        agent_names: list[str]
        confidence: float
        steps: int

    @app.post("/route", response_model=RouteResponse)
    def route(request: RouteRequest) -> RouteResponse:
        result = _router_instance.route(request.query)
        return RouteResponse(
            agents=result.agents,
            agent_names=result.agent_names,
            confidence=result.confidence,
            steps=result.steps,
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/agents")
    def agents() -> dict:
        return {"agents": _router_instance.agents}

    _app_instance = app
    return app
