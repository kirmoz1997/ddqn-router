"""Optional FastAPI server for routing inference (requires 'serve' extras)."""

from __future__ import annotations

try:
    from fastapi import FastAPI
    from pydantic import BaseModel as PydanticBaseModel
except ImportError:
    raise ImportError(
        "FastAPI is required for the serve module. "
        "Install with: pip install ddqn-router[serve]"
    )

from ddqn_router.inference.router import DDQNRouter

_router: DDQNRouter | None = None
app = FastAPI(title="ddqn-router", version="0.1.0")


class RouteRequest(PydanticBaseModel):
    query: str


class RouteResponse(PydanticBaseModel):
    agents: list[int]
    agent_names: list[str]
    confidence: float
    steps: int


def create_app(artifacts_path: str) -> FastAPI:
    global _router
    _router = DDQNRouter.load(artifacts_path)
    return app


@app.post("/route", response_model=RouteResponse)
def route(request: RouteRequest) -> RouteResponse:
    assert _router is not None, "Router not loaded. Call create_app() first."
    result = _router.route(request.query)
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
    assert _router is not None, "Router not loaded. Call create_app() first."
    return {"agents": _router.agents}
