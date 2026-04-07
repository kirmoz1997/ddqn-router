"""Optional FastAPI server for routing inference (requires 'serve' extras)."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fastapi import FastAPI as _FastAPI


def create_app(
    artifacts_path: str,
    cors_origins: Optional[list[str]] = None,
) -> "_FastAPI":
    """Create and return a FastAPI app that serves routing inference.

    Args:
        artifacts_path: Path to the directory with trained model artifacts.
        cors_origins: List of allowed CORS origins. Pass ``["*"]`` to
            allow all origins. ``None`` disables CORS middleware.

    Requires the ``serve`` extras: ``pip install ddqn-router[serve]``
    """
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError(
            "FastAPI is required for the serve module. "
            "Install with: pip install ddqn-router[serve]"
        ) from None

    from pydantic import BaseModel as PydanticBaseModel

    from ddqn_router.inference.router import DDQNRouter

    ddqn_router = DDQNRouter.load(artifacts_path)
    app = FastAPI(title="ddqn-router", version="0.2.0")

    if cors_origins is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    class RouteRequest(PydanticBaseModel):
        query: str

    class RouteBatchRequest(PydanticBaseModel):
        queries: list[str]

    class RouteResponse(PydanticBaseModel):
        agents: list[int]
        agent_names: list[str]
        confidence: float
        steps: int

    @app.post("/route", response_model=RouteResponse)
    def route(request: RouteRequest) -> RouteResponse:
        result = ddqn_router.route(request.query)
        return RouteResponse(
            agents=result.agents,
            agent_names=result.agent_names,
            confidence=result.confidence,
            steps=result.steps,
        )

    @app.post("/route/batch", response_model=list[RouteResponse])
    def route_batch(request: RouteBatchRequest) -> list[RouteResponse]:
        results = ddqn_router.route_batch(request.queries)
        return [
            RouteResponse(
                agents=r.agents,
                agent_names=r.agent_names,
                confidence=r.confidence,
                steps=r.steps,
            )
            for r in results
        ]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/agents")
    def agents() -> dict:
        return {"agents": ddqn_router.agents}

    return app
