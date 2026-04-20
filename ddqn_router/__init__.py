"""ddqn-router: Double DQN-based router for multi-agent systems."""

from ddqn_router.inference.router import (
    DDQNRouter,
    RouteResult,
    RouterNotTrainedError,
    StepTrace,
)

__version__ = "0.3.0"
__all__ = [
    "DDQNRouter",
    "RouteResult",
    "RouterNotTrainedError",
    "StepTrace",
    "__version__",
]
