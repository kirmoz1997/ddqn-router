"""ddqn-router: Double DQN-based router for multi-agent systems."""

from ddqn_router.inference.router import DDQNRouter, RouteResult, RouterNotTrainedError

__version__ = "0.1.0"
__all__ = ["DDQNRouter", "RouteResult", "RouterNotTrainedError", "__version__"]
