FROM python:3.11-slim

ARG PKG_VERSION=

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 1000 router \
    && useradd  --system --gid router --uid 1000 --shell /usr/sbin/nologin router

RUN if [ -n "$PKG_VERSION" ]; then \
        pip install "ddqn-router[serve]==$PKG_VERSION" ; \
    else \
        pip install "ddqn-router[serve]" ; \
    fi

USER router
WORKDIR /home/router

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["ddqn-router", "serve", "--artifacts", "/artifacts", "--host", "0.0.0.0", "--port", "8000"]
