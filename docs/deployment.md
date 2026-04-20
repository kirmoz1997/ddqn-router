# Deployment

`ddqn-router serve` exposes a FastAPI app that loads a trained artifacts
directory and answers routing requests. Inference is CPU-bound and
sub-millisecond per query.

## Docker

The published image is `ghcr.io/kirmoz1997/ddqn-router:<version>` (also
`:latest`). Multi-arch (`linux/amd64`, `linux/arm64`).

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts:/artifacts:ro" \
  ghcr.io/kirmoz1997/ddqn-router:latest
```

```bash
curl -s -X POST http://localhost:8000/route \
  -H 'content-type: application/json' \
  -d '{"query":"my invoice was charged twice"}' | jq
```

## docker compose

```yaml
services:
  router:
    image: ghcr.io/kirmoz1997/ddqn-router:latest
    ports: ["8000:8000"]
    volumes:
      - ./artifacts:/artifacts:ro
```

`docker compose up -d` and you're serving.

## Bare-metal FastAPI

```bash
pip install "ddqn-router[serve]"
ddqn-router serve --artifacts ./artifacts --host 0.0.0.0 --port 8000
```

Behind a systemd unit, a Procfile, or your process manager of choice.

## Production checklist

- **TLS termination + authentication at the reverse proxy.** The bundled
  FastAPI app has no auth, no rate limiting, no metrics by design. Put
  nginx/Caddy/Traefik/Cloudflare in front and enforce auth there.
- **Restrict artifacts to read-only.** The container volume mount already
  uses `:ro`; do the same on bare metal.
- **Resource limits.** Inference is CPU-bound and fast; 1 CPU per ~1000
  req/s is a reasonable starting point. Memory footprint is ~100–300 MB.
- **Warmup.** The first request JIT-compiles torch ops. Fire a throwaway
  request on boot (healthcheck `/health` does not trigger the model).
- **Versioning.** Pin the image to an explicit `v0.x.y` tag in production —
  don't use `:latest`.
- **Observability.** Log at the reverse-proxy layer. If you need custom
  metrics, wrap the FastAPI app yourself (the app is returned by
  `ddqn_router.serve.app.create_app`).
- **Graceful rollouts.** The artifacts directory is mounted read-only; to
  roll out a new model, build a new image or swap the volume and restart.
