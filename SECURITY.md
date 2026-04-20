# Security Policy

## Reporting a vulnerability

**Do not open public GitHub issues for security vulnerabilities.**

If you believe you have found a security vulnerability in `ddqn-router`,
please email the maintainer privately at:

> `security@<maintainer-domain>` *(maintainer: please fill in)*

Include:

- A clear description of the issue.
- Step-by-step reproduction instructions (or a minimal PoC).
- The version of `ddqn-router` you observed the issue on.
- Any relevant stack traces, logs, or configuration.

You should receive an acknowledgement within 7 days. We will coordinate a
patch and a disclosure timeline with you before publishing any advisory.

## Supported versions

Only the latest minor release receives security patches. Older minors should
be upgraded.

| Version | Supported |
| ------- | --------- |
| 0.3.x   | ✅         |
| < 0.3   | ❌         |

## Deployment notes

`ddqn-router serve` exposes an **unauthenticated** FastAPI endpoint by design
(see "Out of scope" in the productionization plan). In production, put it
behind a reverse proxy that terminates TLS and enforces authentication,
rate limiting, and allow-lists. Do not expose the `/route` endpoint directly
to the public internet.
