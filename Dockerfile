# syntax=docker/dockerfile:1

# One Python for every stage: the official slim image (Python 3.14.10, Debian 13), pinned by digest so a
# rebuild gets the same system. To update: docker buildx imagetools inspect python:<tag>, take the index digest.
ARG PYTHON_IMAGE=python:3.14.10-slim-trixie@sha256:f57c4c23b985d95deea5bc5249e2f9c8a512b4cc9d5d3fd31900cbd2f5f0a604

# ---- builder: the locked dependencies, then the app, installed (not editable) into /app/.venv ----
FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.10@sha256:a3ea1357342c1a4bf9a0c53a3deb1b55379aefa037338d8113d5771efa01774f /uv /uvx /bin/
# Bytecode compiled at build time: the runtime's file system is read-only. copy: the cache is a separate mount.
# The image's own Python in every stage, never a downloaded one.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1 UV_PYTHON_DOWNLOADS=0
WORKDIR /app
# Dependencies only: this layer is rebuilt when pyproject.toml or uv.lock change, not when the code does.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

# ---- test: the same environment plus the dev group; the suite runs against the installed package ----
#   docker build --target test -t invoice-recognition-test . && docker run --rm --network none invoice-recognition-test
FROM builder AS test
ENV UV_NO_DEV=0
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable
CMD ["/app/.venv/bin/pytest", "-q", "-p", "no:cacheprovider"]

# ---- runtime (the default target): the environment only; no uv, no sources, not root ----
FROM ${PYTHON_IMAGE} AS runtime
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --home-dir /nonexistent \
    --shell /usr/sbin/nologin app
# Owned by root: the process can read its code but not change it.
COPY --from=builder /app/.venv /app/.venv
# MALLOC_ARENA_MAX: glibc's per-thread arenas let peak memory grow with every thread that decodes a large image
# (about 1.5 GiB was measured without this). With two, under compose.yaml's limits, peaks of up to about 1.3 GiB
# were measured with JOBS_KEPT=50 and about 1.0 GiB with JOBS_KEPT=20.
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2
# Settings come from the environment; there is no .env in the image (.dockerignore).
WORKDIR /app
USER app
EXPOSE 8000
# Straight to the app, never through a proxy set in the environment (HTTP_PROXY would otherwise apply).
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --start-interval=2s --retries=3 \
  CMD ["python", "-c", "import urllib.request as u; u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:8000/health', timeout=2)"]
# One process: jobs live in its memory. 0.0.0.0 inside the container; compose publishes the port on the host's
# 127.0.0.1 only: processes on this machine reach it, and so do containers on it (through host.docker.internal
# with Docker Desktop, or at app:8000 on the compose project's own network); other machines on the network do
# not. Forwarded headers are trusted from any peer so that the access log shows the visitor's real address
# behind a proxy; the app uses the client address for nothing else.
CMD ["uvicorn", "invoice_recognition.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--no-server-header", \
     "--timeout-graceful-shutdown", "5"]
