# The voice agent as one container, for Fly.io (fly.toml) or any Docker host.
#
#   docker build -t voice-agent .
#
# What's baked in: the code, agent.config.yaml, the knowledge base (kb/) and the voice clips
# (assets/audio/). kb/ and the clips are kept out of git but not out of this image: it's
# built from the local folder and pushed to the host's private registry, never to GitHub.
# What isn't: secrets (.env; set them as host secrets) and the database (data/, a mounted
# volume, so it survives deploys). See .dockerignore.

FROM python:3.13-slim-trixie

# The same uv as local development, so `uv sync --locked` reads uv.lock the same way.
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, in their own layer: a code change doesn't reinstall them.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Then the app itself.
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

ENV PATH="/app/.venv/bin:$PATH"

# One worker: the registry's call limits and the shared connections live in this process
# (design doc §14). --proxy-headers: the host's proxy terminates HTTPS in front of us.
EXPOSE 8080
CMD ["uvicorn", "voice_agent.app.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]
