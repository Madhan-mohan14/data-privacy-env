FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY data_privacy_env/pyproject.toml data_privacy_env/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY data_privacy_env/ .
RUN uv sync --frozen --no-dev

EXPOSE 7860

CMD ["/bin/sh", "-c", ".venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 7860"]
