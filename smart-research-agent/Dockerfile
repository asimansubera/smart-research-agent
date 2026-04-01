# Stage 1: dependency builder
FROM python:3.12-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libffi-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2: runtime image
FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.title="Smart Research Agent"
LABEL org.opencontainers.image.version="1.0.0"
WORKDIR /app
COPY --from=builder /install /usr/local
COPY mcp_server/ ./mcp_server/
COPY agent/      ./agent/
COPY tests/      ./tests/
RUN mkdir -p /app/data
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AGENT_USER_ID=docker-user
CMD ["python", "-m", "agent.main"]
