# Stage 1: build the React frontend
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# Stage 2: Python server that also serves the built frontend
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY poker_ai/ poker_ai/
RUN pip install --no-cache-dir ".[server]"
COPY --from=web /web/dist web/dist
ENV PORT=8000 POKER_WEB_DIST=/app/web/dist
EXPOSE 8000
CMD ["sh", "-c", "uvicorn poker_ai.server.app:app --host 0.0.0.0 --port ${PORT}"]
