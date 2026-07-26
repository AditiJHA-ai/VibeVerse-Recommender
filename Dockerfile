# ---- frontend build ----
FROM node:22-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- backend runtime ----
FROM python:3.12-slim
WORKDIR /app

# Render / most PaaS inject PORT at runtime (often 10000).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=10000 \
    RELOAD=0

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY data ./data
COPY main_dataframe.pkl ./main_dataframe.pkl
COPY --from=frontend /app/frontend/dist ./frontend/dist

EXPOSE 10000
CMD ["sh", "-c", "uvicorn backend.api:app --host 0.0.0.0 --port ${PORT:-10000}"]
