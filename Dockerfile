# ===== 构建阶段 =====
FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ===== 运行阶段 =====
FROM python:3.13-slim

WORKDIR /app
COPY --from=builder /install /usr/local

COPY main.py .
COPY src/ ./src/

EXPOSE 8080
CMD ["python", "main.py"]