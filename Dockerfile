FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    curl \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    python3 -m pip install --upgrade pip

WORKDIR /app

RUN pip install --no-cache-dir \
    torch>=2.4.0 \
    fastapi>=0.115.0 \
    uvicorn>=0.30.0 \
    pydantic>=2.8.0

COPY kimi_long_context_engine.py .
COPY kimi_serving_gateway.py .

EXPOSE 8080
CMD ["python3", "kimi_serving_gateway.py"]
