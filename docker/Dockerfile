FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    git openssh-client ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

ARG REQUIREMENTS_FILE=requirements.txt
COPY docker/${REQUIREMENTS_FILE} /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

RUN git config --global --add safe.directory /workspace

CMD ["/bin/bash", "-lc", "sleep infinity"]
