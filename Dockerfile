# syntax=docker/dockerfile:1
FROM python:3.11-slim

#----- General settings -----#
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

#----- OS dependencies -----#
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential git curl pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

#----- Rust (for bittensor native deps) -----#
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

#----- Python tooling: pipx + uv -----#
ENV PIPX_HOME=/opt/pipx \
    PIPX_BIN_DIR=/usr/local/bin
RUN pip install --no-cache-dir pipx && pipx install uv

#----- Project setup -----#
WORKDIR /app
COPY . .

# Create dedicated virtualenv and install deps via uv inside it
RUN /usr/local/bin/uv pip install --system -r requirements.txt

#----- Non-root execution -----#
RUN groupadd -g 1000 app && useradd -m -u 1000 -g app appuser

# Give appuser ownership of the app directory so it can write DB files
RUN chown -R appuser:app /app

# Persist volumes
VOLUME ["/data", "/home/appuser/.bittensor"]

# Switch user
USER appuser

# Expose axon port
EXPOSE 8091

# Entrypoint (override in compose)
CMD ["python", "neurons/validator.py", "--data-dir", "/data"] 