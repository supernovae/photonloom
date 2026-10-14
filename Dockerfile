# Multistage Linux CPU build & runtime for Photonloom
FROM rust:1-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    clang \
    libclang-dev \
    pkg-config \
    zlib1g-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

COPY Cargo.toml Cargo.lock* ./
COPY src ./src
COPY examples ./examples
COPY tests ./tests
COPY ml ./ml

RUN cargo build --release --bin photonloom

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libstdc++6 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 photonloom \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin photonloom \
    && mkdir -p /observations /results \
    && chown -R photonloom:photonloom /results

COPY --from=builder /src/target/release/photonloom /usr/local/bin/photonloom

USER 10001:10001
WORKDIR /home/photonloom

ENTRYPOINT ["photonloom"]
CMD ["run", "/observations", "--output", "/results", "--accelerator", "cpu"]
