# Build stage
FROM swift:6.0-jammy AS builder

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    libxml2-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy package files first for better caching
COPY Package.swift Package.resolved ./

# Resolve dependencies
RUN swift package resolve

# Copy source code
COPY Sources ./Sources
COPY Tests ./Tests

# Build release
RUN swift build -c release --static-swift-stdlib

# Runtime stage
FROM ubuntu:24.04 AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libxml2 \
    libcurl4 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r pagedattention && useradd -r -g pagedattention pagedattention

WORKDIR /app

# Copy binaries from builder
COPY --from=builder /app/.build/release/PagedAttentionServer /app/PagedAttentionServer
COPY --from=builder /app/.build/release/MinimalLLM /app/MinimalLLM
COPY --from=builder /app/.build/release/Profiler /app/Profiler

# Create directories
RUN mkdir -p /app/models /app/logs /app/config && \
    chown -R pagedattention:pagedattention /app

USER pagedattention

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Configuration
ENV RUST_LOG=info \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8080 \
    MAX_BATCH_SIZE=16 \
    MAX_SEQUENCES=128 \
    MAX_BLOCKS=2048 \
    BLOCK_SIZE=16 \
    LOG_LEVEL=info

# Entry point
ENTRYPOINT ["/app/PagedAttentionServer"]