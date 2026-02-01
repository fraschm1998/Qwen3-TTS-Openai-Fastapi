# Default host for benchmarks
host := "localhost"

# Start N replicas behind nginx (1-4)
run n="2":
    #!/usr/bin/env bash
    set -euo pipefail
    n={{n}}
    if [[ "$n" -lt 1 || "$n" -gt 4 ]]; then
        echo "Error: 1-4 replicas supported"
        exit 1
    fi
    services="nginx"
    for i in $(seq 1 "$n"); do
        services="$services qwen3-tts-$i"
    done
    docker compose up -d $services

# Stop all containers
stop:
    docker compose down

# Rebuild and start N replicas
rebuild n="2":
    docker compose build
    just run {{n}}

# Show running containers and GPU usage
status:
    docker ps --filter "name=qwen3-tts" --format "table {{"{{"}}.Names{{"}}"}}\t{{"{{"}}.Status{{"}}"}}\t{{"{{"}}.Ports{{"}}"}}"
    @echo ""
    nvidia-smi

# Tail logs from all running containers
logs *args="":
    docker compose logs -f {{args}}

# Run benchmark against host
bench concurrency="1 4 8 12":
    ./scripts/bench_concurrent.sh {{host}}

# Start single instance (no nginx)
run-single:
    docker compose --profile single up -d qwen3-tts-gpu

# Start vLLM backend
run-vllm:
    docker compose --profile vllm up -d qwen3-tts-vllm

# Start CPU-only backend
run-cpu:
    docker compose --profile cpu up -d qwen3-tts-cpu
