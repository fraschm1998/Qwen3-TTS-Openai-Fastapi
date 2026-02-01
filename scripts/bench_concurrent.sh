#!/usr/bin/env bash
# Benchmark TTS throughput at different concurrency levels.
# Usage: ./scripts/bench_concurrent.sh [host]
# Example: ./scripts/bench_concurrent.sh 192.168.10.14

set -euo pipefail

HOST="${1:-localhost}"
URL="http://${HOST}:8880/v1/audio/speech"
TEXT="This is a benchmark test sentence to measure throughput at different concurrency levels with vLLM."
TMPDIR=$(mktemp -d)

bench() {
    local concurrency=$1
    local pids=()
    local start end elapsed

    echo "-------------------------------------------"
    echo "Concurrency: $concurrency"
    echo "-------------------------------------------"

    # Warmup (single request, discard)
    curl -s -X POST "$URL" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"qwen3-tts\",\"input\":\"warmup\",\"voice\":\"Vivian\"}" \
        -o /dev/null

    start=$(date +%s%N)

    for i in $(seq 1 "$concurrency"); do
        curl -s -X POST "$URL" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"qwen3-tts\",\"input\":\"$TEXT\",\"voice\":\"Vivian\"}" \
            -o "${TMPDIR}/out_${concurrency}_${i}.mp3" &
        pids+=($!)
    done

    # Wait for all to finish
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))  # ms
    per_request=$(( elapsed / concurrency ))
    rps=$(echo "scale=2; $concurrency * 1000 / $elapsed" | bc)

    echo "  Total time:    ${elapsed}ms"
    echo "  Per request:   ${per_request}ms (avg)"
    echo "  Throughput:    ${rps} req/s"
    echo ""
}

echo ""
echo "========================================="
echo "  TTS Concurrent Throughput Benchmark"
echo "  Host: $HOST"
echo "========================================="
echo ""

for c in 1 4 8 12; do
    bench "$c"
done

# Cleanup
rm -rf "$TMPDIR"

echo "Done."
