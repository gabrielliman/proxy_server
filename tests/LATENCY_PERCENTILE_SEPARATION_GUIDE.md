# Latency-Based Request Separation Guide

## Overview

The benchmark tool now includes functionality to separate and analyze requests based on their end-to-end (E2EL) latency percentiles. This allows you to understand performance characteristics at different latency thresholds.

## Key Components

### 1. **calculate_percentile_thresholds()**
Calculates latency thresholds at specified percentiles (default: 50, 75, 90, 95, 99).

**Returns:** Dictionary mapping percentile → threshold latency in seconds

Example:
```python
thresholds = calculate_percentile_thresholds(requests)
# Output: {50: 0.145, 75: 0.210, 90: 0.285, 95: 0.340, 99: 0.512}
```

### 2. **separate_requests_by_latency()**
Separates a program's requests into percentile buckets and calculates detailed metrics for each.

**Parameters:**
- `program`: ProgramMetrics object
- `percentiles`: List of percentiles to calculate (default: [50, 75, 90, 95, 99])

**Returns:** Dictionary containing:
- `program_id`: ID of the program
- `thresholds`: Latency thresholds for each percentile
- `percentile_buckets`: Lists of requests that qualify for each percentile
- `metrics_per_percentile`: Performance metrics for each percentile group

**Metrics per percentile include:**
- `threshold_ms`: Threshold latency in milliseconds
- `num_valid_requests`: Count of requests within threshold
- `valid_request_percentage`: Percentage of requests within threshold
- `request_goodput_rps`: Requests per second (valid only)
- `output_token_goodput`: Output tokens per second (valid only)
- `total_token_goodput`: Total tokens per second (valid only)
- E2EL/TTFT/TPOT/ITL statistics (mean, median, p95, p99)

### 3. **separate_all_programs_by_latency()**
Applies request separation to all programs in the benchmark.

**Returns:** Dictionary mapping program_id → separation results for each program

## Usage Example

### Running the benchmark with latency separation:

```bash
python benchmark_stateful.py \
  --dataset /path/to/dataset.json \
  --base-url http://localhost:8080 \
  --model Qwen/Qwen3-4B \
  --limit 20 \
  --output-json output.json \
  --request-rate 1
```

### Output Structure

The generated JSON file includes:

```json
{
  "full_metrics": { ... },
  "per_program": [ ... ],
  "latency_separation": {
    "program_id_1": {
      "thresholds": { 50: 0.145, 75: 0.210, ... },
      "metrics_per_percentile": {
        50: { ... },
        75: { ... },
        ...
      }
    },
    ...
  },
  "full_latency_separation": {
    "thresholds": { 50: 0.145, 75: 0.210, ... },
    "metrics_per_percentile": { ... }
  }
}
```

### Console Output

You'll see a detailed breakdown like:

```
======== LATENCY PERCENTILE SEPARATION (FULL DATASET) ========

Percentile P50:
  Threshold: 145.23ms
  Valid Requests: 512 (100.0%)
  Request Goodput: 2.45 rps
  Token Goodput: 125.34 tps
  E2EL - Median: 143.45ms, P99: 144.87ms

Percentile P75:
  Threshold: 210.56ms
  Valid Requests: 512 (100.0%)
  Request Goodput: 2.45 rps
  Token Goodput: 125.34 tps
  E2EL - Median: 195.23ms, P99: 210.12ms

...
```

## Interpretation Guide

### Understanding the Buckets

- **P50 bucket**: Requests with latency ≤ 50th percentile (fastest 50%)
- **P75 bucket**: Requests with latency ≤ 75th percentile (fastest 75%)
- **P90 bucket**: Requests with latency ≤ 90th percentile (fastest 90%)
- **P95 bucket**: Requests with latency ≤ 95th percentile (fastest 95%)
- **P99 bucket**: All valid requests (up to 99th percentile)

### Key Metrics to Watch

1. **valid_request_percentage**: How many requests fall within each percentile
   - Should increase as percentile goes up (e.g., P50 < P75 < P90)

2. **request_goodput_rps** and **token_goodput**: 
   - Shows throughput considering only requests within the latency budget
   - Useful for understanding SLA compliance

3. **Goodput vs Throughput**:
   - **Throughput**: Overall performance (all requests)
   - **Goodput**: Performance considering only SLA-compliant requests

## Use Cases

1. **SLA Analysis**: Determine what percentage of requests meet different latency targets
2. **Performance Characterization**: Understand how throughput changes with stricter latency requirements
3. **Scheduling Optimization**: Compare different schedulers at specific latency targets
4. **Load Balancer Evaluation**: Measure goodput under various latency constraints

## API Reference

### Function: `separate_requests_by_latency(program, percentiles=[50, 75, 90, 95, 99])`

```python
from benchmark_stateful import separate_requests_by_latency, ProgramMetrics

# Assuming you have a ProgramMetrics object
separation_result = separate_requests_by_latency(program)

# Access results
thresholds = separation_result["thresholds"]  # {50: 0.145, ...}
buckets = separation_result["percentile_buckets"]  # {50: [req1, req2, ...], ...}
metrics = separation_result["metrics_per_percentile"]  # {50: {...}, ...}

# Get specific metric for P90
p90_metrics = metrics[90]
print(f"P90 Threshold: {p90_metrics['threshold_ms']}ms")
print(f"Valid Requests: {p90_metrics['num_valid_requests']}")
print(f"Request Goodput: {p90_metrics['request_goodput_rps']} rps")
```

## Integration with Existing Code

The latency separation is automatically computed during benchmark execution:

```python
per_program, full_metrics, latency_separation, full_latency_separation = asyncio.run(
    benchmark_sharegpt(...)
)

# Access per-program separation
prog1_separation = latency_separation["program_id_1"]
p90_requests = prog1_separation["percentile_buckets"][90]

# Access full dataset separation
full_p90_metrics = full_latency_separation["metrics_per_percentile"][90]
```

## Notes

- Percentile buckets are **inclusive** (threshold ≤ latency)
- Requests with 0 latency are filtered out
- All latency values are stored internally in seconds but converted to milliseconds in output
- The function handles empty request lists gracefully, returning 0/None values

