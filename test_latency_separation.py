#!/usr/bin/env python3
"""
Simple test script for latency-based request separation.
Run without needing to start benchmark servers.
"""

import sys
import numpy as np
from benchmark_stateful import (
    RequestMetrics,
    ProgramMetrics,
    calculate_percentile_thresholds,
    separate_requests_by_latency,
    percentile,
)


def create_mock_requests(num_requests=100, seed=42):
    """
    Create mock request metrics with realistic latency distribution.
    Uses a lognormal distribution to simulate real LLM latencies.
    """
    np.random.seed(seed)
    
    # Lognormal distribution simulates skewed latencies (some fast, some slow)
    latencies = np.random.lognormal(mean=np.log(0.5), sigma=0.5, size=num_requests)
    
    requests = []
    for i, latency in enumerate(latencies):
        # Simple model: for a 0.5s latency with 100 output tokens
        # TTFT ~ 50ms, rest is per-token time
        output_tokens = np.random.randint(20, 200)
        ttft = 0.05  # 50ms
        
        req = RequestMetrics(
            ttft=ttft,
            latency=float(latency),
            itl=[(latency - ttft) / (output_tokens - 1)] * (output_tokens - 1) if output_tokens > 1 else [],
            output_tokens=int(output_tokens),
            input_tokens=np.random.randint(10, 100),
            start_time=float(i * 0.01),
            end_time=float(i * 0.01 + latency),
        )
        requests.append(req)
    
    return requests


def test_basic_separation():
    """Test latency separation with mock data."""
    print("=" * 70)
    print("TEST 1: Basic Latency Separation")
    print("=" * 70)
    
    # Create mock program
    requests = create_mock_requests(num_requests=100)
    program = ProgramMetrics(program_id="test_program_1", requests=requests)
    
    # Get raw latencies
    latencies_ms = [r.latency * 1000 for r in requests]
    print(f"\n📊 Generated {len(requests)} mock requests")
    print(f"   Latency range: {min(latencies_ms):.2f}ms - {max(latencies_ms):.2f}ms")
    
    # Run separation
    result = separate_requests_by_latency(program, percentiles=[50, 75, 90, 95, 99])
    
    print(f"\n🎯 Percentile Thresholds (End-to-End Latency):")
    for p in sorted(result["thresholds"].keys()):
        threshold_ms = result["thresholds"][p] * 1000
        print(f"   P{p}: {threshold_ms:.2f}ms")
    
    print(f"\n📈 Metrics by Percentile:")
    for p in sorted(result["metrics_per_percentile"].keys()):
        print(p, result["metrics_per_percentile"][p]["threshold_ms"])
        metrics = result["metrics_per_percentile"][p]
        valid_count = metrics["num_valid_requests"]
        valid_pct = metrics["valid_request_percentage"]
        goodput = metrics["request_goodput_rps"]
        token_goodput = metrics["output_token_goodput"]
        
        print(f"\n   P{p}:")
        print(f"     ├─ Threshold: {metrics['threshold_ms']:.2f}ms")
        print(f"     ├─ Valid requests: {valid_count}/{len(requests)} ({valid_pct:.1f}%)")
        print(f"     ├─ Request goodput: {goodput:.2f} rps" if goodput else "     ├─ Request goodput: N/A")
        print(f"     ├─ Token goodput: {token_goodput:.2f} tok/s" if token_goodput else "     ├─ Token goodput: N/A")
        print(f"     └─ Median E2EL in bucket: {metrics['median_e2el_ms']:.2f}ms" if metrics['median_e2el_ms'] else "     └─ Median E2EL: N/A")


def test_percentile_calculation():
    """Test percentile threshold calculation."""
    print("\n" + "=" * 70)
    print("TEST 2: Percentile Threshold Calculation")
    print("=" * 70)
    
    requests = create_mock_requests(num_requests=1000)
    
    thresholds = calculate_percentile_thresholds(requests, percentiles=[25, 50, 75, 90, 99])
    
    latencies_ms = [r.latency * 1000 for r in requests]
    actual_p50 = np.percentile(latencies_ms, 50)
    actual_p99 = np.percentile(latencies_ms, 99)
    
    print(f"\n✓ Calculated thresholds vs numpy percentiles:")
    print(f"   P50:  Calculated={thresholds[50]*1000:.2f}ms, NumPy={actual_p50:.2f}ms")
    print(f"   P99:  Calculated={thresholds[99]*1000:.2f}ms, NumPy={actual_p99:.2f}ms")
    print(f"   ✓ Match: {np.isclose(thresholds[50]*1000, actual_p50)}")


def test_goodput_vs_throughput():
    """Demonstrate the difference between goodput and throughput."""
    print("\n" + "=" * 70)
    print("TEST 3: Goodput vs Throughput")
    print("=" * 70)
    
    requests = create_mock_requests(num_requests=100)
    program = ProgramMetrics(program_id="test_program_3", requests=requests)
    
    result = separate_requests_by_latency(program, percentiles=[50, 90, 99])
    
    print("\n📊 Understanding Goodput vs Throughput:")
    print("   • Throughput = all requests / total time")
    print("   • Goodput = valid requests (within SLA) / total time")
    
    # Get all metrics
    all_latencies = [r.latency for r in requests if r.latency > 0]
    all_tokens = sum(r.output_tokens for r in requests)
    total_time = sum(all_latencies)
    throughput = len(all_latencies) / total_time
    token_throughput = all_tokens / total_time
    
    print(f"\n   Overall Throughput: {throughput:.2f} req/s, {token_throughput:.2f} tok/s")
    
    for p in [50, 90, 99]:
        metrics = result["metrics_per_percentile"][p]
        goodput = metrics["request_goodput_rps"]
        token_goodput = metrics["output_token_goodput"]
        valid_pct = metrics["valid_request_percentage"]
        
        print(f"\n   With {p}th percentile SLA ({metrics['threshold_ms']:.0f}ms):")
        print(f"     └─ {valid_pct:.1f}% requests meet SLA")
        print(f"     └─ Goodput: {goodput:.2f} req/s, {token_goodput:.2f} tok/s")


def test_empty_and_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 70)
    print("TEST 4: Edge Cases")
    print("=" * 70)
    
    # Empty program
    empty_program = ProgramMetrics(program_id="empty", requests=[])
    result = separate_requests_by_latency(empty_program)
    print(f"\n✓ Empty program handled gracefully")
    print(f"   Result keys: {list(result.keys())}")
    
    # Single request
    single_request = [RequestMetrics(
        ttft=0.05, latency=0.5, itl=[], output_tokens=100,
        input_tokens=50, start_time=0.0, end_time=0.5
    )]
    single_program = ProgramMetrics(program_id="single", requests=single_request)
    result = separate_requests_by_latency(single_program, percentiles=[50, 99])
    print(f"\n✓ Single request program handled")
    print(f"   P50 threshold: {result['thresholds'][50]*1000:.2f}ms")


def test_bucket_contents():
    """Show what requests are in each bucket."""
    print("\n" + "=" * 70)
    print("TEST 5: Bucket Contents")
    print("=" * 70)
    
    requests = create_mock_requests(num_requests=10)
    program = ProgramMetrics(program_id="test_buckets", requests=requests)
    
    result = separate_requests_by_latency(program, percentiles=[50, 99])
    
    latencies_ms = [r.latency * 1000 for r in requests]
    print(f"\n📋 All request latencies (ms):")
    print(f"   {', '.join(f'{l:.1f}' for l in sorted(latencies_ms))}")
    
    for p in sorted(result["percentile_buckets"].keys()):
        bucket = result["percentile_buckets"][p]
        bucket_latencies = sorted([r.latency * 1000 for r in bucket])
        threshold = result['thresholds'][p] * 1000
        
        print(f"\n   P{p} bucket (threshold={threshold:.1f}ms):")
        print(f"   ├─ Count: {len(bucket)}/{len(requests)}")
        print(f"   └─ Latencies: {', '.join(f'{l:.1f}' for l in bucket_latencies)}")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Latency-Based Request Separation - Test Suite".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        test_basic_separation()
        test_percentile_calculation()
        test_goodput_vs_throughput()
        test_empty_and_edge_cases()
        test_bucket_contents()
        
        print("\n" + "=" * 70)
        print("✅ All tests completed successfully!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
