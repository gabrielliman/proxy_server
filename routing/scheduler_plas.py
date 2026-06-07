"""
PLAS scheduler helper: compute per-call priority using per-program EWMA and waiting-age.

This module exposes a simple API:
- compute_priority(program_id, call_id) -> (priority_value, seq)
- update_on_completion(program_id, service_time)

Priority semantics: lower numeric value -> higher scheduler priority (matches PriorityQueue).
"""
import time
import itertools
from typing import Tuple

from routing.process_table import PROCESS_TABLE
from config.settings import PLAS_EWMA_ALPHA, PLAS_AGE_WEIGHT, PLAS_PRIORITY_BASE, PLAS_METRIC_TYPE

_SEQ = itertools.count()


def compute_priority(program_id: str, call_id: str) -> Tuple[float, int]:
    """Compute numeric priority for a (program_id, call_id).

    Uses per-program EWMA based on PLAS_METRIC_TYPE:
    - 'service_ewma': EWMA of actual service time (LAS - Least Attained Service)
    - 'kv_token_time': EWMA of KV token-time (d*c = pd + d²/2)
    
    Also factors in waiting age (time since enqueue/arrival).
    Returns (priority_value, seq) where smaller value is higher priority.
    """
    now = time.time()

    stats = PROCESS_TABLE.get_program_stats(program_id) or {}
    
    # Select metric based on PLAS_METRIC_TYPE config
    if PLAS_METRIC_TYPE == "kv_token_time":
        base_metric = stats.get("kv_token_time_ewma")
    else:
        # Default to service_ewma (LAS)
        base_metric = stats.get("service_ewma")

    if base_metric is None:
        # fallback to base
        base = PLAS_PRIORITY_BASE
    else:
        base = float(base_metric)

    # derive age: use most_recent_call_arrival as conservative proxy
    last_arrival = stats.get("most_recent_call_arrival")
    age = max(0.0, now - last_arrival) if last_arrival else 0.0

    # priority formula: prefer smaller base (shorter expected service) and increase priority by age
    # numeric: priority = base - age * age_weight
    priority_value = max(0.0, base - age * PLAS_AGE_WEIGHT)
    seq = next(_SEQ)
    return priority_value, seq


def update_on_completion(program_id: str, service_time: float, kv_token_time: float = None):
    """Update per-program EWMA in the PROCESS_TABLE using config alpha.
    
    Args:
        program_id: The program identifier
        service_time: Actual service time for this call
        kv_token_time: KV token-time for this call (d*c = pd + d²/2), optional
    """
    alpha = PLAS_EWMA_ALPHA
    # read current stats and update under the process table lock
    stats = PROCESS_TABLE.get_program_stats(program_id)
    if stats is None:
        # ensure process exists and set ewma
        PROCESS_TABLE.ensure_process(program_id)
        with PROCESS_TABLE.lock:
            PROCESS_TABLE.table[program_id]["service_ewma"] = service_time
            PROCESS_TABLE.table[program_id]["call_count"] = PROCESS_TABLE.table[program_id].get("call_count", 0) + 1
            if kv_token_time is not None:
                PROCESS_TABLE.table[program_id]["kv_token_time_ewma"] = kv_token_time
        return

    # apply EWMA update for service_ewma
    with PROCESS_TABLE.lock:
        entry = PROCESS_TABLE.table.get(program_id)
        if entry is None:
            return
        ewma = entry.get("service_ewma")
        if ewma is None:
            entry["service_ewma"] = service_time
        else:
            entry["service_ewma"] = alpha * service_time + (1 - alpha) * ewma
        
        # Also update kv_token_time_ewma if provided
        if kv_token_time is not None:
            kv_ewma = entry.get("kv_token_time_ewma")
            if kv_ewma is None:
                entry["kv_token_time_ewma"] = kv_token_time
            else:
                entry["kv_token_time_ewma"] = alpha * kv_token_time + (1 - alpha) * kv_ewma


def map_priority_to_queue_index(priority_value: float, num_buckets: int, base_span: float) -> int:
    """Map continuous PLAS priority to a discrete queue index.
    
    Priority ranges are equal-width from [0, base_span) divided into num_buckets.
    Index 0 = Q1 (highest priority), index num_buckets-1 = QK (lowest priority).
    Lower numeric priority_value -> lower index (higher priority).
    """
    if priority_value < 0:
        priority_value = 0
    if base_span <= 0:
        base_span = 1.0
    
    bucket_width = base_span / num_buckets
    queue_idx = int(priority_value / bucket_width)
    return min(queue_idx, num_buckets - 1)  # clamp to max queue index
