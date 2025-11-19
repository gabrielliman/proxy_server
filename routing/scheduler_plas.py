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
from config.settings import PLAS_EWMA_ALPHA, PLAS_AGE_WEIGHT, PLAS_PRIORITY_BASE

_SEQ = itertools.count()


def compute_priority(program_id: str, call_id: str) -> Tuple[float, int]:
    """Compute numeric priority for a (program_id, call_id).

    Uses per-program EWMA service time and current age (time since enqueue/arrival).
    Returns (priority_value, seq) where smaller value is higher priority.
    """
    now = time.time()

    stats = PROCESS_TABLE.get_program_stats(program_id) or {}
    ewma = stats.get("service_ewma")
    if ewma is None:
        # fallback to base
        base = PLAS_PRIORITY_BASE
    else:
        base = float(ewma)

    # derive age: use most_recent_call_arrival as conservative proxy
    last_arrival = stats.get("most_recent_call_arrival")
    age = max(0.0, now - last_arrival) if last_arrival else 0.0

    # priority formula: prefer smaller base (shorter expected service) and increase priority by age
    # numeric: priority = base - age * age_weight
    priority_value = max(0.0, base - age * PLAS_AGE_WEIGHT)
    seq = next(_SEQ)
    return priority_value, seq


def update_on_completion(program_id: str, service_time: float):
    """Update per-program EWMA in the PROCESS_TABLE using config alpha."""
    alpha = PLAS_EWMA_ALPHA
    # read current stats and update under the process table lock
    stats = PROCESS_TABLE.get_program_stats(program_id)
    if stats is None:
        # ensure process exists and set ewma
        PROCESS_TABLE.ensure_process(program_id)
        with PROCESS_TABLE.lock:
            PROCESS_TABLE.table[program_id]["service_ewma"] = service_time
            PROCESS_TABLE.table[program_id]["call_count"] = PROCESS_TABLE.table[program_id].get("call_count", 0) + 1
        return

    # apply EWMA update (we already incremented call_count in record_call_completion)
    with PROCESS_TABLE.lock:
        entry = PROCESS_TABLE.table.get(program_id)
        if entry is None:
            return
        ewma = entry.get("service_ewma")
        if ewma is None:
            entry["service_ewma"] = service_time
        else:
            entry["service_ewma"] = alpha * service_time + (1 - alpha) * ewma
