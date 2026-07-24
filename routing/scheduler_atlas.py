"""
ATLAS scheduler helper: compute per-call priority using the thread's inherited critical path.
"""
import itertools
from typing import Tuple

from routing.process_table import PROCESS_TABLE
from config.settings import PLAS_METRIC_TYPE

_SEQ = itertools.count()


def compute_priority(program_id: str, call_id: str) -> Tuple[float, int]:
    """Compute numeric priority for a (program_id, call_id).

    Strictly follows Autellix Eq. 2 (ATLAS): Priority is the maximum cumulative 
    service time across all threads in the same program (the inherited critical path).
    Returns (priority_value, seq) where smaller value is higher priority.
    """
    th_stats = PROCESS_TABLE.get_thread_stats(program_id, call_id) or {}
    
    # Select metric based on config
    if PLAS_METRIC_TYPE == "kv_token_time":
        priority_value = th_stats.get("inherited_kv_critical_path", 0.0)
    else:
        # Default to standard ATLAS (max critical path of service time)
        priority_value = th_stats.get("inherited_critical_path", 0.0)

    seq = next(_SEQ)
    return priority_value, seq


def update_on_completion(program_id: str, service_time: float, kv_token_time: float = None):
    """
    ProcessTable.record_call_completion handles the max() scalar update.
    """
    pass


def map_priority_to_queue_index(priority_value: float, num_buckets: int, base_span: float) -> int:
    """Map continuous ATLAS priority to a discrete queue index."""
    if priority_value < 0:
        priority_value = 0
    if base_span <= 0:
        base_span = 1.0
    
    bucket_width = base_span / num_buckets
    queue_idx = int(priority_value / bucket_width)
    return min(queue_idx, num_buckets - 1)