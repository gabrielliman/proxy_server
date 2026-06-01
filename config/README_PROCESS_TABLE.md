Process Table (Autellix integration)
=================================

Overview
--------
Autellix maintains a global process table that records runtime metrics for each active program (identified by a `program_id`). This table helps with scheduling, anti-starvation, load-balancing and debugging.

Location
--------
- Runtime implementation: `routing/process_table.py` (singleton `PROCESS_TABLE`)
- Monitoring endpoint: `GET /processes` (exposed by `api/endpoints_status.py`)

Process entry fields
--------------------
- `service_time_cumulative` : float — accumulated service time across completed calls for this program (seconds).
- `service_time_max` : float — largest single call service time observed (seconds).
- `waiting_time_cumulative` : float — accumulated waiting time observed across calls (seconds). Waiting time is computed from enqueue -> dequeue or arrival -> start when enqueue is not available.
- `engine_ids` : set[str] — the set of backend engine IDs (URLs) currently associated with running threads for this program.
- `threads` : dict(call_id -> metadata) — per-call metadata for active and completed LLM calls. Each thread record contains:
  - `arrival_time` : timestamp when the call arrived (seconds since epoch)
  - `enqueue_time` : timestamp when placed into the engine queue (may be None)
  - `dequeue_time` : timestamp when removed from the engine queue (may be None)
  - `start_time` : timestamp when service/processing started (may be None)
  - `completion_time` : timestamp when call completed (may be None)
  - `waiting_time` : float — time spent waiting (seconds)
  - `service_time` : float — time spent executing on the engine (seconds)
  - `engine_id` : engine/backend id assigned for the call (may be None)
  - `state` : string — one of `waiting`, `running`, `completed`.
- `most_recent_call_arrival` : timestamp of the last call arrival for this program.
- `most_recent_call_completion` : timestamp of the last completed call for this program.

How instrumentation is wired
---------------------------
- `api/endpoints_chat.py` and `api/endpoints_completion.py` record a call arrival using `PROCESS_TABLE.record_call_arrival(program_id, call_id)`; `call_id` is generated with `uuid4()` in the endpoints.
- Dispatchers add enqueue/dequeue events:
  - When queueing, dispatchers call `PROCESS_TABLE.record_enqueue(program_id, call_id)`.
  - When the worker or dispatcher takes the item from the queue, `PROCESS_TABLE.record_dequeue(program_id, call_id)` is called.
- When a call begins executing on an engine, the code calls `PROCESS_TABLE.record_call_start(program_id, call_id, engine_id=...)`.
- When a call completes, the code calls `PROCESS_TABLE.record_call_completion(program_id, call_id)`.

Monitoring and debugging
------------------------
- HTTP: GET `/processes` returns a snapshot of all processes and their threads. Example:

  curl http://localhost:8000/processes

- Python (in-process):

  from routing.process_table import PROCESS_TABLE
  p = PROCESS_TABLE.get_process("my-program-id")

- The process table is thread-safe but kept in memory. Long-lived or many distinct `program_id`s can increase memory usage; remove entries with `PROCESS_TABLE.remove_process(program_id)` when appropriate.

Notes and next steps
--------------------
- Currently the table is a best-effort, in-memory structure intended for monitoring and scheduling decisions. For persistence or cross-process visibility, export snapshots to metrics or a central store.
- Consider pruning stale programs based on `most_recent_call_arrival` to avoid unbounded growth.
