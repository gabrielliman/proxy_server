from prometheus_client.parser import text_string_to_metric_families


def _first_value(samples):
    """Safely get first sample value from iterable."""
    try:
        return next(iter(samples)).value
    except StopIteration:
        return 0


def parse_prometheus_metrics(raw_text):
    metrics_data = {}

    for family in text_string_to_metric_families(raw_text):
        # IMPORTANT: keep samples, don't collapse labels
        metrics_data[family.name] = family.samples

    running = int(_first_value(metrics_data.get("vllm:num_requests_running", [])))
    waiting = int(_first_value(metrics_data.get("vllm:num_requests_waiting", [])))

    kv_samples = metrics_data.get("vllm:kv_cache_usage_perc", [])

    kv_list = []
    for s in kv_samples:
        value = s.value * 100 if s.value <= 1 else s.value

        kv_list.append({
            "engine": s.labels.get("engine"),
            "model_name": s.labels.get("model_name"),
            "kv_cache": value,
            "running": running,
            "waiting": waiting
        })

    return {
        "running": running,
        "waiting": waiting,
        "kv_cache": kv_list,  # ← now supports multiple engines safely
    }
