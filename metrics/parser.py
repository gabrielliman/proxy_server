from prometheus_client.parser import text_string_to_metric_families

def parse_prometheus_metrics(raw_text):
    metrics_data = {}
    for family in text_string_to_metric_families(raw_text):
        metrics_data[family.name] = {s.name: s.value for s in family.samples}

    running = int(metrics_data.get("vllm:num_requests_running", {}).get("vllm:num_requests_running", 0))
    waiting = int(metrics_data.get("vllm:num_requests_waiting", {}).get("vllm:num_requests_waiting", 0))
    raw_kv = float(metrics_data.get("vllm:kv_cache_usage_perc", {}).get("vllm:kv_cache_usage_perc", 0.0))

    kv_cache = raw_kv * 100 if raw_kv <= 1 else raw_kv

    return {
        "running": running,
        "waiting": waiting,
        "kv_cache": kv_cache
    }
