import os
import re
import json
import argparse
import pandas as pd

# =========================
# Regex metrics (CSV)
# =========================
RE_QUERIES = re.compile(r"prefix_cache_queries_total.*?\s([0-9.e+]+)")
RE_HITS = re.compile(r"prefix_cache_hits_total.*?\s([0-9.e+]+)")

# =========================
# Regex filename (JSON)
# =========================
RE_JSON = re.compile(r"output_(chat|completion)_(\d+)_run(\d+)\.json")

# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build benchmark dataset from vLLM prefix metrics + JSON outputs"
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="CSV with raw prefix cache metrics (cumulative)"
    )
    parser.add_argument(
        "--outputs-dir",
        required=True,
        help="Directory containing output_{mode}_{len}_run{n}.json files"
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to save the final merged dataset CSV"
    )
    return parser.parse_args()

# =========================
# Main
# =========================
def main():
    args = parse_args()

    rows = []

    # =========================
    # 1. Parse CSV manually
    # =========================
    with open(args.input_csv, "r", encoding="utf-8") as f:
        next(f)  # skip header

        for line in f:
            parts = line.strip().split(",", 3)
            if len(parts) != 4:
                continue

            mode = parts[0]
            max_chat_len = int(parts[1])
            run = int(parts[2])
            metrics = parts[3]

            q = RE_QUERIES.search(metrics)
            h = RE_HITS.search(metrics)

            rows.append({
                "mode": mode,
                "max_chat_len": max_chat_len,
                "run": run,
                "queries_total": float(q.group(1)) if q else None,
                "hits_total": float(h.group(1)) if h else None,
            })

    df = pd.DataFrame(rows).reset_index(drop=True)

    # =========================
    # 2. Global diff (cumulativo real)
    # =========================
    df["queries_real"] = df["queries_total"].diff()
    df["hits_real"] = df["hits_total"].diff()

    df.loc[0, "queries_real"] = df.loc[0, "queries_total"]
    df.loc[0, "hits_real"] = df.loc[0, "hits_total"]

    df["queries_real"] = df["queries_real"].clip(lower=0)
    df["hits_real"] = df["hits_real"].clip(lower=0)

    # =========================
    # 3. Load JSON benchmark files
    # =========================
    json_metrics = {}

    for fname in os.listdir(args.outputs_dir):
        m = RE_JSON.match(fname)
        if not m:
            continue

        mode, max_chat_len, run = m.group(1), int(m.group(2)), int(m.group(3))
        path = os.path.join(args.outputs_dir, fname)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fm = data.get("full_metrics", {})

        json_metrics[(mode, max_chat_len, run)] = {
            "num_requests": fm.get("num_requests"),
            "total_input_tokens": fm.get("total_input_tokens"),
            "total_output_tokens": fm.get("total_output_tokens"),
            "request_throughput_rps": fm.get("request_throughput_rps"),
            "output_token_throughput": fm.get("output_token_throughput"),
            "total_token_throughput": fm.get("total_token_throughput"),
            "mean_tpot_ms": fm.get("mean_tpot_ms"),
            "mean_itl_ms": fm.get("mean_itl_ms"),
            "mean_e2el_ms": fm.get("mean_e2el_ms"),
            "benchmark_wall_time_s": fm.get("benchmark_wall_time_s"),
        }

    # =========================
    # 4. Join JSON → CSV
    # =========================
    json_df = df.apply(
        lambda r: json_metrics.get(
            (r["mode"], r["max_chat_len"], r["run"]), {}
        ),
        axis=1,
        result_type="expand"
    )

    df_final = pd.concat([df, json_df], axis=1)

    # =========================
    # 5. Save
    # =========================
    df_final.to_csv(args.output_csv, index=False)
    print(f"✅ Dataset final gerado: {args.output_csv}")

# =========================
if __name__ == "__main__":
    main()
