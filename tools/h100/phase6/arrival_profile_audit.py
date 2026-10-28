"""Read-only audit of frozen ablation-4 arrival mappings and CUDA traces.

Profile durations are diagnostic: profiler overhead prevents treating them as
unprofiled optimizer-step timings or as an estimate of attainable speedup.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics


POLICIES = {
    "A": "interleaved_all_attn_mlp",
    "B": "interleaved_arrival_all_attn_mlp",
}
ROLES = ("attn", "mlp")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mapping_audit(compare, role):
    tables = {}
    paths = {}
    for label, policy in POLICIES.items():
        path = compare / "results" / f"mapping-{policy}-{role}.csv"
        with path.open() as handle:
            rows = [{k: int(v) for k, v in row.items()}
                    for row in csv.DictReader(handle)]
        table = {(row["rank"], row["index"]): row for row in rows}
        require(len(table) == len(rows) and rows, f"Duplicate/empty mapping: {path}")
        tables[label], paths[label] = table, str(path)
    a, b = tables["A"], tables["B"]
    require(a.keys() == b.keys(), f"A/B logical slots differ: {role}")
    ranks = sorted({key[0] for key in a})
    require(ranks == list(range(8)), f"Expected eight TP ranks: {role}")
    changed = {}
    for rank in ranks:
        keys = sorted(key for key in a if key[0] == rank)
        require([key[1] for key in keys] == list(range(len(keys))),
                f"Missing mapping slots: {role}, rank {rank}")
        ca = [(a[key]["m"], a[key]["n"]) for key in keys]
        cb = [(b[key]["m"], b[key]["n"]) for key in keys]
        require(len(set(ca)) == len(ca) and sorted(ca) == sorted(cb),
                f"A/B physical tile sets differ: {role}, rank {rank}")
        changed[str(rank)] = sum(x != y for x, y in zip(ca, cb))
    destination_changes = sum(a[key]["destination"] != b[key]["destination"]
                              for key in a)
    total_changed = sum(changed.values())
    return {
        "files": paths,
        "logical_slots": len(a),
        "changed_coordinates": total_changed,
        "changed_fraction": total_changed / len(a),
        "changed_per_rank": changed,
        "changed_destinations": destination_changes,
        "destination_sequence_preserved": destination_changes == 0,
    }


def trace_projection_records(path):
    """Associate CPU scopes and GPU kernels by the profiler's External id."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    scopes = defaultdict(list)
    kernels = defaultdict(list)
    calls = []
    for event in events:
        category, name = event.get("cat"), event.get("name", "")
        if category == "user_annotation" and name.startswith("scope."):
            if name.split(".")[1] in ROLES:
                scopes[event["tid"]].append(event)
        elif category == "cpu_op" and name == "GemmRSFunction":
            calls.append(event)
        elif category == "kernel":
            kernels[event.get("args", {}).get("External id")].append(event)
    for tid in scopes:
        scopes[tid].sort(key=lambda event: event["ts"])
    starts = {tid: [event["ts"] for event in group]
              for tid, group in scopes.items()}
    records = defaultdict(list)
    used_scopes = set()
    for call in calls:
        tid = call["tid"]
        index = bisect_right(starts.get(tid, []), call["ts"]) - 1
        require(index >= 0, f"GemmRSFunction has no projection scope: {path}")
        scope = scopes[tid][index]
        require(call["ts"] + call["dur"] <= scope["ts"] + scope["dur"] + 0.01,
                f"Projection scope does not contain GemmRSFunction: {path}")
        require((tid, index) not in used_scopes, f"Repeated projection scope: {path}")
        used_scopes.add((tid, index))
        role = scope["name"].split(".")[1]
        group = sorted(kernels[call["args"]["External id"]],
                       key=lambda event: event["ts"])
        gemm = [event for event in group if "flux_bf16" in event["name"]
                and "reducescatter" in event["name"]]
        decode = [event for event in group if "taco_decode" in event["name"]]
        barriers = [event for event in group if "CudaIpcBarrierAllKernel" in event["name"]]
        require(len(gemm) == len(decode) == 1 and len(barriers) == 2,
                f"Expected one GEMM, one decode, two barriers: {path}, {role}")
        gemm, decode = gemm[0], decode[0]
        first, second = barriers
        selected = (gemm, first, decode, second)
        ordered = all(left["ts"] + left["dur"] <= right["ts"] + 0.001
                      for left, right in zip(selected, selected[1:]))
        overlap = (max(gemm["ts"], decode["ts"]) <
                   min(gemm["ts"] + gemm["dur"], decode["ts"] + decode["dur"]))
        records[role].append({
            "gemm_us": gemm["dur"],
            "pre_decode_barrier_us": first["dur"],
            "decode_us": decode["dur"],
            "post_decode_barrier_us": second["dur"],
            "gemm_end_to_decode_start_us": decode["ts"] - gemm["ts"] - gemm["dur"],
            "same_stream": len({event["args"]["stream"] for event in selected}) == 1,
            "gemm_barrier_decode_barrier_serial": ordered,
            "gemm_decode_overlap": overlap,
        })
    require(len(used_scopes) == sum(map(len, scopes.values())) and used_scopes,
            f"Missing GPU/CPU association for projection scope: {path}")
    require(set(records) == set(ROLES), f"Missing projection role: {path}")
    return records


def summarize(records):
    duration_keys = [key for key in records[0] if key.endswith("_us")]
    return {
        "projection_calls": len(records),
        "same_stream_calls": sum(row["same_stream"] for row in records),
        "gemm_barrier_decode_barrier_serial_calls": sum(
            row["gemm_barrier_decode_barrier_serial"] for row in records),
        "gemm_decode_overlap_calls": sum(row["gemm_decode_overlap"] for row in records),
        "mean_us": {key: statistics.mean(row[key] for row in records)
                    for key in duration_keys},
        "median_us": {key: statistics.median(row[key] for row in records)
                      for key in duration_keys},
    }


def audit(compare, ranks):
    compare = compare.resolve()
    result = {"compare": str(compare),
              "mappings": {role: mapping_audit(compare, role) for role in ROLES},
              "profiles": {}}
    for label, policy in POLICIES.items():
        combined = defaultdict(list)
        rank_summaries = {}
        for rank in ranks:
            path = compare / "results" / f"profile-{policy}" / f"rank{rank}-profile.json.gz"
            records = trace_projection_records(path)
            rank_summaries[str(rank)] = {
                "file": str(path), "roles": {role: summarize(rows)
                                               for role, rows in records.items()}}
            for role, rows in records.items():
                combined[role].extend(rows)
        result["profiles"][label] = {
            "policy": policy, "ranks": rank_summaries,
            "combined_roles": {role: summarize(rows) for role, rows in combined.items()},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compare", type=Path, nargs="+")
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--out", type=Path, help="New JSON file; refuses to overwrite")
    args = parser.parse_args()
    require(len(set(args.ranks)) == len(args.ranks) and all(0 <= rank < 8 for rank in args.ranks),
            "Ranks must be unique values from 0 through 7")
    require(len({path.resolve() for path in args.compare}) == len(args.compare),
            "Duplicate compare directory")
    require(args.out is None or not args.out.exists(), "Refusing to overwrite output")
    result = {
        "schema": "h100-ablation4-arrival-profile-audit-v1",
        "limitation": "Profile durations include profiler perturbation. They do not estimate "
                      "unprofiled optimizer-step performance or attainable speedup. "
                      "The sequence audit covers the specified traces and ranks only.",
        "compares": [audit(path, args.ranks) for path in args.compare],
    }
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.out is None:
        print(encoded, end="")
    else:
        with args.out.open("x") as handle:
            handle.write(encoded)


if __name__ == "__main__":
    main()
