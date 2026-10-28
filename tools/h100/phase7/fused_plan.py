"""Fit an optional coordinated swizzle inside the frozen fused Phase 6 runtime.

Only logical-to-physical tile tables change. All three variants preserve the
exact selective mask. Historical BF16 samples are fitting evidence, not new
quantized calibration or a measured speedup. Pass 2 never selects a mapping.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

from fused_support import PHASE6, REPO

import scope_plan as legacy_scope
from ablation_plan import mask_sha, resolve_input
from common import read, require, sha, validate_entry
from fused_protocol import PROTOCOL, POLICIES, ORDERS, definition, digest, policy_id

_SPEC = importlib.util.spec_from_file_location("phase7_fused_arrival_parser", REPO / "tools/a800/arrival/plan.py")
_ARRIVAL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ARRIVAL)

GROUP_TILES = 8  # Predeclared; do not tune this using pass 2.
ASSUMPTIONS = [
    "Receiver-relative fragment observations are combined within each destination, never as absolute cross-GPU clocks.",
    "Different physical tiles were sampled on different invocations; their joined vector is not one invocation's joint timeline.",
    "A robust nonnegative linear slot slope approximates position cost; residual latency is assumed transferable after permutation.",
    "The fixed selective mask, codec and full-operator barrier are preserved; BF16 samples are not post-order/quantized recalibration.",
    "Predicted p95/max of tile joins are diagnostic surrogate objectives, not measured operator or training-step speedup.",
]


def _entry(source, maps):
    fields = ("shape", "M", "N", "K_global", "K_local", "tiles", "base_coords", "mask")
    result = {key: copy.deepcopy(source[key]) for key in fields}
    result["maps"] = copy.deepcopy(maps)
    result["coords"] = [[result["base_coords"][rank][index] for index in row] for rank, row in enumerate(maps)]
    result["selected_per_source"] = list(map(sum, result["mask"]))
    result["changed_per_rank"] = [sum(index != value for index, value in enumerate(row)) for row in maps]
    return result


def _inverse(coords):
    result = []
    for row in coords:
        inverse = [0] * len(row)
        for slot, tile in enumerate(row):
            inverse[tile] = slot
        result.append(inverse)
    return result


def _p95(values):
    require(bool(values), "Cannot evaluate an empty diagnostic")
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def _slope(pairs):
    """Theil-Sen median slope, constrained nonnegative; no held-out fitting."""
    pairs = sorted(pairs)
    values = [(right_y - left_y) / (right_x - left_x)
              for i, (left_x, left_y) in enumerate(pairs)
              for right_x, right_y in pairs[i+1:] if right_x != left_x]
    return max(0.0, statistics.median(values)) if values else 0.0


def _validate_observations(observations, world, tiles):
    require(isinstance(observations, dict) and set(observations) <= set(range(world)), "Unknown observation destination")
    per = tiles // world
    for destination, rows in observations.items():
        require(type(destination) is int and isinstance(rows, dict), "Invalid observation destination")
        for key, item in rows.items():
            require(isinstance(key, tuple) and len(key) == 2, "Expected (pass,tile) observation key")
            passed, tile = key
            require(type(passed) is int and passed in (0, 1, 2) and type(tile) is int
                    and destination*per <= tile < (destination+1)*per, "Observation coordinate outside destination/pass")
            require(isinstance(item, (list, tuple)) and len(item) == 2, "Expected arrival vector and uncertainty")
            arrivals, uncertainty = item
            require(isinstance(arrivals, (list, tuple)) and len(arrivals) == world
                    and all(type(x) in (int, float) and math.isfinite(x) for x in arrivals), "Missing/nonfinite source arrivals")
            require(type(uncertainty) in (int, float) and math.isfinite(uncertainty) and uncertainty > 0,
                    "Arrival uncertainty must be finite and positive")


def _metrics(destination, passed, observations, base_positions, positions, slopes, per):
    joins = []
    rows = observations[destination]
    for tile in range(destination*per, (destination+1)*per):
        arrivals = rows[(passed, tile)][0]
        predicted = [value + slope * (positions[source][tile] - base_positions[source][tile])
                     for source, (value, slope) in enumerate(zip(arrivals, slopes))]
        joins.append(max(predicted))
    return dict(surrogate_p95_join_us=_p95(joins), surrogate_max_join_us=max(joins),
                surrogate_mean_join_us=statistics.mean(joins))


def _admissible(candidate, legacy, minimum_gain_us):
    # Never accept a p95 improvement obtained by worsening either pass's maximum.
    keys = ("surrogate_p95_join_us", "surrogate_max_join_us")
    eps = 1e-9
    return (all(candidate[p][key] <= legacy[p][key] + eps for p in (0, 1) for key in keys)
            and any(candidate[p][key] < legacy[p][key] - max(eps, minimum_gain_us) for p in (0, 1) for key in keys))


def fit_maps(entry, config, observations):
    """Pure CPU fit returning (maps, diagnostics); caller supplies evidence kind.

    A destination with incomplete training data falls back to its exact legacy
    schedule. Each candidate is checked against legacy, not identity, on both
    training passes. Pass 2 is read only after the mapping has been selected.
    """
    validate_entry(entry, config, entry["shape"])
    world, tiles = config["tp"], entry["tiles"]
    per = tiles // world
    require(world == 8 and per == 64 and config["window"] == 64, "Only frozen TP8/64-slot destination windows supported")
    _validate_observations(observations, world, tiles)
    base_positions = _inverse(entry["base_coords"])
    legacy_positions = _inverse(entry["coords"])
    maps = copy.deepcopy(entry["maps"])
    diagnostics = dict(training_passes=[0, 1], heldout_passes=[2], group_tiles=GROUP_TILES,
                       assumptions=list(ASSUMPTIONS), not_recalibrated=True,
                       measured_performance=False, destinations=[])
    for destination in range(world):
        physical = list(range(destination*per, (destination+1)*per))
        rows = observations.get(destination, {})
        missing = [[passed, tile] for passed in (0, 1) for tile in physical if (passed, tile) not in rows]
        diagnostic = dict(destination=destination, selected="legacy", missing_training=len(missing),
                          reason="incomplete_training_fallback", train_metrics={}, heldout_metrics=None)
        diagnostics["destinations"].append(diagnostic)
        if missing:
            continue
        slopes = [[_slope([(base_positions[source][tile], rows[(passed, tile)][0][source]) for tile in physical])
                   for source in range(world)] for passed in (0, 1)]
        residuals = [[[rows[(passed, tile)][0][source] - slopes[passed][source]*base_positions[source][tile]
                       for tile in physical] for source in range(world)] for passed in (0, 1)]
        # Explicitly penalize two-pass variation and the larger polling error.
        # This is a lower-residual heuristic, not a statistical confidence bound.
        stable = [[statistics.mean(residuals[passed][source][offset] for passed in (0, 1))
                   - abs(residuals[0][source][offset] - residuals[1][source][offset])/2
                   - max(rows[(passed, tile)][1] for passed in (0, 1))
                   for offset, tile in enumerate(physical)] for source in range(world)]
        # Round only below measurement resolution so floating cancellation does
        # not permute equal priorities when the receiver-relative origin shifts.
        shared = sorted(physical, key=lambda tile: (-round(max(stable[source][tile-destination*per] for source in range(world)), 9), tile))
        candidates = {}
        for name in ("shared_residual", "coordinated_residual"):
            candidate = copy.deepcopy(entry["maps"])
            for source in range(world):
                order = list(shared)
                if name == "coordinated_residual":
                    order = [tile for start in range(0, per, GROUP_TILES)
                             for tile in sorted(shared[start:start+GROUP_TILES],
                                                key=lambda tile: (-round(stable[source][tile-destination*per], 9), tile))]
                slots = sorted(base_positions[source][tile] for tile in physical)
                for slot, tile in zip(slots, order):
                    candidate[source][slot] = base_positions[source][tile]
            candidates[name] = candidate
        legacy_metrics = [_metrics(destination, passed, observations, base_positions, legacy_positions, slopes[passed], per)
                          for passed in (0, 1)]
        candidate_metrics = {}
        admissible = []
        minimum_gain = statistics.median(rows[(passed, tile)][1] for passed in (0, 1) for tile in physical)
        for name, candidate in candidates.items():
            positions = _inverse([[entry["base_coords"][source][index] for index in row]
                                  for source, row in enumerate(candidate)])
            metrics = [_metrics(destination, passed, observations, base_positions, positions, slopes[passed], per)
                       for passed in (0, 1)]
            candidate_metrics[name] = metrics
            if _admissible(metrics, legacy_metrics, minimum_gain):
                admissible.append(name)
        if admissible:
            selected = min(admissible, key=lambda name: (round(sum(row["surrogate_p95_join_us"] for row in candidate_metrics[name]), 9),
                                                        round(sum(row["surrogate_max_join_us"] for row in candidate_metrics[name]), 9), name))
            for source in range(world):
                for tile in physical:
                    slot = base_positions[source][tile]
                    maps[source][slot] = candidates[selected][source][slot]
            diagnostic.update(selected=selected, reason="both_training_passes_nonworse_p95_and_max_with_strict_model_gain")
        else:
            diagnostic["reason"] = "no_candidate_beats_legacy_on_both_training_passes"
        diagnostic["slopes_us_per_slot"] = slopes
        diagnostic["minimum_surrogate_gain_us"] = minimum_gain
        diagnostic["priority_rule"] = "mean residual - half absolute two-pass difference - maximum polling uncertainty"
        diagnostic["shared_physical_order"] = shared
        diagnostic["train_metrics"] = dict(legacy=legacy_metrics, **candidate_metrics)
        # Freeze the destination choice before touching pass 2. Use training
        # slopes; heldout timestamps cannot refit a model or reject the choice.
        if all((2, tile) in rows for tile in physical):
            held_slopes = [statistics.mean(slopes[p][source] for p in (0, 1)) for source in range(world)]
            final_positions = _inverse([[entry["base_coords"][source][index] for index in row]
                                        for source, row in enumerate(maps)])
            diagnostic["heldout_metrics"] = dict(
                legacy=_metrics(destination, 2, observations, base_positions, legacy_positions, held_slopes, per),
                improved=_metrics(destination, 2, observations, base_positions, final_positions, held_slopes, per),
                used_for_selection=False)
    improved = _entry(entry, maps)
    validate_entry(improved, config, entry["shape"])
    diagnostics["changed_vs_legacy_per_source"] = [sum(a != b for a, b in zip(left, right))
                                                  for left, right in zip(maps, entry["maps"])]
    diagnostics["accepted_destinations"] = sum(row["selected"] != "legacy" for row in diagnostics["destinations"])
    return maps, diagnostics


def load_observations(record):
    """Load the exact hashed per-rank reports named by the original role record."""
    require(not record.get("test_only") and record.get("gpu_calibrated"), "Real frozen calibration records required")
    records = {}
    provenance = {}
    for name, expected_sha in record["inputs"].items():
        if re.fullmatch(r"b0-interleaved-instrumented-rank[0-7]\.json", Path(name).name) is None:
            continue
        path = resolve_input(name, expected_sha)
        report = read(path)
        rank = report.get("rank")
        require(type(rank) is int and 0 <= rank < 8 and rank not in records, "Duplicate/missing calibration receiver rank")
        require(report.get("completed") and not report.get("smoke") and report.get("policy") == "interleaved"
                and report.get("world_size") == 8, "Incomplete/wrong calibration report")
        require(legacy_scope.hardware_identity(report["hardware"]) == legacy_scope.hardware_identity(record["hardware"]),
                "Calibration report hardware disagrees with frozen role")
        require(report["hardware"].get("local_rank") == rank, "Calibration rank/device order differs")
        matches = [shape for shape in report["shapes"]
                   if [shape["M"], shape["N"], shape["K_global"]] == record["entry"]["shape"]]
        require(len(matches) == 1, "Missing/ambiguous exact role shape in raw report")
        records[rank] = (report, matches[0])
        provenance[name] = dict(resolved=str(path), sha256=expected_sha)
    require(set(records) == set(range(8)), "Original per-rank receiver samples are required for all eight ranks")
    require(len({report["run_id"] for report, _ in records.values()}) == 1, "Calibration reports mix run IDs")
    observations = {rank: _ARRIVAL.observations(shape, rank, 8, relative=True)
                    for rank, (_, shape) in records.items()}
    return observations, provenance


def _derived(role, entry, source):
    return dict(role=role, entry=entry, entry_sha256=digest(entry), mask_sha256=mask_sha(entry["mask"]),
                source_entry_sha256=source["entry_sha256"], not_recalibrated=True,
                measured_performance=False)


def bundle(attn_path=None, mlp_path=None, source_only=False):
    """Return nine runtime-compatible variants, without altering historical files."""
    parent = legacy_scope.bundle(attn_path, mlp_path, source_only)
    role_entries = {}
    diagnostics = {}
    inputs = {str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
    for path in (attn_path, mlp_path):
        if path is not None:
            inputs[str(Path(path).resolve())] = sha(path)
    for role, record in parent["roles"].items():
        for name, value in record["inputs"].items():
            require(name not in inputs or inputs[name] == value, "Conflicting frozen evidence hashes")
            inputs[name] = value
        source = record["entry"]
        if source_only:
            maps, diagnostic = fit_maps(source, record["config"], {})
            diagnostic.update(evidence_kind="source_only_no_arrival_fit", calibration_inputs={})
        else:
            observations, provenance = load_observations(record)
            maps, diagnostic = fit_maps(source, record["config"], observations)
            diagnostic.update(evidence_kind="historical_bf16_receiver_samples", calibration_inputs=provenance)
        role_entries[role] = dict(base=_entry(source, [list(range(source["tiles"])) for _ in range(8)]),
                                  legacy=_entry(source, source["maps"]), improved=_entry(source, maps))
        diagnostics[role] = diagnostic
    variants = {}
    for policy in POLICIES:
        defined = definition(policy)
        variants[policy] = dict(definition=defined, roles={role: _derived(role, role_entries[role][defined["order"]], parent["roles"][role])
                                                         for role in defined["roles"]})
    result = dict(protocol=PROTOCOL, parent=parent, source_only=bool(source_only),
                  gpu_calibrated=parent["gpu_calibrated"], variants=variants, diagnostics=diagnostics,
                  inputs=inputs, status="source_only" if source_only else "pending_fused_gpu_validation",
                  not_recalibrated=True, measured_performance=False)
    validate(result)
    return result


def order_entries(plans, order):
    require(order in ORDERS, "Unknown fused order")
    return {role: plans["variants"][policy_id(order, role)]["roles"][role]["entry"] for role in ("attn", "mlp")}


def validate(plans, verify_evidence=True):
    require(plans["protocol"] == PROTOCOL and set(plans["variants"]) == set(POLICIES), "Wrong/missing fused variants")
    require(type(plans["source_only"]) is bool and type(plans["gpu_calibrated"]) is bool, "Invalid evidence flags")
    require(plans["not_recalibrated"] is True and plans["measured_performance"] is False,
            "Static fitted tables cannot claim recalibration/performance validation")
    parent = plans["parent"]
    require(plans["source_only"] == parent["source_only"] and plans["gpu_calibrated"] == parent["gpu_calibrated"],
            "Parent calibration provenance changed")
    require(not plans["source_only"] or not plans["gpu_calibrated"], "Source-only cannot be GPU-calibrated")
    if verify_evidence:
        for name, value in plans["inputs"].items():
            resolve_input(name, value)
    require(set(plans["diagnostics"]) == {"attn", "mlp"}, "Missing role diagnostics")
    for role in ("attn", "mlp"):
        legacy_scope.validate(parent["roles"][role], role, allow_test=plans["source_only"], verify_evidence=verify_evidence)
    for policy, variant in plans["variants"].items():
        defined = definition(policy)
        require(variant["definition"] == defined and set(variant["roles"]) == set(defined["roles"]), "Fused definition/role set changed")
        for role, derived in variant["roles"].items():
            source = parent["roles"][role]
            entry = derived["entry"]
            require(derived["role"] == role and derived["source_entry_sha256"] == source["entry_sha256"], "Derived role origin differs")
            require(derived["not_recalibrated"] is True and derived["measured_performance"] is False,
                    "Derived tables cannot be marked as remeasured")
            require(derived["entry_sha256"] == digest(entry) and derived["mask_sha256"] == mask_sha(entry["mask"]), "Derived plan hash differs")
            require(entry["mask"] == source["entry"]["mask"] and entry["base_coords"] == source["entry"]["base_coords"],
                    "Fused change may not alter physical mask or interleaved base")
            validate_entry(entry, source["config"], source["entry"]["shape"])
            require(entry["selected_per_source"] == list(map(sum, entry["mask"])), "Mask counts changed")
            if defined["order"] == "base":
                require(all(row == list(range(entry["tiles"])) for row in entry["maps"]), "Base must disable only arrival lookup")
            if defined["order"] == "legacy" or plans["source_only"] and defined["order"] == "improved":
                require(entry["maps"] == source["entry"]["maps"], "Legacy/source-only lookup differs")
            reference = order_entries(plans, defined["order"])[role]
            require(entry == reference, "A role's lookup differs across scopes")
    return plans


def verify_fit(plans):
    """Expensive provenance audit: reconstruct the table from hashed raw inputs.

    Ordinary ``validate`` checks frozen hashes and structure; this separate
    function proves the improved table/diagnostics were produced by the stated
    fitting algorithm. Run it when freezing/building sources, not per inference.
    """
    validate(plans, verify_evidence=True)
    for role, record in plans["parent"]["roles"].items():
        observations = {} if plans["source_only"] else load_observations(record)[0]
        maps, diagnostic = fit_maps(record["entry"], record["config"], observations)
        require(order_entries(plans, "improved")[role] == _entry(record["entry"], maps),
                f"{role} improved table differs from reconstruction using frozen receiver evidence")
        for key, value in diagnostic.items():
            require(plans["diagnostics"][role].get(key) == value, f"{role} fitting diagnostic differs: {key}")
    return dict(protocol=PROTOCOL, reconstructed=True, not_recalibrated=True, measured_performance=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--attn", type=Path)
    parser.add_argument("--mlp", type=Path)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    result = bundle(args.attn, args.mlp, args.source_only)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(args.out)


if __name__ == "__main__":
    main()
