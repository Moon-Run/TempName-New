"""Offline, bounded joint assignment for the receiver-local tail surrogate.

Unlike v1's individually improving swaps, a complete assignment may move several
tied tails together. The unconfigured v2 path retains its historical 6*error
gate exactly. Explicit v3 options allow a frozen screening margin of 1..6 times
the original error proxy. Margins below 4*error are exploratory, not a bounded
observation-error guarantee. Neither mode establishes measured GPU speedup.
"""
import math

from arrival_tail import (HELDOUT_PASS, TRAINING_PASSES, UNCERTAINTY_MULTIPLIER,
                          _mapping_rows, _predicted, _samples)


METHOD = 'receiver-tail-assignment-v2'
CONFIGURED_METHOD = 'receiver-tail-assignment-v3'
CONDITIONAL_ERROR_MULTIPLIER = 4.0
MAX_DISPLACEMENT = 8
STRICT_TOLERANCE_US = 1e-12


def normalize_options(options):
    """None means exact historical v2; explicit options select the v3 schema."""
    if options is None:
        return None
    if not isinstance(options, dict) or set(options) != {'uncertainty_multiplier'}:
        raise ValueError('arrival_options must contain only uncertainty_multiplier')
    multiplier = options['uncertainty_multiplier']
    if (type(multiplier) not in (int, float) or
            not 1 <= multiplier <= UNCERTAINTY_MULTIPLIER or not math.isfinite(multiplier)):
        raise ValueError('uncertainty_multiplier must be finite and between 1 and 6')
    return dict(uncertainty_multiplier=float(multiplier))


def _safe_sample(observations, pass_index, physical, source, world):
    try:
        sample = _samples(observations, pass_index, physical, source, world)
        if sample is None:
            return None
        sample['uncertainty'] = float(sample['uncertainty'])
        numbers = (sample['arrivals'] + sample['baseline'] + sample['residual'] +
                   [UNCERTAINTY_MULTIPLIER * sample['uncertainty']])
        return sample if all(math.isfinite(value) for value in numbers) else None
    except (ArithmeticError, TypeError, ValueError):
        return None


def _matching(costs, threshold):
    """Deterministic feasibility for cost <= threshold; None forbids an edge."""
    size = len(costs)
    adjacent = [sorted((tile for tile, cost in enumerate(row)
                        if cost is not None and cost <= threshold),
                       key=lambda tile: (tile != slot, abs(tile - slot), tile))
                for slot, row in enumerate(costs)]
    owner = [-1] * size

    def augment(slot, seen):
        for tile in adjacent[slot]:
            if seen[tile]:
                continue
            seen[tile] = True
            if owner[tile] < 0 or augment(owner[tile], seen):
                owner[tile] = slot
                return True
        return False

    for slot in range(size):
        if not augment(slot, [False] * size):
            return None
    order = [-1] * size
    for tile, slot in enumerate(owner):
        order[slot] = tile
    return order


def _minimum_movement(costs, threshold):
    """Hungarian secondary objective: fewest changed slots, then distance.

    All edges retained here attain the optimal bottleneck.  Integer costs keep
    this tie-breaking independent of timing magnitudes and measurement units.
    """
    size = len(costs)
    movement_weight = size * max(0, size - 1) + 1
    forbidden = size * (movement_weight + size) + 1
    weights = [[(int(slot != tile) * movement_weight + abs(slot - tile))
                if cost is not None and cost <= threshold else forbidden
                for tile, cost in enumerate(row)] for slot, row in enumerate(costs)]
    u, v, owner, previous = ([0] * (size + 1) for _ in range(4))
    for slot in range(1, size + 1):
        owner[0] = slot
        column = 0
        minimum = [forbidden * (size + 1)] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            row, delta, next_column = owner[column], forbidden * (size + 1), 0
            for tile in range(1, size + 1):
                if used[tile]:
                    continue
                reduced = weights[row - 1][tile - 1] - u[row] - v[tile]
                if reduced < minimum[tile]:
                    minimum[tile], previous[tile] = reduced, column
                if minimum[tile] < delta:
                    delta, next_column = minimum[tile], tile
            for tile in range(size + 1):
                if used[tile]:
                    u[owner[tile]] += delta
                    v[tile] -= delta
                else:
                    minimum[tile] -= delta
            column = next_column
            if owner[column] == 0:
                break
        while column:
            preceding = previous[column]
            owner[column] = owner[preceding]
            column = preceding
    order = [-1] * size
    for tile in range(1, size + 1):
        order[owner[tile] - 1] = tile - 1
    if any(costs[slot][tile] is None or costs[slot][tile] > threshold
           for slot, tile in enumerate(order)):
        raise ValueError('Secondary assignment violated bottleneck feasibility')
    return order


def _bottleneck_assignment(costs):
    """Exact minimax assignment over a finite square cost matrix with None edges."""
    levels = sorted({cost for row in costs for cost in row if cost is not None})
    if not levels or _matching(costs, levels[-1]) is None:
        return None
    low, high = 0, len(levels) - 1
    while low < high:
        middle = (low + high) // 2
        if _matching(costs, levels[middle]) is None:
            low = middle + 1
        else:
            high = middle
    threshold = levels[low]
    return _minimum_movement(costs, threshold), threshold


def _cost_matrix(samples, maxima, margins, max_displacement=None):
    size = len(samples[0]['arrivals'])
    result = []
    for slot in range(size):
        row = []
        for tile in range(size):
            if max_displacement is not None and abs(tile - slot) > max_displacement:
                row.append(None)
                continue
            pass_costs = [sample['baseline'][slot] + sample['residual'][tile] - maximum + margin
                          for sample, maximum, margin in zip(samples, maxima, margins)]
            if not all(math.isfinite(cost) for cost in pass_costs):
                return None
            row.append(max(pass_costs))
        result.append(row)
    return result


def _fit_window(samples, max_displacement, uncertainty_multiplier=UNCERTAINTY_MULTIPLIER):
    identity = list(range(len(samples[0]['arrivals'])))
    maxima = [max(sample['arrivals']) for sample in samples]
    margins = [uncertainty_multiplier * sample['uncertainty'] for sample in samples]
    bounds = []
    for pass_index, sample, maximum, margin in zip(TRAINING_PASSES, samples, maxima, margins):
        # For one nondecreasing baseline, oppositely sorting residuals minimizes
        # max(b_i+r_pi(i)). Independent passes and unbounded movement RELAX the
        # real problem; failure of either bound proves no joint candidate can pass.
        ideal_arrivals = [base + residual for base, residual in
                          zip(sample['baseline'], sorted(sample['residual'], reverse=True))]
        optimistic = max(ideal_arrivals)
        gain = maximum - optimistic
        if not all(math.isfinite(value) for value in ideal_arrivals + [gain]):
            return identity, dict(status='nonfinite_model_identity', optimistic_bound=[], candidate=None)
        bounds.append(dict(pass_index=pass_index, before_max_us=maximum,
                           after_max_us=optimistic, gain_upper_bound_us=gain,
                           required_margin_us=margin,
                           clears_margin=gain > margin + STRICT_TOLERANCE_US))
    result = dict(optimistic_bound=bounds, candidate=None)
    if any(bound['gain_upper_bound_us'] <= STRICT_TOLERANCE_US for bound in bounds):
        result.update(status='no_model_gain_identity', rejection='single_pass_optimistic_bound')
        return identity, result
    if not all(bound['clears_margin'] for bound in bounds):
        result.update(status='uncertainty_limited_identity', rejection='single_pass_optimistic_bound')
        return identity, result
    costs = _cost_matrix(samples, maxima, margins, max_displacement)
    if costs is None:
        result.update(status='nonfinite_model_identity', rejection='nonfinite_assignment_cost')
        return identity, result
    # Strict gate before optimizing: no need to solve a positive-cost assignment.
    negative = [[cost if cost is not None and cost < -STRICT_TOLERANCE_US else None
                 for cost in row] for row in costs]
    if _matching(negative, 0.0) is None:
        unbounded = _cost_matrix(samples, maxima, margins)
        if unbounded is None:
            result.update(status='nonfinite_model_identity', rejection='nonfinite_unbounded_cost')
        else:
            negative_unbounded = [[cost if cost < -STRICT_TOLERANCE_US else None for cost in row]
                                  for row in unbounded]
            limited = _matching(negative_unbounded, 0.0) is not None
            result.update(status='displacement_limited_identity' if limited else 'inconsistent_training_identity',
                          rejection='joint_strict_gate_infeasible',
                          unrestricted_joint_gate_feasible=limited)
        return identity, result
    order, bottleneck = _bottleneck_assignment(negative)
    after = [max(_predicted(sample, order)) for sample in samples]
    gains = [before - end for before, end in zip(maxima, after)]
    if not all(math.isfinite(value) for value in after + gains):
        result.update(status='nonfinite_model_identity', rejection='nonfinite_final_gain')
        return identity, result
    result['candidate'] = dict(order=order, before_max_us=maxima, after_max_us=after,
                               gain_us=gains, required_margin_us=margins,
                               bottleneck_cost_us=bottleneck,
                               changed_slots=sum(slot != tile for slot, tile in enumerate(order)),
                               max_displacement=max(abs(slot - tile) for slot, tile in enumerate(order)))
    if not all(gain > margin + STRICT_TOLERANCE_US for gain, margin in zip(gains, margins)):
        result.update(status='numerical_guard_identity', rejection='final_strict_gate_failed')
        return identity, result
    result.update(status='candidate', rejection=None)
    return order, result


def fit_order(rows, observations, world, window, max_displacement=MAX_DISPLACEMENT, options=None):
    """Same mapping interface as v1, with bounded joint window assignments.

    Displacement counts positions WITHIN a destination window, not CTA waves or
    physical execution time. Pass 2 is diagnostic only. Missing/invalid training
    observations retain the complete identity window. Explicit options freeze a
    v3 margin; they never scale the recorded observation errors or inspect pass 2
    to choose the margin. None reproduces historical v2 output without new keys.
    """
    options = normalize_options(options)
    multiplier = UNCERTAINTY_MULTIPLIER if options is None else options['uncertainty_multiplier']
    if type(window) is not int or window <= 0:
        raise ValueError('window must be a positive integer')
    if type(max_displacement) is not int or max_displacement < 0:
        raise ValueError('max_displacement must be a nonnegative integer')
    by_rank, tn, tiles = _mapping_rows(rows, world)
    maps, groups = [list(range(tiles)) for _ in range(world)], []
    for source, rank_rows in enumerate(by_rank):
        for destination in sorted({row['destination'] for row in rank_rows}):
            destination_slots = [row['index'] for row in rank_rows if row['destination'] == destination]
            for start in range(0, len(destination_slots), window):
                slots = destination_slots[start:start + window]
                physical = [rank_rows[slot]['m'] * tn + rank_rows[slot]['n'] for slot in slots]
                samples = [_safe_sample(observations, p, physical, source, world) for p in TRAINING_PASSES]
                order = list(range(len(slots)))
                group = dict(source=source, destination=destination, window_index=start // window,
                             slots=slots, physical_tiles=physical, training=[], optimistic_bound=[], candidate=None)
                if any(sample is None for sample in samples):
                    group.update(status='incomplete_training_identity', rejection='missing_or_invalid_training')
                else:
                    order, fitted = _fit_window(samples, max_displacement, multiplier)
                    group.update(fitted)
                    if options is not None and group['candidate'] is not None:
                        # The baseline is the observed identity maximum (error
                        # <= epsilon), not another shifted prediction. PAVA is
                        # L-infinity nonexpansive, giving <= 3*epsilon error for
                        # the candidate and <= 4*epsilon for their difference.
                        # This is conditional on a uniform observation bound
                        # and the surrogate; it does not cover model mismatch.
                        group['candidate']['conservative_gain_lower_bound_us'] = [
                            gain - CONDITIONAL_ERROR_MULTIPLIER * sample['uncertainty']
                            for gain, sample in zip(group['candidate']['gain_us'], samples)]
                    for p, sample in zip(TRAINING_PASSES, samples):
                        predicted = (_predicted(sample, order) if group['status'] == 'candidate'
                                     else sample['arrivals'][:])
                        group['training'].append(dict(pass_index=p, arrivals_us=sample['arrivals'],
                            baseline_us=sample['baseline'], residual_us=sample['residual'],
                            max_uncertainty_us=sample['uncertainty'], before_max_us=max(sample['arrivals']),
                            after_max_us=max(predicted), predicted_arrivals_us=predicted))
                group.update(changed_slots=sum(slot != tile for slot, tile in enumerate(order)),
                             max_displacement=max(abs(slot - tile) for slot, tile in enumerate(order)))
                for slot, original in zip(slots, order):
                    maps[source][slot] = slots[original]
                heldout = _safe_sample(observations, HELDOUT_PASS, physical, source, world)
                if heldout is None:
                    group['heldout'] = dict(pass_index=HELDOUT_PASS, available=False)
                else:
                    predicted = _predicted(heldout, order)
                    before, after = max(heldout['arrivals']), max(predicted)
                    gain = before - after
                    if not all(math.isfinite(value) for value in predicted + [before, after, gain]):
                        group['heldout'] = dict(pass_index=HELDOUT_PASS, available=False)
                    else:
                        group['heldout'] = dict(pass_index=HELDOUT_PASS, available=True, before_max_us=before,
                            after_max_us=after, gain_us=gain, max_uncertainty_us=heldout['uncertainty'])
                groups.append(group)
    changed = [sum(slot != tile for slot, tile in enumerate(row)) for row in maps]
    diagnostics = dict(method=METHOD, training_passes=list(TRAINING_PASSES), heldout_passes=[HELDOUT_PASS],
        heldout_metric='Frozen-order proxy recomputed with pass-2 PAVA baseline; not model validation.',
        optimistic_bound_metric='Independent single-pass optimum with unrestricted displacement; an optimistic relaxation.',
        identity_fallback=not any(changed), changed_per_rank=changed,
        accepted_windows=sum(group['status'] == 'candidate' for group in groups),
        max_displacement=max_displacement, displacement_unit='position within destination window',
        uncertainty_multiplier=multiplier, strict_tolerance_us=STRICT_TOLERANCE_US,
        groups=groups, gpu_validated=False,
        limitation='Receiver-local per-source/window surrogate from separate sampled invocations. '
                   'The 6*error bound covers bounded observation error conditional on this model, '
                   'not statistical confidence or model misspecification. No global makespan or GPU '
                   'performance claim. Requires independent same-codec intervention and measurement.')
    if options is not None:
        diagnostics.update(method=CONFIGURED_METHOD, arrival_options=options,
            exploratory=multiplier < CONDITIONAL_ERROR_MULTIPLIER,
            conditional_error_multiplier=CONDITIONAL_ERROR_MULTIPLIER,
            accepted_windows_with_positive_conservative_bound=sum(
                group['status'] == 'candidate' and
                all(value > STRICT_TOLERANCE_US for value in
                    group['candidate']['conservative_gain_lower_bound_us']) for group in groups),
            limitation='Configured receiver-local tail surrogate; the recorded error proxy is unchanged. '
                'A margin below 4*error admits exploratory candidates without a positive conservative '
                'gain guarantee. gain-4*error is a lower bound only conditional on uniformly bounded '
                'observation error and this surrogate, not statistical confidence or model validity. '
                'Pass 2 is diagnostic only. Independent same-codec GPU measurement is required.')
    return maps, diagnostics
