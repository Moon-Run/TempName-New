"""Conservative CPU-only receiver-tail candidate ordering.

This is an offline surrogate, not a CUDA execution-time model.  Each source /
receiver / destination-window is fitted independently; no clocks from different
receivers are compared.  A nondecreasing fit removes the delay already explained
by the original logical order.  Only repeatable residuals may justify a swap.

Passes 0/1 select the mapping.  Pass 2 diagnoses that frozen mapping and never
accepts, rejects, or chooses a candidate.  BF16 observations cannot establish
performance on a quantized runtime: new same-codec GPU measurements are required.
"""
import math


METHOD = 'receiver-tail-residual-v1'
TRAINING_PASSES = (0, 1)
HELDOUT_PASS = 2
MAX_SWAPS_PER_WINDOW = 4
UNCERTAINTY_MULTIPLIER = 6.0


def _pava(values):
    """Equal-weight least-squares nondecreasing fit, preserving input positions."""
    blocks = []
    for value in values:
        blocks.append([value, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            right_sum, right_count = blocks.pop()
            blocks[-1][0] += right_sum
            blocks[-1][1] += right_count
    return [total / count for total, count in blocks for _ in range(count)]


def _mapping_rows(rows, world):
    if type(world) is not int or world < 2:
        raise ValueError('world must be an integer >= 2')
    if not rows or {row['rank'] for row in rows} != set(range(world)):
        raise ValueError('Missing mapping rank')
    tn = max(row['n'] for row in rows) + 1
    if tn <= 0:
        raise ValueError('Invalid tile columns')
    by_rank = [sorted((r for r in rows if r['rank'] == rank), key=lambda r: r['index'])
               for rank in range(world)]
    tiles = len(by_rank[0])
    destinations = {}
    for rr in by_rank:
        if [r['index'] for r in rr] != list(range(tiles)):
            raise ValueError('Missing or duplicate logical slot')
        physical = []
        for row in rr:
            if row['m'] < 0 or not 0 <= row['n'] < tn or not 0 <= row['destination'] < world:
                raise ValueError('Invalid physical tile/destination')
            if 'tn' in row and row['tn'] != tn:
                raise ValueError('Inconsistent tile-column count')
            tile = row['m'] * tn + row['n']
            physical.append(tile)
            if destinations.setdefault(tile, row['destination']) != row['destination']:
                raise ValueError('Physical tile has inconsistent destinations')
        if sorted(physical) != list(range(tiles)):
            raise ValueError('Physical mapping must be a complete bijection')
    return by_rank, tn, tiles


def _samples(observations, pass_index, physical, source, world):
    values, errors = [], []
    for tile in physical:
        observation = observations.get((pass_index, tile))
        if not isinstance(observation, (tuple, list)) or len(observation) != 2:
            return None
        arrivals, uncertainty = observation
        if not isinstance(arrivals, (tuple, list)) or len(arrivals) != world:
            return None
        try:
            valid = all(math.isfinite(value) for value in arrivals)
            valid = valid and math.isfinite(uncertainty) and uncertainty >= 0
        except (TypeError, ValueError):
            return None
        if not valid:
            return None
        values.append(float(arrivals[source]))
        errors.append(float(uncertainty))
    baseline = _pava(values)
    if not all(math.isfinite(value) for value in baseline):
        return None
    return dict(arrivals=values, baseline=baseline,
                residual=[value - fitted for value, fitted in zip(values, baseline)],
                uncertainty=max(errors, default=0.0))


def _predicted(sample, order):
    return [sample['baseline'][slot] + sample['residual'][tile]
            for slot, tile in enumerate(order)]


def _fit_window(samples, slots, physical):
    order = list(range(len(slots)))
    accepted = []
    # Max-norm uncertainty: each PAVA fitted value is within max(error) if
    # observations are, so a shifted prediction has <= 3*error uncertainty.
    # Comparing two shifted predictions uses 6*error, also conservatively
    # covering the first swap whose unmodified proxy has only error uncertainty.
    # This addresses observation error only, not misspecification of the model.
    margins = [UNCERTAINTY_MULTIPLIER * sample['uncertainty'] for sample in samples]
    for _ in range(MAX_SWAPS_PER_WINDOW):
        current = [_predicted(sample, order) for sample in samples]
        maxima = [max(values) for values in current]
        # After replacing two slots, the unaffected maximum is among the top 3.
        largest = [sorted(range(len(slots)), key=lambda i: (-values[i], i))[:3]
                   for values in current]
        best = None
        for early in range(len(slots)):
            for late in range(early + 1, len(slots)):
                earlier_tile, later_tile = order[early], order[late]
                # No useful signal in a monotone schedule, an unresolved
                # residual difference, or one that changes sign across passes.
                if not all(sample['residual'][later_tile] > sample['uncertainty'] and
                           sample['residual'][later_tile] - sample['residual'][earlier_tile] >
                           2 * sample['uncertainty'] for sample in samples):
                    continue
                after = []
                for sample, values, top in zip(samples, current, largest):
                    rest = next((values[i] for i in top if i not in (early, late)), -math.inf)
                    # Include BOTH tiles. Advancing one contribution is not
                    # useful if its displaced tile creates an equal/later tail.
                    after.append(max(rest,
                        sample['baseline'][early] + sample['residual'][later_tile],
                        sample['baseline'][late] + sample['residual'][earlier_tile]))
                gains = [before - new for before, new in zip(maxima, after)]
                if not all(gain > margin + 1e-12 for gain, margin in zip(gains, margins)):
                    continue
                # Deterministic best-first selection, preferring less movement
                # when conservative gains tie. No held-out values are visible.
                key = (min(gain - margin for gain, margin in zip(gains, margins)),
                       sum(gains), -(slots[late] - slots[early]), -early, -late)
                if best is None or key > best[0]:
                    best = (key, early, late, after, gains)
        if best is None:
            break
        _, early, late, after, gains = best
        accepted.append(dict(slots=[slots[early], slots[late]],
                             physical_tiles=[physical[order[early]], physical[order[late]]],
                             before_max_us=maxima, after_max_us=after,
                             gain_us=gains, required_margin_us=margins))
        order[early], order[late] = order[late], order[early]
    return order, accepted


def fit_order(rows, observations, world, window):
    """Return ``(maps, diagnostics)`` without modifying inputs.

    ``rows`` is the all-rank mapping CSV as dictionaries, with integer rank,
    index, m, n, destination (tn optional). ``observations[(pass, physical_tile)]``
    is ``(arrival_us_by_source, uncertainty_us)`` in that tile receiver's local
    time frame. ``maps[source][slot]`` is the original logical slot moved there.

    Missing/invalid training samples leave their complete destination-window
    unchanged. A window's candidate must reduce its source-specific worst arrival
    in BOTH training passes, beyond measurement uncertainty. This deliberately
    gives up speculative opportunities instead of claiming a measured speedup.
    """
    if type(window) is not int or window <= 0:
        raise ValueError('window must be a positive integer')
    by_rank, tn, tiles = _mapping_rows(rows, world)
    maps = [list(range(tiles)) for _ in range(world)]
    groups = []
    for source, rr in enumerate(by_rank):
        for destination in sorted({r['destination'] for r in rr}):
            destination_slots = [r['index'] for r in rr if r['destination'] == destination]
            for start in range(0, len(destination_slots), window):
                slots = destination_slots[start:start + window]
                physical = [rr[i]['m'] * tn + rr[i]['n'] for i in slots]
                group = dict(source=source, destination=destination, window_index=start // window,
                             slots=slots, physical_tiles=physical, accepted_swaps=[], training=[])
                samples = [_samples(observations, p, physical, source, world) for p in TRAINING_PASSES]
                order = list(range(len(slots)))
                if any(sample is None for sample in samples):
                    group['status'] = 'incomplete_training_identity'
                else:
                    order, swaps = _fit_window(samples, slots, physical)
                    group['accepted_swaps'] = swaps
                    group['status'] = 'candidate' if swaps else 'no_resolved_gain_identity'
                    for p, sample in zip(TRAINING_PASSES, samples):
                        predicted = _predicted(sample, order)
                        group['training'].append(dict(pass_index=p, arrivals_us=sample['arrivals'],
                            baseline_us=sample['baseline'], residual_us=sample['residual'],
                            max_uncertainty_us=sample['uncertainty'],
                            before_max_us=max(sample['arrivals']), after_max_us=max(predicted),
                            predicted_arrivals_us=predicted))
                for slot, original in zip(slots, order):
                    maps[source][slot] = slots[original]
                # Diagnostics only: pass 2 re-estimates its own nuisance trend;
                # this is not an out-of-sample test of the timing model.
                # Neither this fit nor its result can change order/status.
                heldout = _samples(observations, HELDOUT_PASS, physical, source, world)
                if heldout is None:
                    group['heldout'] = dict(pass_index=HELDOUT_PASS, available=False)
                else:
                    before = max(heldout['arrivals'])
                    after = max(_predicted(heldout, order))
                    group['heldout'] = dict(pass_index=HELDOUT_PASS, available=True,
                        before_max_us=before, after_max_us=after, gain_us=before - after,
                        max_uncertainty_us=heldout['uncertainty'])
                groups.append(group)
    changed = [sum(i != j for i, j in enumerate(row)) for row in maps]
    diagnostics = dict(method=METHOD, training_passes=list(TRAINING_PASSES), heldout_passes=[HELDOUT_PASS],
        heldout_metric='Frozen-order proxy recomputed with pass-2 PAVA baseline; not model validation.',
        identity_fallback=not any(changed), changed_per_rank=changed,
        accepted_swaps=sum(len(group['accepted_swaps']) for group in groups),
        max_swaps_per_window=MAX_SWAPS_PER_WINDOW, uncertainty_multiplier=UNCERTAINTY_MULTIPLIER,
        groups=groups, gpu_validated=False,
        limitation='Receiver-local, per-source/window surrogate from separate sampled invocations; '
                   'logical slots are not CUDA execution times. No cross-receiver global makespan '
                   'or performance claim. Requires independent same-codec GPU measurement.')
    return maps, diagnostics
