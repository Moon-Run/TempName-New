"""CPU-only closed reduction groups and a budget-matched Latin control.

The strategy is fixed before execution; phase rotates all eight column residues.
No arrival observations or GPU timing choose these masks.
"""
import hashlib
import json

PROTOCOL = "h100-reduction-closure-v1"
ARMS = ("SP", "CP", "SM", "CM")
WORLD, TILES, TILES_PER_DESTINATION = 8, 512, 64
WIRE_BYTES_PER_RANK = 13_819_904


def validate(arm, phase):
    if arm not in ARMS or type(phase) is not int or not 0 <= phase < 8:
        raise ValueError("Expected closure arm SP/CP/SM/CM and integer phase 0..7")


def mul2(value):
    if type(value) is not int or not 0 <= value < 8:
        raise ValueError("GF(8) element must be an integer 0..7")
    return ((value << 1) ^ (0b1011 if value & 4 else 0)) & 7


def latin(source, destination):
    return mul2(source) ^ mul2(destination) ^ destination


def selected(arm, phase, source, destination, within):
    validate(arm, phase)
    if any(type(x) is not int or not 0 <= x < 8 for x in (source, destination)):
        raise ValueError("Source and destination must be integer ranks 0..7")
    if type(within) is not int or not 0 <= within < 64:
        raise ValueError("Within-destination tile must be an integer 0..63")
    residue = (within + phase) % 8
    return source != destination and residue == (
        destination if arm[0] == "C" else latin(source, destination))


def mask(arm, phase):
    """Return uint8-compatible [source][physical tile] values, no device state."""
    validate(arm, phase)
    return [[int(selected(arm, phase, s, tile // 64, tile % 64))
             for tile in range(512)] for s in range(8)]


def mask_sha256(arm, phase):
    return hashlib.sha256(bytes(v for row in mask(arm, phase) for v in row)).hexdigest()


def statistics(arm, phase):
    values = mask(arm, phase)
    counts = [sum(row) for row in values]
    links = [[sum(values[s][d * 64:(d + 1) * 64]) for d in range(8)] for s in range(8)]
    fanin = [[sum(values[s][d * 64 + t] for s in range(8)) for t in range(64)] for d in range(8)]
    inverse_tiles = [sum(k if arm[1] == "P" else int(k > 0) for k in row) for row in fanin]
    # 64 tiles per receiver, 128 independent H128 rows per tile.
    inverse_groups = [count * 128 for count in inverse_tiles]
    wire = [sum((128 * 136 if value else 128 * 256)
                for tile, value in enumerate(row) if tile // 64 != source)
            for source, row in enumerate(values)]
    expected_inverse = 1024 if arm == "CM" else 7168
    if counts != [56] * 8 or any(links[s][d] != (0 if s == d else 8)
                               for s in range(8) for d in range(8)):
        raise ValueError("Closure budget invariants failed")
    if inverse_groups != [expected_inverse] * 8 or wire != [WIRE_BYTES_PER_RANK] * 8:
        raise ValueError("Closure work or wire invariants failed")
    return dict(arm=arm, phase=phase, mask_sha256=mask_sha256(arm, phase),
                selected_per_source=counts, selected_per_link=links,
                quantized_contributions=448, quantized_fraction=.125,
                inverse_groups_per_receiver=inverse_groups,
                wire_bytes_per_rank=wire, budget_matched=True,
                fanin_histogram_per_receiver=[{str(k): row.count(k) for k in sorted(set(row))}
                                              for row in fanin])


def protocol_record():
    return dict(protocol=PROTOCOL, arms=list(ARMS), phases=list(range(8)),
                mask_layout="source,physical_tile; destination=tile//64; within=tile%64",
                phase_formula="residue=(within+phase)%8",
                plans={str(phase): {arm: statistics(arm, phase) for arm in ARMS}
                       for phase in range(8)})


if __name__ == "__main__":
    print(json.dumps(protocol_record(), indent=2))
