"""Propagator network tools — CellValue lattice, GF(3) trit, AGM belief revision.

Mirrors semantics from zig-syrup:
  propagator.zig  → CellValue{Nothing, Value, Contradiction}, latticeMerge
  continuation.zig → AGM belief revision (Levi identity), GF(3) Trit
"""

import json
import math


# -- CellValue lattice --------------------------------------------------------

def lattice_merge(existing: str, incoming: str) -> dict:
    """Merge two CellValue states following the partial information lattice.

    Nothing < Value < Contradiction.
    Mirrors propagator.zig latticeMerge semantics.

    Args:
        existing: Current cell state — 'nothing', 'value', or 'contradiction'.
        incoming: New cell state to merge.
    """
    e = existing.lower().strip()
    i = incoming.lower().strip()

    if e == "nothing":
        return {"result": i, "changed": i != "nothing"}
    if e == "contradiction":
        return {"result": "contradiction", "changed": False}
    # existing is 'value'
    if i == "nothing":
        return {"result": "value", "changed": False}
    if i == "value":
        return {"result": "value", "changed": False}
    if i == "contradiction":
        return {"result": "contradiction", "changed": True}
    # anything else: treat as different value → contradiction
    return {"result": "contradiction", "changed": True}


def lattice_merge_many(states: str) -> dict:
    """Merge a list of CellValue states left-to-right.

    Args:
        states: JSON array of cell states, e.g. '["nothing", "value", "value"]'.
    """
    try:
        items = json.loads(states)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON array", "status": "contradiction"}

    current = "nothing"
    changes = 0
    for item in items:
        result = lattice_merge(current, str(item))
        if result.get("changed"):
            changes += 1
        current = result["result"]

    return {"result": current, "changes": changes, "steps": len(items)}


# -- GF(3) trit arithmetic ---------------------------------------------------

TRIT_VALUES = {"minus": -1, "zero": 0, "plus": 1}
TRIT_NAMES = {-1: "minus", 0: "zero", 1: "plus"}


def trit_classify(value: float) -> dict:
    """Classify a numeric value into GF(3) trit.

    Args:
        value: Numeric value to classify.
    """
    if value < -0.33:
        trit = "minus"
    elif value > 0.33:
        trit = "plus"
    else:
        trit = "zero"
    return {"value": value, "trit": trit, "gf3": TRIT_VALUES[trit]}


def trit_add(a: str, b: str) -> dict:
    """Add two GF(3) trits (mod 3 arithmetic).

    Args:
        a: First trit — 'minus', 'zero', or 'plus'.
        b: Second trit — 'minus', 'zero', or 'plus'.
    """
    va = TRIT_VALUES.get(a.lower().strip(), 0)
    vb = TRIT_VALUES.get(b.lower().strip(), 0)
    # mod 3 in balanced ternary: map {-1,0,1} → {0,1,2}, add mod 3, map back
    raw = ((va + 1) + (vb + 1)) % 3 - 1
    name = TRIT_NAMES[raw]
    return {"a": a, "b": b, "result": name, "gf3": raw}


def trit_mul(a: str, b: str) -> dict:
    """Multiply two GF(3) trits.

    Args:
        a: First trit — 'minus', 'zero', or 'plus'.
        b: Second trit — 'minus', 'zero', or 'plus'.
    """
    va = TRIT_VALUES.get(a.lower().strip(), 0)
    vb = TRIT_VALUES.get(b.lower().strip(), 0)
    raw = va * vb
    # clamp to {-1, 0, 1}
    raw = max(-1, min(1, raw))
    name = TRIT_NAMES[raw]
    return {"a": a, "b": b, "result": name, "gf3": raw}


def trit_conservation_check(trits: str) -> dict:
    """Check if a sequence of trits conserves the GF(3) invariant (sums to zero).

    Args:
        trits: JSON array of trit names, e.g. '["plus", "minus", "zero"]'.
    """
    try:
        items = json.loads(trits)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON array", "conserved": False}

    total = 0
    for t in items:
        total = ((total + 1) + (TRIT_VALUES.get(str(t).lower().strip(), 0) + 1)) % 3 - 1

    return {
        "trits": items,
        "sum_trit": TRIT_NAMES.get(total, "zero"),
        "conserved": total == 0,
    }


# -- AGM belief revision -----------------------------------------------------

def agm_revise(belief_set: str, new_belief: str) -> dict:
    """Perform AGM belief revision: (K - ~p) + p (Levi identity).

    Args:
        belief_set: JSON array of current belief strings.
        new_belief: The new belief proposition to revise into the set.
    """
    try:
        beliefs = json.loads(belief_set)
    except json.JSONDecodeError:
        beliefs = [belief_set]

    negation = f"~{new_belief}"
    # Contract: remove negation
    contracted = [b for b in beliefs if b != negation]
    # Expand: add new belief
    if new_belief not in contracted:
        contracted.append(new_belief)

    return {
        "operation": "revise",
        "levi_identity": f"(K - {negation}) + {new_belief}",
        "original_size": len(beliefs),
        "revised": contracted,
        "revised_size": len(contracted),
        "trit": "zero",
    }


def agm_contract(belief_set: str, belief: str) -> dict:
    """Contract a belief from the set (AGM contraction).

    Args:
        belief_set: JSON array of current belief strings.
        belief: The belief to remove.
    """
    try:
        beliefs = json.loads(belief_set)
    except json.JSONDecodeError:
        beliefs = [belief_set]

    contracted = [b for b in beliefs if b != belief and b != f"~{belief}"]
    return {
        "operation": "contract",
        "removed": belief,
        "revised": contracted,
        "trit": "zero",
    }
