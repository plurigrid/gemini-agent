"""Tests for CellValue lattice, GF(3) trit arithmetic, and AGM belief revision."""

import json
import pytest
from my_agent.tools_propagator import (
    lattice_merge,
    lattice_merge_many,
    trit_classify,
    trit_add,
    trit_mul,
    trit_conservation_check,
    agm_revise,
    agm_contract,
)


# -- CellValue lattice --------------------------------------------------------

class TestLatticeMerge:
    def test_nothing_absorbs_incoming(self):
        assert lattice_merge("nothing", "value")["result"] == "value"
        assert lattice_merge("nothing", "contradiction")["result"] == "contradiction"
        assert lattice_merge("nothing", "nothing")["result"] == "nothing"

    def test_contradiction_absorbs_everything(self):
        assert lattice_merge("contradiction", "nothing")["result"] == "contradiction"
        assert lattice_merge("contradiction", "value")["result"] == "contradiction"
        assert lattice_merge("contradiction", "contradiction")["result"] == "contradiction"

    def test_value_idempotent(self):
        r = lattice_merge("value", "value")
        assert r["result"] == "value"
        assert r["changed"] is False

    def test_value_nothing_stays(self):
        r = lattice_merge("value", "nothing")
        assert r["result"] == "value"
        assert r["changed"] is False

    def test_value_contradiction_escalates(self):
        r = lattice_merge("value", "contradiction")
        assert r["result"] == "contradiction"
        assert r["changed"] is True

    def test_ordering_nothing_lt_value_lt_contradiction(self):
        """Nothing < Value < Contradiction — the lattice is monotone."""
        states = ["nothing", "value", "contradiction"]
        for i, s in enumerate(states):
            for j, t in enumerate(states):
                r = lattice_merge(s, t)
                # result should be >= max(s, t) in the lattice ordering
                result_idx = states.index(r["result"])
                assert result_idx >= max(i, j)


class TestLatticeMergeMany:
    def test_all_nothing(self):
        r = lattice_merge_many('["nothing", "nothing", "nothing"]')
        assert r["result"] == "nothing"

    def test_single_value(self):
        r = lattice_merge_many('["nothing", "value", "nothing"]')
        assert r["result"] == "value"

    def test_contradiction_poisons(self):
        r = lattice_merge_many('["value", "value", "contradiction", "value"]')
        assert r["result"] == "contradiction"

    def test_invalid_json(self):
        r = lattice_merge_many("not json")
        assert r.get("status") == "contradiction"


# -- GF(3) trit arithmetic ---------------------------------------------------

class TestTritClassify:
    def test_minus(self):
        assert trit_classify(-0.5)["trit"] == "minus"

    def test_zero(self):
        assert trit_classify(0.0)["trit"] == "zero"

    def test_plus(self):
        assert trit_classify(0.7)["trit"] == "plus"

    def test_boundary_minus(self):
        assert trit_classify(-0.33)["trit"] == "zero"

    def test_boundary_plus(self):
        assert trit_classify(0.33)["trit"] == "zero"


class TestTritAdd:
    def test_zero_identity(self):
        assert trit_add("plus", "zero")["result"] == "plus"
        assert trit_add("minus", "zero")["result"] == "minus"
        assert trit_add("zero", "zero")["result"] == "zero"

    def test_plus_minus_cancel(self):
        assert trit_add("plus", "minus")["result"] == "zero"

    def test_plus_plus_wraps(self):
        # In balanced ternary mod 3: +1 + +1 = +2 → -1 (mod 3)
        assert trit_add("plus", "plus")["result"] == "minus"

    def test_minus_minus_wraps(self):
        assert trit_add("minus", "minus")["result"] == "plus"


class TestTritMul:
    def test_zero_annihilates(self):
        assert trit_mul("plus", "zero")["result"] == "zero"
        assert trit_mul("zero", "minus")["result"] == "zero"

    def test_plus_identity(self):
        assert trit_mul("plus", "plus")["result"] == "plus"
        assert trit_mul("minus", "plus")["result"] == "minus"

    def test_minus_minus_positive(self):
        assert trit_mul("minus", "minus")["result"] == "plus"


class TestTritConservation:
    def test_balanced(self):
        r = trit_conservation_check('["plus", "minus", "zero"]')
        assert r["conserved"] is True

    def test_unbalanced(self):
        r = trit_conservation_check('["plus", "plus", "zero"]')
        assert r["conserved"] is False

    def test_all_zero(self):
        r = trit_conservation_check('["zero", "zero", "zero"]')
        assert r["conserved"] is True

    def test_three_plus(self):
        # +1+1+1 = 3 → 0 mod 3 → balanced
        r = trit_conservation_check('["plus", "plus", "plus"]')
        assert r["conserved"] is True


# -- AGM belief revision ------------------------------------------------------

class TestAGMRevise:
    def test_basic_revision(self):
        r = agm_revise('["p", "q", "~r"]', "r")
        assert "r" in r["revised"]
        assert "~r" not in r["revised"]

    def test_levi_identity_format(self):
        r = agm_revise('["p"]', "q")
        assert "~q" in r["levi_identity"]

    def test_no_duplicate(self):
        r = agm_revise('["p", "q"]', "q")
        assert r["revised"].count("q") == 1


class TestAGMContract:
    def test_removes_belief(self):
        r = agm_contract('["p", "q", "~q"]', "q")
        assert "q" not in r["revised"]
        assert "~q" not in r["revised"]
        assert "p" in r["revised"]
