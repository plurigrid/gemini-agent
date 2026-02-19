"""Tests for MARL tools — Markov category channels and mutual information."""

from my_agent.tools_marl import (
    generative_channel,
    recognition_channel,
    channel_compose,
    marl_reward_design,
    mutual_information,
)


class TestMarkovChannels:
    def test_generative_is_forward(self):
        r = generative_channel('{"charge": 0.5}', '{"action": "discharge"}')
        assert r["direction"] == "forward"
        assert r["morphism"] == "P(s', r | s, a)"
        assert r["status"] == "value"

    def test_recognition_is_inverse(self):
        r = recognition_channel('{"price": 42.0}')
        assert r["direction"] == "inverse"
        assert r["morphism"] == "Q(z | x)"
        assert r["status"] == "value"

    def test_compose_closes_loop(self):
        g = generative_channel('{"s": 1}', '{"a": 0}')
        r = recognition_channel('{"obs": 1}')
        c = channel_compose(str(g), str(r))
        assert c["loop_closed"] is True
        assert c["identity_gap"] == "KL(Q||P)"

    def test_trit_conservation(self):
        """All Markov category tools should conserve GF(3) trit at zero."""
        g = generative_channel('{"s": 1}', '{"a": 0}')
        r = recognition_channel('{"obs": 1}')
        c = channel_compose(str(g), str(r))
        assert g["trit"] == "zero"
        assert r["trit"] == "zero"
        assert c["trit"] == "zero"


class TestMARLRewards:
    def test_known_objective(self):
        r = marl_reward_design("cost_savings", '["agent_1", "agent_2"]')
        assert r["reward_spec"]["signal"] == "price_delta"
        assert r["status"] == "value"

    def test_unknown_objective_fallback(self):
        r = marl_reward_design("custom_metric", '["a"]')
        assert r["reward_spec"]["signal"] == "custom_metric"

    def test_cooperative_default(self):
        r = marl_reward_design("reliability", '["a"]')
        assert r["reward_type"] == "cooperative"


class TestMutualInformation:
    def test_returns_mi_structure(self):
        r = mutual_information('{"type": "gaussian"}', '{"type": "uniform"}')
        assert "I_XY" in r
        assert "H_X" in r
        assert "H_Y" in r
        assert r["status"] == "value"
