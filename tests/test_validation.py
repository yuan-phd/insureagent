"""
tests/test_validation.py
Unit tests for data/validation.py
"""

import pytest
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.validation import validate_traces


# ── HELPERS ───────────────────────────────────────────────────────────────────

def make_valid_trace(verdict: str = "APPROVED") -> dict:
    """Create a minimal valid trace."""
    return {
        "conversations": [
            {"role": "system", "content": "You are an insurance adjuster."},
            {"role": "user", "content": "Process this claim."},
            {"role": "assistant", "content": 'Thought: Check policy.\nAction: lookup_policy({"user_id": "P-1001"})'},
            {"role": "user", "content": 'Observation: {"plan_type": "Premium"}'},
            {"role": "assistant", "content": 'Thought: Check rules.\nAction: check_rules({"claim_type": "hail", "plan_type": "Premium", "policy_covers": ["hail"], "policy_status": "active", "claims_this_year": 0})'},
            {"role": "user", "content": 'Observation: {"eligible": true}'},
            {"role": "assistant", "content": f"Verdict: {verdict}\nPayout: $700\nReasoning: Valid claim."},
        ]
    }


def write_jsonl(traces: list, path: str):
    with open(path, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")


# ── TESTS ─────────────────────────────────────────────────────────────────────

class TestValidateTraces:

    def test_valid_traces_pass(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        traces = [make_valid_trace("APPROVED")] * 10 + [make_valid_trace("DENIED")] * 5
        write_jsonl(traces, path)
        report = validate_traces(path)
        assert report["valid"] == 15
        assert report["invalid"] == 0
        assert report["error_rate"] == 0.0
        os.unlink(path)

    def test_missing_verdict_detected(self):
        bad_trace = {
            "conversations": [
                {"role": "system", "content": "You are an adjuster."},
                {"role": "user", "content": "Process this claim."},
                {"role": "assistant", "content": 'Action: lookup_policy({"user_id": "P-1001"})'},
                {"role": "user", "content": 'Observation: {"plan_type": "Premium"}'},
                {"role": "assistant", "content": "Thought: I need more info."},  # no Verdict
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        write_jsonl([bad_trace], path)
        # error_rate = 1.0 > 0.05 threshold → should raise
        with pytest.raises(ValueError, match="validation failed"):
            validate_traces(path)
        os.unlink(path)

    def test_hallucinated_observation_detected(self):
        bad_trace = {
            "conversations": [
                {"role": "system", "content": "You are an adjuster."},
                {"role": "user", "content": "Process this claim."},
                {"role": "assistant", "content": (
                    'Action: lookup_policy({"user_id": "P-1001"})\n'
                    'Observation: {"plan_type": "Premium"}'  # assistant generating Observation
                )},
                {"role": "assistant", "content": "Verdict: APPROVED\nPayout: $700\nReasoning: Valid."},
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        write_jsonl([bad_trace], path)
        with pytest.raises(ValueError, match="validation failed"):
            validate_traces(path)
        os.unlink(path)

    def test_too_few_turns_detected(self):
        bad_trace = {
            "conversations": [
                {"role": "system", "content": "You are an adjuster."},
                {"role": "user", "content": "Process this claim."},
                {"role": "assistant", "content": "Verdict: APPROVED\nPayout: $700\nReasoning: Valid."},
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        write_jsonl([bad_trace], path)
        with pytest.raises(ValueError, match="validation failed"):
            validate_traces(path)
        os.unlink(path)

    def test_error_rate_below_threshold_passes(self):
        """1 bad trace out of 25 = 4% error rate, below 5% threshold."""
        traces = [make_valid_trace()] * 24
        bad_trace = {
            "conversations": [
                {"role": "system", "content": "You are an adjuster."},
                {"role": "user", "content": "Claim."},
                {"role": "assistant", "content": "Action: lookup_policy"},
                {"role": "user", "content": "Observation: {}"},
                # no verdict
                {"role": "assistant", "content": "Thinking..."},
            ]
        }
        traces.append(bad_trace)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        write_jsonl(traces, path)
        report = validate_traces(path)  # should not raise
        assert report["error_rate"] < 0.05
        os.unlink(path)

    def test_report_structure(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        write_jsonl([make_valid_trace()] * 5, path)
        report = validate_traces(path)
        assert "total" in report
        assert "valid" in report
        assert "invalid" in report
        assert "error_rate" in report
        assert "errors" in report
        os.unlink(path)