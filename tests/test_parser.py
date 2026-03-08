"""
tests/test_parser.py
Unit tests for agent/parser.py (Teacher parser — JSON format only)
and api/inference.py parse_action_student (extended formats)
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.parser import parse_action


# ── TEACHER PARSER (agent/parser.py) ─────────────────────────────────────────

class TestTeacherParser:

    def test_lookup_policy_json(self):
        text = 'Thought: I need to look up the policy.\nAction: lookup_policy({"user_id": "P-1001"})'
        name, args = parse_action(text)
        assert name == "lookup_policy"
        assert args == {"user_id": "P-1001"}

    def test_check_rules_json(self):
        text = (
            'Action: check_rules({"claim_type": "hail", "plan_type": "Premium", '
            '"policy_covers": ["hail"], "policy_status": "active", "claims_this_year": 1})'
        )
        name, args = parse_action(text)
        assert name == "check_rules"
        assert args["claim_type"] == "hail"
        assert args["claims_this_year"] == 1

    def test_calculate_payout_json(self):
        text = (
            'Action: calculate_payout({"claimed_amount": 1200, "deductible": 500, '
            '"max_single_claim": 15000, "max_annual_payout": 50000})'
        )
        name, args = parse_action(text)
        assert name == "calculate_payout"
        assert args["claimed_amount"] == 1200
        assert args["deductible"] == 500

    def test_single_quotes_normalised(self):
        text = "Action: lookup_policy({'user_id': 'P-1002'})"
        name, args = parse_action(text)
        assert name == "lookup_policy"
        assert args == {"user_id": "P-1002"}

    def test_no_action_returns_none(self):
        text = "Thought: I have all the information I need."
        name, args = parse_action(text)
        assert name is None
        assert args is None

    def test_invalid_json_returns_none_args(self):
        text = "Action: lookup_policy(not valid json)"
        name, args = parse_action(text)
        assert name == "lookup_policy"
        assert args is None

    def test_extra_whitespace(self):
        text = "Action:  check_rules(  {\"claim_type\": \"storm\", \"plan_type\": \"Basic\", \"policy_covers\": [\"storm\"], \"policy_status\": \"active\", \"claims_this_year\": 0}  )"
        name, args = parse_action(text)
        assert name == "check_rules"
        assert args is not None

    def test_multiline_thought_before_action(self):
        text = (
            "Thought: I need to check the rules.\n"
            "The policy is active and covers hail.\n"
            'Action: check_rules({"claim_type": "hail", "plan_type": "Premium", '
            '"policy_covers": ["hail"], "policy_status": "active", "claims_this_year": 0})'
        )
        name, args = parse_action(text)
        assert name == "check_rules"
        assert args["claim_type"] == "hail"

    def test_verdict_text_no_action(self):
        text = "Verdict: APPROVED\nPayout: $700\nReasoning: Claim is valid."
        name, args = parse_action(text)
        assert name is None
        assert args is None


# ── STUDENT PARSER (api/inference.py) ────────────────────────────────────────

class TestStudentParser:
    """
    Test the extended student parser that handles JSON, kwargs, and positional formats.
    Imported directly from api/inference.py.
    """

    @pytest.fixture(autouse=True)
    def import_student_parser(self):
        from api.inference import parse_action_student
        self.parse = parse_action_student

    def test_json_format(self):
        text = 'Action: lookup_policy({"user_id": "P-1001"})'
        name, args = self.parse(text)
        assert name == "lookup_policy"
        assert args["user_id"] == "P-1001"

    def test_kwargs_format(self):
        text = 'Action: calculate_payout(claimed_amount=1200, deductible=500, max_single_claim=15000, max_annual_payout=50000)'
        name, args = self.parse(text)
        assert name == "calculate_payout"
        assert args["claimed_amount"] == 1200
        assert args["deductible"] == 500

    def test_positional_format(self):
        text = 'Action: calculate_payout(1200, 500, 15000, 50000, 0)'
        name, args = self.parse(text)
        assert name == "calculate_payout"
        assert args["claimed_amount"] == 1200
        assert args["deductible"] == 500
        assert args["max_single_claim"] == 15000

    def test_already_claimed_defaults_to_zero_json(self):
        text = 'Action: calculate_payout({"claimed_amount": 1200, "deductible": 500, "max_single_claim": 15000, "max_annual_payout": 50000})'
        name, args = self.parse(text)
        assert args["already_claimed_this_year"] == 0

    def test_already_claimed_defaults_to_zero_positional(self):
        text = 'Action: calculate_payout(1200, 500, 15000, 50000)'
        name, args = self.parse(text)
        assert args.get("already_claimed_this_year", 0) == 0

    def test_no_action_returns_none(self):
        text = "Thought: I need to think about this."
        name, args = self.parse(text)
        assert name is None
        assert args is None