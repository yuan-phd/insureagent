# Parser resolves text into executable python tool and param

import re
import json

def parse_action(text: str):
    """
    Extract tool name and arguments from model output.
    Handles messy formatting gracefully.
    Returns (tool_name, args_dict) or (None, None) if no valid action found.
    """

    # Match Action: tool_name(...) — capture everything inside outer parens
    match = re.search(
        r'Action:\s*(\w+)\s*\((.*)\)',
        text,
        re.DOTALL
    )

    if not match:
        return None, None

    tool_name = match.group(1).strip()
    args_str  = match.group(2).strip()

    # Must start with { to be valid JSON object
    if not args_str.startswith('{'):
        return tool_name, None

    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        try:
            args_str_fixed = args_str.replace("'", '"')
            args = json.loads(args_str_fixed)
        except json.JSONDecodeError:
            return tool_name, None

    return tool_name, args


if __name__ == "__main__":
    t1 = 'Thought: I need to look up the policy.\nAction: lookup_policy({"user_id": "P-1001"})'
    print("Test 1:", parse_action(t1))

    t2 = 'Action:  check_rules( {"claim_type": "hail", "plan_type": "Premium", "policy_covers": ["hail"], "policy_status": "active", "claims_this_year": 1} )'
    print("Test 2:", parse_action(t2))

    t3 = 'Thought: I have all the information I need to make a decision.'
    print("Test 3:", parse_action(t3))

    t4 = "Action: lookup_policy({'user_id': 'P-1002'})"
    print("Test 4:", parse_action(t4))

    t5 = 'Action: calculate_payout({"claimed_amount": 1200, "deductible": 500, "max_single_claim": 15000, "max_annual_payout": 50000})'
    print("Test 5:", parse_action(t5))