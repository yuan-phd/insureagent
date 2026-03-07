# prompt for both teacher and student mode
# 1, tell model is insuance claims adjuster with 3 tools
# 2, strict output format loop: thought - action - observation - thought - ... - verdict
# no skipping or creating custom observations, aligning same training data format

import json

TOOL_DEFINITIONS = [
    {
        "name": "lookup_policy",
        "description": "Look up a policyholder's coverage details by their user ID. Always call this first.",
        "parameters": {
            "user_id": {
                "type": "string",
                "description": "The policyholder ID, e.g. 'P-1001'"
            }
        }
    },
    {
        "name": "check_rules",
        "description": "Check if a specific claim type is eligible under the policyholder's plan. Call this after lookup_policy.",
        "parameters": {
            "claim_type": {
                "type": "string",
                "description": "Type of damage: storm, hail, theft, fire, flood, or collision"
            },
            "plan_type": {
                "type": "string",
                "description": "The policyholder's plan: Basic, Standard, or Premium"
            },
            "policy_covers": {
                "type": "list",
                "description": "List of covered claim types from the policy lookup"
            },
            "policy_status": {
                "type": "string",
                "description": "Policy status from the policy lookup: active, lapsed, or cancelled"
            },
            "claims_this_year": {
                "type": "integer",
                "description": "Number of claims already filed this year from the policy lookup"
            }
        }
    },
    {
        "name": "calculate_payout",
        "description": "Calculate the payout for an eligible claim. Only call this if check_rules returns eligible=true.",
        "parameters": {
            "claimed_amount": {
                "type": "float",
                "description": "Amount the policyholder is claiming in dollars"
            },
            "deductible": {
                "type": "float",
                "description": "Policy deductible from the policy lookup"
            },
            "max_single_claim": {
                "type": "float",
                "description": "Maximum allowed for a single claim from check_rules"
            },
            "max_annual_payout": {
                "type": "float",
                "description": "Maximum annual payout from the policy lookup"
            },
            "already_claimed_this_year": {
                "type": "float",
                "description": "Total already claimed this year. Use 0 if unknown.",
                "default": 0
            }
        }
    }
]

TOOL_DEFINITIONS_JSON = json.dumps(TOOL_DEFINITIONS, indent=2)

SYSTEM_PROMPT = f"""You are an expert insurance claims adjuster. You process claims by reasoning step-by-step and calling tools to look up real data.

Available tools:
{TOOL_DEFINITIONS_JSON}

You MUST follow this EXACT format for every step — no exceptions:

Thought: [your reasoning about what to do next]
Action: tool_name({{"param": "value", "param2": "value2"}})

After each Action, you will receive an Observation with the real tool result.
Use that result in your next Thought.

When you have enough information to make a final decision, output:

Verdict: APPROVED or DENIED
Payout: $[amount] (or $0 if denied)
Reasoning: [2-3 sentences citing specific data from the tool results]

Rules:
- Always call lookup_policy first
- Never skip check_rules — always verify eligibility explicitly
- Only call calculate_payout if check_rules returns eligible=true
- Never invent or guess tool results — wait for the Observation
- Never output an Observation yourself
"""

if __name__ == "__main__":
    print("SYSTEM_PROMPT length:", len(SYSTEM_PROMPT), "chars")
    print("Number of tools defined:", len(TOOL_DEFINITIONS))
    print("Tool names:", [t["name"] for t in TOOL_DEFINITIONS])
    print("\nFirst 300 chars of prompt:\n", SYSTEM_PROMPT[:300])