# manual test cases not in training set or gpt
# avoid leakage

TEST_CASES = [

    # ── APPROVALS ────────────────────────────────────────────────
    {
        "user_id": "P-1021",
        "claim_text": "Strong winds from a storm knocked over my garden fence and damaged my shed roof.",
        "claimed_amount": 2800,
        "expected_verdict": "APPROVED",
        "expected_payout": 2050,  # 2800 - 750 deductible
        "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"],
        "denial_reason": None
    },
    {
        "user_id": "P-1024",
        "claim_text": "Hailstorm caused multiple dents on my car bonnet and cracked the windshield.",
        "claimed_amount": 3200,
        "expected_verdict": "APPROVED",
        "expected_payout": 2700,  # 3200 - 500 deductible
        "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"],
        "denial_reason": None
    },
    {
        "user_id": "P-1030",
        "claim_text": "An electrical fire in my garage destroyed my car and damaged the walls.",
        "claimed_amount": 22000,
        "expected_verdict": "APPROVED",
        "expected_payout": 21500,  # 22000 - 500, under 40000 cap
        "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"],
        "denial_reason": None
    },
    {
        "user_id": "P-1017",
        "claim_text": "My bicycle and laptop were stolen from my car while parked downtown.",
        "claimed_amount": 4500,
        "expected_verdict": "APPROVED",
        "expected_payout": 3750,  # 4500 - 750 deductible
        "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"],
        "denial_reason": None
    },
    {
        "user_id": "P-1029",
        "claim_text": "Storm damaged my roof and water leaked into the bedroom causing ceiling damage.",
        "claimed_amount": 5500,
        "expected_verdict": "APPROVED",
        "expected_payout": 4750,  # 5500 - 750 deductible
        "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"],
        "denial_reason": None
    },
    {
        "user_id": "P-1009",
        "claim_text": "Heavy flooding in my area caused water to enter my ground floor rooms.",
        "claimed_amount": 12000,
        "expected_verdict": "APPROVED",
        "expected_payout": 11500,  # 12000 - 500 deductible
        "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"],
        "denial_reason": None
    },
    {
        "user_id": "P-1012",
        "claim_text": "A collision on the motorway caused significant damage to the front of my car.",
        "claimed_amount": 8000,
        "expected_verdict": "APPROVED",
        "expected_payout": 7500,  # 8000 - 500 deductible
        "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"],
        "denial_reason": None
    },

    # ── DENIALS: NOT COVERED ──────────────────────────────────────
    {
        "user_id": "P-1019",
        "claim_text": "My basement flooded after a heavy rainfall ruined my furniture.",
        "claimed_amount": 6000,
        "expected_verdict": "DENIED",
        "expected_payout": 0,
        "expected_tools": ["lookup_policy", "check_rules"],
        "denial_reason": "not_covered"
    },
    {
        "user_id": "P-1010",
        "claim_text": "Storm winds damaged my garden fence and outdoor furniture.",
        "claimed_amount": 1800,
        "expected_verdict": "DENIED",
        "expected_payout": 0,
        "expected_tools": ["lookup_policy", "check_rules"],
        "denial_reason": "not_covered"
    },
    {
        "user_id": "P-1025",
        "claim_text": "A hailstorm dented my car roof and cracked a window.",
        "claimed_amount": 2200,
        "expected_verdict": "DENIED",
        "expected_payout": 0,
        "expected_tools": ["lookup_policy", "check_rules"],
        "denial_reason": "not_covered"
    },

    # ── DENIALS: POLICY LAPSED OR CANCELLED ──────────────────────
    {
        "user_id": "P-1013",
        "claim_text": "A fire in my kitchen caused damage to the cabinets and appliances.",
        "claimed_amount": 7000,
        "expected_verdict": "DENIED",
        "expected_payout": 0,
        "expected_tools": ["lookup_policy", "check_rules"],
        "denial_reason": "policy_inactive"
    },
    {
        "user_id": "P-1022",
        "claim_text": "My car was stolen from the street outside my house last night.",
        "claimed_amount": 15000,
        "expected_verdict": "DENIED",
        "expected_payout": 0,
        "expected_tools": ["lookup_policy", "check_rules"],
        "denial_reason": "policy_inactive"
    },
    {
        "user_id": "P-1028",
        "claim_text": "Storm damage cracked several roof tiles and broke a skylight.",
        "claimed_amount": 3500,
        "expected_verdict": "DENIED",
        "expected_payout": 0,
        "expected_tools": ["lookup_policy", "check_rules"],
        "denial_reason": "policy_inactive"
    },

    # ── DENIALS: ANNUAL LIMIT REACHED ────────────────────────────
    {
        "user_id": "P-1011",
        "claim_text": "Hail damaged my car windshield during a storm yesterday.",
        "claimed_amount": 1500,
        "expected_verdict": "DENIED",
        "expected_payout": 0,
        "expected_tools": ["lookup_policy", "check_rules"],
        "denial_reason": "claim_limit"
    },

    # ── EDGE: BELOW DEDUCTIBLE ────────────────────────────────────
    {
        "user_id": "P-1021",
        "claim_text": "A small hail shower left a few minor dents on my car door.",
        "claimed_amount": 400,
        "expected_verdict": "APPROVED",
        "expected_payout": 0,  # below 750 deductible
        "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"],
        "denial_reason": None
    },
    {
        "user_id": "P-1001",
        "claim_text": "Minor scratches on my bumper from a low speed parking collision.",
        "claimed_amount": 350,
        "expected_verdict": "APPROVED",
        "expected_payout": 0,  # below 500 deductible
        "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"],
        "denial_reason": None
    },

    # ── EDGE: EXCEEDS SINGLE CLAIM CAP ───────────────────────────
    {
        "user_id": "P-1030",
        "claim_text": "A major collision on the motorway totalled my car completely.",
        "claimed_amount": 45000,
        "expected_verdict": "APPROVED",
        "expected_payout": 19500,  # capped at 20000 - 500 deductible
        "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"],
        "denial_reason": None
    },
    {
        "user_id": "P-1009",
        "claim_text": "Severe flooding destroyed all ground floor furniture and flooring.",
        "claimed_amount": 60000,
        "expected_verdict": "APPROVED",
        "expected_payout": 34500,  # capped at 35000 - 500 deductible
        "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"],
        "denial_reason": None
    },

    # ── EDGE: NEAR ANNUAL PREMIUM LIMIT ──────────────────────────
    {
        "user_id": "P-1015",
        "claim_text": "Storm damaged my conservatory roof and cracked several windows.",
        "claimed_amount": 4000,
        "expected_verdict": "APPROVED",
        "expected_payout": 3250,  # 4000 - 750 deductible, P-1015 has 9 claims (Premium allows 10)
        "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"],
        "denial_reason": None
    },
]

if __name__ == "__main__":
    from collections import Counter
    verdicts = Counter(t["expected_verdict"] for t in TEST_CASES)
    denial_reasons = Counter(t["denial_reason"] for t in TEST_CASES if t["denial_reason"])
    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"Verdicts: {dict(verdicts)}")
    print(f"Denial reasons: {dict(denial_reasons)}")