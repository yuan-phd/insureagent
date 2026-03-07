# define insurance rules: coverage type and max claim time

COVERAGE_RULES = {
    "storm":     {"requires_inspection": False, "max_single_claim": 15000, "waiting_period_days": 0},
    "hail":      {"requires_inspection": False, "max_single_claim": 15000, "waiting_period_days": 0},
    "theft":     {"requires_inspection": True,  "max_single_claim": 25000, "waiting_period_days": 30},
    "fire":      {"requires_inspection": True,  "max_single_claim": 40000, "waiting_period_days": 0},
    "flood":     {"requires_inspection": True,  "max_single_claim": 35000, "waiting_period_days": 90},
    "collision": {"requires_inspection": True,  "max_single_claim": 20000, "waiting_period_days": 0},
}

MAX_CLAIMS_PER_YEAR = {
    "Basic":    2,
    "Standard": 4,
    "Premium":  10,
}

def check_rules(claim_type: str, plan_type: str, policy_covers: list,
                policy_status: str, claims_this_year: int) -> dict:
    """Check if a claim is eligible based on policy rules."""

    if policy_status != "active":
        return {"eligible": False, "reason": f"Policy status is '{policy_status}'. Must be active to file a claim."}

    if claim_type not in policy_covers:
        return {"eligible": False, "reason": f"Claim type '{claim_type}' is not covered under this policy."}

    max_claims = MAX_CLAIMS_PER_YEAR.get(plan_type, 2)
    if claims_this_year >= max_claims:
        return {"eligible": False, "reason": f"Annual claim limit of {max_claims} reached for {plan_type} plan."}

    rule = COVERAGE_RULES.get(claim_type)
    if not rule:
        return {"eligible": False, "reason": f"Unknown claim type: '{claim_type}'."}

    return {
        "eligible":             True,
        "max_single_claim":     rule["max_single_claim"],
        "requires_inspection":  rule["requires_inspection"],
        "waiting_period_days":  rule["waiting_period_days"],
    }


if __name__ == "__main__":
    # Test 1: valid hail claim, Premium active policy
    print("Test 1 (valid hail):", check_rules(
        claim_type="hail",
        plan_type="Premium",
        policy_covers=["storm", "hail", "theft", "fire", "flood", "collision"],
        policy_status="active",
        claims_this_year=1
    ))

    # Test 2: lapsed policy
    print("Test 2 (lapsed):", check_rules(
        claim_type="storm",
        plan_type="Premium",
        policy_covers=["storm", "hail"],
        policy_status="lapsed",
        claims_this_year=0
    ))

    # Test 3: claim type not covered
    print("Test 3 (not covered):", check_rules(
        claim_type="flood",
        plan_type="Basic",
        policy_covers=["fire", "theft"],
        policy_status="active",
        claims_this_year=0
    ))

    # Test 4: annual limit reached
    print("Test 4 (limit reached):", check_rules(
        claim_type="fire",
        plan_type="Standard",
        policy_covers=["storm", "fire", "theft", "collision"],
        policy_status="active",
        claims_this_year=4
    ))