# tool to calculate final payout, including 4 steps
# deductibile, single payout limit, remaining annual limit, min value taken as final payout

def calculate_payout(claimed_amount: float, deductible: float,
                     max_single_claim: float, max_annual_payout: float,
                     already_claimed_this_year: float = 0) -> dict:
    """Calculate the actual payout for an eligible claim."""

    if claimed_amount <= deductible:
        return {
            "payout": 0,
            "reason": f"Claimed amount (${claimed_amount}) does not exceed deductible (${deductible})."
        }

    after_deductible  = claimed_amount - deductible
    capped_by_single  = min(after_deductible, max_single_claim)
    annual_remaining  = max_annual_payout - already_claimed_this_year
    final_payout      = min(capped_by_single, annual_remaining)
    final_payout      = round(final_payout, 2)

    return {
        "payout": final_payout,
        "breakdown": {
            "claimed_amount":          claimed_amount,
            "deductible_applied":      deductible,
            "after_deductible":        round(after_deductible, 2),
            "single_claim_cap":        max_single_claim,
            "annual_remaining":        round(annual_remaining, 2),
            "final_payout":            final_payout,
        }
    }


if __name__ == "__main__":
    # Test 1: normal claim
    print("Test 1 (normal):", calculate_payout(
        claimed_amount=1200, deductible=500,
        max_single_claim=15000, max_annual_payout=50000
    ))

    # Test 2: below deductible
    print("Test 2 (below deductible):", calculate_payout(
        claimed_amount=300, deductible=500,
        max_single_claim=15000, max_annual_payout=50000
    ))

    # Test 3: exceeds single claim cap
    print("Test 3 (exceeds cap):", calculate_payout(
        claimed_amount=35000, deductible=500,
        max_single_claim=20000, max_annual_payout=50000
    ))

    # Test 4: annual limit nearly exhausted
    print("Test 4 (annual limit):", calculate_payout(
        claimed_amount=5000, deductible=500,
        max_single_claim=15000, max_annual_payout=50000,
        already_claimed_this_year=48000
    ))