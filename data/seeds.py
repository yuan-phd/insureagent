# setup 15 seed claim scenarios
# these serve as master templates for data generation. each seed is expanded by GPT into 15-20 variations.

SEED_CLAIMS = [

    # ── CLEAN APPROVALS ──────────────────────────────────────────
    {
        "user_id": "P-1001",
        "claim_text": "My car windshield was cracked by hail during a storm last Tuesday.",
        "claimed_amount": 1200,
        "expected_verdict": "APPROVED",
        "claim_type": "hail"
    },
    {
        "user_id": "P-1005",
        "claim_text": "A kitchen fire caused by an electrical fault damaged my cabinets and appliances.",
        "claimed_amount": 8500,
        "expected_verdict": "APPROVED",
        "claim_type": "fire"
    },
    {
        "user_id": "P-1009",
        "claim_text": "Heavy storm winds tore off part of my roof and broke three windows.",
        "claimed_amount": 6200,
        "expected_verdict": "APPROVED",
        "claim_type": "storm"
    },
    {
        "user_id": "P-1006",
        "claim_text": "My car was stolen from the office car park overnight.",
        "claimed_amount": 18000,
        "expected_verdict": "APPROVED",
        "claim_type": "theft"
    },

    # ── DENIALS: CLAIM TYPE NOT COVERED ──────────────────────────
    {
        "user_id": "P-1002",
        "claim_text": "My basement flooded after heavy rainfall and damaged my furniture.",
        "claimed_amount": 5000,
        "expected_verdict": "DENIED",
        "claim_type": "flood"
    },
    {
        "user_id": "P-1007",
        "claim_text": "Storm damage blew tiles off my roof.",
        "claimed_amount": 3200,
        "expected_verdict": "DENIED",
        "claim_type": "storm"
    },

    # ── DENIALS: POLICY LAPSED OR CANCELLED ──────────────────────
    {
        "user_id": "P-1004",
        "claim_text": "My car was stolen from the driveway overnight.",
        "claimed_amount": 18000,
        "expected_verdict": "DENIED",
        "claim_type": "theft"
    },
    {
        "user_id": "P-1008",
        "claim_text": "A fire in my kitchen caused significant damage to the countertops.",
        "claimed_amount": 4500,
        "expected_verdict": "DENIED",
        "claim_type": "fire"
    },

    # ── DENIALS: ANNUAL CLAIM LIMIT REACHED ──────────────────────
    {
        "user_id": "P-1011",
        "claim_text": "Storm damage cracked several roof tiles and broke a window.",
        "claimed_amount": 2800,
        "expected_verdict": "DENIED",
        "claim_type": "storm"
    },

    # ── EDGE CASES: AMOUNT BELOW DEDUCTIBLE ──────────────────────
    {
        "user_id": "P-1001",
        "claim_text": "A minor scratch on my car door from a parking lot incident.",
        "claimed_amount": 300,
        "expected_verdict": "APPROVED",
        "claim_type": "collision"
    },
    {
        "user_id": "P-1005",
        "claim_text": "Small hail dents on the bonnet of my car.",
        "claimed_amount": 400,
        "expected_verdict": "APPROVED",
        "claim_type": "hail"
    },

    # ── EDGE CASES: AMOUNT EXCEEDS SINGLE CLAIM CAP ──────────────
    {
        "user_id": "P-1001",
        "claim_text": "A serious collision on the motorway caused major damage to my vehicle.",
        "claimed_amount": 35000,
        "expected_verdict": "APPROVED",
        "claim_type": "collision"
    },
    {
        "user_id": "P-1009",
        "claim_text": "A large fire destroyed most of my home interior.",
        "claimed_amount": 80000,
        "expected_verdict": "APPROVED",
        "claim_type": "fire"
    },

    # ── EDGE CASES: NEAR ANNUAL LIMIT ────────────────────────────
    {
        "user_id": "P-1015",
        "claim_text": "Storm damage to my garden shed and fencing.",
        "claimed_amount": 3000,
        "expected_verdict": "APPROVED",
        "claim_type": "storm"
    },
    {
        "user_id": "P-1027",
        "claim_text": "Hail cracked two windows and dented my car roof.",
        "claimed_amount": 2200,
        "expected_verdict": "APPROVED",
        "claim_type": "hail"
    },
]

if __name__ == "__main__":
    from collections import Counter
    verdicts = Counter(s["expected_verdict"] for s in SEED_CLAIMS)
    claim_types = Counter(s["claim_type"] for s in SEED_CLAIMS)
    print(f"Total seeds: {len(SEED_CLAIMS)}")
    print(f"Verdicts: {dict(verdicts)}")
    print(f"Claim types: {dict(claim_types)}")