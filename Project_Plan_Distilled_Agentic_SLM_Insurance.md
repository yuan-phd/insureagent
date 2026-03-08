# Distilled Agentic SLM for Insurance Claims Processing

## Project Summary

**Problem:** Large Language Models (GPT-4, Claude) are too expensive and raise data privacy concerns for regulated industries like insurance and finance. Client data cannot be sent to external APIs.

**Solution:** Build a fully functional agentic claims processing pipeline with real tool execution using a large Teacher model, then distill that capability into a 1.5B parameter Student model that runs locally — fast, cheap, and private.

**Interview Narrative:** "I built an end-to-end agentic insurance claims system where the model autonomously queries a policyholder database, checks coverage rules, and calculates payouts — with every step verifiable. I then distilled that reasoning capability from a large Teacher model into a 1.5B SLM using LoRA, achieving over 90% cost reduction while preserving tool-calling accuracy. The result is deployable on a single laptop with no external API dependency — critical for regulated environments."

---

## Accenture Job Requirement Mapping

| Job Requirement | How This Project Proves It |
|---|---|
| Agentic LLMs & Tool Use | Real tool execution loop with DB, rule engine, calculator |
| Small Language Models (SLMs) | Distilled 1.5B model with LoRA fine-tuning |
| Domain adaptation techniques | General model → Insurance adjuster specialist |
| External knowledge integration | SQLite database, rule engine, structured tool results |
| Combine reasoning, knowledge, and tools | Full ReAct-style chain: Think → Act → Observe → Think |
| Experimental design with baselines | Teacher vs Student comparison on multiple metrics |
| Strong Python / PyTorch skills | Custom PEFT training script (not AutoTrain) |
| Research dissemination | GitHub repo with technical write-up |
| Finance/Insurance domain | End-to-end insurance claims use case |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   CLAIM INPUT                        │
│  "My car was damaged by hail during a storm"         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              AGENTIC REASONING LOOP                  │
│                                                      │
│  STEP 1: Model generates a THOUGHT                   │
│  "I need to look up this policyholder's coverage"    │
│                                                      │
│  STEP 2: Model generates a TOOL CALL                 │
│  Action: lookup_policy(user_id="P-1042")             │
│                                                      │
│  STEP 3: SYSTEM EXECUTES the tool (real code runs)   │
│  Observation: {plan: "Premium", covers: ["storm",    │
│  "hail", "theft"], deductible: 500}                  │
│                                                      │
│  STEP 4: Model generates next THOUGHT                │
│  "Hail is covered. Now I need to check the           │
│   claimed amount and calculate payout."              │
│                                                      │
│  STEP 5: Model generates next TOOL CALL              │
│  Action: check_rules(claim_type="hail",              │
│          plan="Premium")                             │
│                                                      │
│  STEP 6: SYSTEM EXECUTES the tool                    │
│  Observation: {eligible: true, max_payout: 15000,    │
│  requires_inspection: false}                         │
│                                                      │
│  STEP 7: Model generates next TOOL CALL              │
│  Action: calculate_payout(claimed=2200,              │
│          deductible=500, max=15000)                   │
│                                                      │
│  STEP 8: SYSTEM EXECUTES the tool                    │
│  Observation: {payout: 1700}                         │
│                                                      │
│  STEP 9: Model generates FINAL VERDICT               │
│  Verdict: APPROVED | Payout: $1,700                  │
│  Reasoning: Hail damage covered under Premium plan.  │
│  Claimed $2,200 minus $500 deductible = $1,700.      │
└─────────────────────────────────────────────────────┘
```

**Critical distinction from your original plan:** The model does NOT generate fake tool results. It generates the tool call, the system actually executes it against a real database, and the real result is fed back to the model for its next reasoning step. This is what makes it genuinely agentic.

---

## Phase 1: Tool Infrastructure & Teacher Agent (Day 1)

### 1.1 Build the Real Tools

These are simple Python functions backed by real data. This is the foundation everything else depends on.

**Tool 1: Policy Database (SQLite)**

```python
import sqlite3
import json

def create_database():
    conn = sqlite3.connect("insurance.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS policies (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            plan_type TEXT,          -- "Basic", "Standard", "Premium"
            covers TEXT,             -- JSON list: ["storm", "theft", "fire", ...]
            deductible REAL,
            max_annual_payout REAL,
            status TEXT,             -- "active", "lapsed", "cancelled"
            claims_this_year INTEGER
        )
    """)

    # Insert 25-30 sample policyholders with varied profiles
    policies = [
        ("P-1001", "Alice Murphy", "Premium", '["storm","hail","theft","fire","flood","collision"]', 500, 50000, "active", 1),
        ("P-1002", "Brian O'Neill", "Basic", '["fire","theft"]', 1000, 15000, "active", 0),
        ("P-1003", "Ciara Walsh", "Standard", '["storm","fire","theft","collision"]', 750, 30000, "active", 3),
        ("P-1004", "David Chen", "Premium", '["storm","hail","theft","fire","flood","collision"]', 500, 50000, "lapsed", 0),
        # ... add 20+ more with diverse profiles
    ]

    c.executemany("INSERT OR REPLACE INTO policies VALUES (?,?,?,?,?,?,?,?)", policies)
    conn.commit()
    return conn

def lookup_policy(user_id: str) -> dict:
    """Tool 1: Look up a policyholder's coverage details."""
    conn = sqlite3.connect("insurance.db")
    c = conn.cursor()
    c.execute("SELECT * FROM policies WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    if not row:
        return {"error": f"No policy found for {user_id}"}
    return {
        "user_id": row[0], "name": row[1], "plan_type": row[2],
        "covers": json.loads(row[3]), "deductible": row[4],
        "max_annual_payout": row[5], "status": row[6],
        "claims_this_year": row[7]
    }
```

**Tool 2: Rule Engine**

```python
COVERAGE_RULES = {
    "storm":     {"requires_inspection": False, "max_single_claim": 15000, "waiting_period_days": 0},
    "hail":      {"requires_inspection": False, "max_single_claim": 15000, "waiting_period_days": 0},
    "theft":     {"requires_inspection": True,  "max_single_claim": 25000, "waiting_period_days": 30},
    "fire":      {"requires_inspection": True,  "max_single_claim": 40000, "waiting_period_days": 0},
    "flood":     {"requires_inspection": True,  "max_single_claim": 35000, "waiting_period_days": 90},
    "collision": {"requires_inspection": True,  "max_single_claim": 20000, "waiting_period_days": 0},
}

MAX_CLAIMS_PER_YEAR = {"Basic": 2, "Standard": 4, "Premium": 10}

def check_rules(claim_type: str, plan_type: str, policy_covers: list,
                policy_status: str, claims_this_year: int) -> dict:
    """Tool 2: Check if a claim is eligible based on coverage rules."""

    # Check policy is active
    if policy_status != "active":
        return {"eligible": False, "reason": f"Policy status is '{policy_status}'. Must be active."}

    # Check claim type is covered
    if claim_type not in policy_covers:
        return {"eligible": False, "reason": f"'{claim_type}' not covered under this plan."}

    # Check annual claim limit
    max_claims = MAX_CLAIMS_PER_YEAR.get(plan_type, 2)
    if claims_this_year >= max_claims:
        return {"eligible": False, "reason": f"Annual claim limit ({max_claims}) reached."}

    rule = COVERAGE_RULES.get(claim_type)
    if not rule:
        return {"eligible": False, "reason": f"Unknown claim type: {claim_type}"}

    return {
        "eligible": True,
        "max_single_claim": rule["max_single_claim"],
        "requires_inspection": rule["requires_inspection"],
        "waiting_period_days": rule["waiting_period_days"]
    }
```

**Tool 3: Payout Calculator**

```python
def calculate_payout(claimed_amount: float, deductible: float,
                     max_single_claim: float, max_annual_payout: float,
                     already_claimed_this_year: float = 0) -> dict:
    """Tool 3: Calculate the actual payout amount."""

    if claimed_amount <= deductible:
        return {"payout": 0, "reason": f"Claimed amount (${claimed_amount}) does not exceed deductible (${deductible})."}

    after_deductible = claimed_amount - deductible
    capped = min(after_deductible, max_single_claim)
    annual_remaining = max_annual_payout - already_claimed_this_year
    final_payout = min(capped, annual_remaining)

    return {
        "payout": round(final_payout, 2),
        "breakdown": {
            "claimed": claimed_amount,
            "deductible_applied": deductible,
            "after_deductible": after_deductible,
            "single_claim_cap": max_single_claim,
            "annual_remaining": annual_remaining,
            "final_payout": round(final_payout, 2)
        }
    }
```

### 1.2 Define the Tool Schema (for the LLM)

This is what the model sees. It decides which function to call and with what arguments.

```python
TOOL_DEFINITIONS = [
    {
        "name": "lookup_policy",
        "description": "Look up a policyholder's coverage details by their user ID.",
        "parameters": {
            "user_id": {"type": "string", "description": "The policyholder ID, e.g. 'P-1001'"}
        }
    },
    {
        "name": "check_rules",
        "description": "Check if a specific claim type is eligible under the policyholder's plan.",
        "parameters": {
            "claim_type": {"type": "string", "description": "Type of damage: storm, hail, theft, fire, flood, collision"},
            "plan_type": {"type": "string", "description": "The policyholder's plan: Basic, Standard, or Premium"},
            "policy_covers": {"type": "list", "description": "List of covered claim types from the policy"},
            "policy_status": {"type": "string", "description": "Policy status: active, lapsed, or cancelled"},
            "claims_this_year": {"type": "integer", "description": "Number of claims already filed this year"}
        }
    },
    {
        "name": "calculate_payout",
        "description": "Calculate the payout for an eligible claim.",
        "parameters": {
            "claimed_amount": {"type": "float", "description": "Amount the policyholder is claiming"},
            "deductible": {"type": "float", "description": "Policy deductible amount"},
            "max_single_claim": {"type": "float", "description": "Maximum allowed for a single claim of this type"},
            "max_annual_payout": {"type": "float", "description": "Maximum annual payout remaining"},
            "already_claimed_this_year": {"type": "float", "description": "Total already claimed this year", "default": 0}
        }
    }
]
```

### 1.3 Build the Agentic Execution Loop

This is the orchestrator — it feeds the model's output to real tools and loops until the model produces a final verdict.

```python
import re
import json

TOOL_REGISTRY = {
    "lookup_policy": lookup_policy,
    "check_rules": check_rules,
    "calculate_payout": calculate_payout,
}

SYSTEM_PROMPT = """You are an expert insurance claims adjuster. You process claims
by reasoning step-by-step and calling tools to look up information.

Available tools:
{tool_definitions}

You MUST follow this exact format for every step:

Thought: [your reasoning about what to do next]
Action: tool_name({{"param": "value", "param2": "value2"}})

After receiving tool results, continue reasoning. When you have enough
information to make a decision, output:

Verdict: APPROVED or DENIED
Payout: $amount (or $0 if denied)
Reasoning: [2-3 sentence explanation citing specific data from tool results]
"""

def parse_action(text: str):
    """Extract tool name and arguments from model output."""
    match = re.search(r'Action:\s*(\w+)\((.*)\)', text, re.DOTALL)
    if not match:
        return None, None
    tool_name = match.group(1)
    try:
        args = json.loads(match.group(2))
    except json.JSONDecodeError:
        return tool_name, None
    return tool_name, args

def run_agent(claim_text: str, user_id: str, llm_client, max_steps: int = 6):
    """Execute the full agentic loop with real tool calls."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(
            tool_definitions=json.dumps(TOOL_DEFINITIONS, indent=2))},
        {"role": "user", "content": f"Process this claim from policyholder {user_id}: {claim_text}"}
    ]

    trace = []  # Full reasoning trace for distillation

    for step in range(max_steps):
        # Call the LLM
        response = llm_client.generate(messages)
        model_output = response.text

        trace.append({"role": "assistant", "content": model_output})

        # Check if model has reached a verdict
        if "Verdict:" in model_output:
            trace.append({"role": "system", "content": "[END - Verdict reached]"})
            break

        # Parse and execute tool call
        tool_name, args = parse_action(model_output)
        if tool_name and tool_name in TOOL_REGISTRY and args:
            result = TOOL_REGISTRY[tool_name](**args)
            observation = f"Observation: {json.dumps(result, indent=2)}"
        else:
            observation = "Observation: Error - invalid tool call. Check the tool name and argument format."

        trace.append({"role": "user", "content": observation})
        messages.append({"role": "assistant", "content": model_output})
        messages.append({"role": "user", "content": observation})

    return trace
```

### Day 1 Deliverable

A working pipeline: you give it a claim and a user ID, it reasons through multiple steps, calls real tools, and produces a verifiable verdict. Test it on 10 manual claims to make sure it works reliably before moving on.

---

## Phase 2: Synthetic Data Generation (Day 2)

### 2.1 Create Seed Scenarios

Write 10-15 diverse seed claims covering different outcomes. This is critical — your model will only be as good as the variety in your training data.

```python
SEED_CLAIMS = [
    # APPROVALS - straightforward
    {"user_id": "P-1001", "claim": "My car windshield was cracked by hail last Tuesday.", "claimed_amount": 1200},
    {"user_id": "P-1005", "claim": "Kitchen fire caused by electrical fault. Damage to cabinets and appliances.", "claimed_amount": 8500},

    # DENIALS - policy doesn't cover it
    {"user_id": "P-1002", "claim": "Basement flooded during heavy rain.", "claimed_amount": 5000},
    # P-1002 has Basic plan which doesn't cover flood

    # DENIALS - policy lapsed
    {"user_id": "P-1004", "claim": "Car stolen from driveway overnight.", "claimed_amount": 18000},
    # P-1004 has lapsed policy

    # DENIALS - claim limit reached
    {"user_id": "P-1003", "claim": "Storm blew off roof tiles.", "claimed_amount": 3000},
    # P-1003 has 3 claims this year, Standard allows 4 — borderline

    # EDGE CASES - amount below deductible
    {"user_id": "P-1001", "claim": "Minor scratch from parking lot collision.", "claimed_amount": 300},
    # P-1001 has $500 deductible, so payout = $0

    # EDGE CASES - requires inspection
    {"user_id": "P-1006", "claim": "Jewelry stolen during break-in.", "claimed_amount": 12000},

    # EDGE CASES - amount exceeds caps
    {"user_id": "P-1001", "claim": "Major collision damage to vehicle.", "claimed_amount": 35000},
    # Max single collision claim is $20,000
]
```

### 2.2 Generate Variations with Teacher Model

Use DeepSeek-R1 or GPT-4o to generate claim variations, then run each through your real pipeline.

```python
VARIATION_PROMPT = """Generate 20 realistic insurance claim descriptions for the
claim type '{claim_type}'. Each should be 1-2 sentences, written as a
policyholder would describe the incident. Vary the severity, circumstances,
and claimed amounts. Include some ambiguous or tricky cases.

Return as a JSON array of objects with 'claim_text' and 'claimed_amount' fields.
No other text."""

# For each seed, generate 20 variations = 200-300 total claims
# Then run each through your Phase 1 pipeline with the Teacher model
# Capture the full multi-turn trace
```

### 2.3 Structure Training Data as Multi-Turn Conversations

**This is the key format change from your original plan.** Each training example captures the full loop including real tool results.

```json
{
    "conversations": [
        {
            "role": "system",
            "content": "You are an expert insurance claims adjuster..."
        },
        {
            "role": "user",
            "content": "Process this claim from policyholder P-1001: My car windshield was cracked by hail last Tuesday. Claimed amount: $1,200."
        },
        {
            "role": "assistant",
            "content": "Thought: I need to check this policyholder's coverage details to see if hail damage is covered.\nAction: lookup_policy({\"user_id\": \"P-1001\"})"
        },
        {
            "role": "user",
            "content": "Observation: {\"user_id\": \"P-1001\", \"name\": \"Alice Murphy\", \"plan_type\": \"Premium\", \"covers\": [\"storm\", \"hail\", \"theft\", \"fire\", \"flood\", \"collision\"], \"deductible\": 500, \"max_annual_payout\": 50000, \"status\": \"active\", \"claims_this_year\": 1}"
        },
        {
            "role": "assistant",
            "content": "Thought: Policy is active and covers hail damage. I need to check the specific rules for hail claims under the Premium plan.\nAction: check_rules({\"claim_type\": \"hail\", \"plan_type\": \"Premium\", \"policy_covers\": [\"storm\", \"hail\", \"theft\", \"fire\", \"flood\", \"collision\"], \"policy_status\": \"active\", \"claims_this_year\": 1})"
        },
        {
            "role": "user",
            "content": "Observation: {\"eligible\": true, \"max_single_claim\": 15000, \"requires_inspection\": false, \"waiting_period_days\": 0}"
        },
        {
            "role": "assistant",
            "content": "Thought: Claim is eligible. No inspection needed. Now I'll calculate the payout with the deductible applied.\nAction: calculate_payout({\"claimed_amount\": 1200, \"deductible\": 500, \"max_single_claim\": 15000, \"max_annual_payout\": 50000, \"already_claimed_this_year\": 0})"
        },
        {
            "role": "user",
            "content": "Observation: {\"payout\": 700, \"breakdown\": {\"claimed\": 1200, \"deductible_applied\": 500, \"after_deductible\": 700, \"single_claim_cap\": 15000, \"annual_remaining\": 50000, \"final_payout\": 700}}"
        },
        {
            "role": "assistant",
            "content": "Verdict: APPROVED\nPayout: $700\nReasoning: Policyholder P-1001 (Alice Murphy) has an active Premium plan that covers hail damage. The claimed amount of $1,200 minus the $500 deductible yields a payout of $700, well within the $15,000 single-claim cap and $50,000 annual limit."
        }
    ]
}
```

### Target Data Composition

| Category | Count | Purpose |
|---|---|---|
| Clean approvals (various claim types) | 100-120 | Core pattern learning |
| Denials — not covered | 40-50 | Learn to deny correctly |
| Denials — policy lapsed/cancelled | 20-30 | Status checking |
| Denials — claim limit reached | 15-20 | Limit enforcement |
| Edge cases — below deductible | 15-20 | Deductible logic |
| Edge cases — exceeds caps | 15-20 | Cap application |
| Ambiguous/multi-step | 20-30 | Complex reasoning |
| **Total** | **~250-300** | |

### Day 2 Deliverable

A `train.jsonl` file with 250-300 multi-turn conversation traces, all generated by running real claims through your real tool infrastructure with the Teacher model.

---

## Phase 3: Fine-Tuning with LoRA (Day 3)

### 3.1 Training Script (Write This Yourself — Not AutoTrain)

Use HuggingFace PEFT and TRL libraries. This is roughly 60 lines of meaningful code.

```python
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# Load base model
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # or "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto")

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # Rank — 16 is a good balance for this scale
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Should show ~1-2% of total params

# Load your training data
dataset = load_dataset("json", data_files="train.jsonl", split="train")

# Format conversations into the model's chat template
def format_conversation(example):
    return {"text": tokenizer.apply_chat_template(
        example["conversations"], tokenize=False, add_generation_prompt=False
    )}

dataset = dataset.map(format_conversation)

# Training configuration
training_args = TrainingArguments(
    output_dir="./insurance-agent-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
)

# Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
)

trainer.train()
trainer.save_model("./insurance-agent-lora-final")
```

### 3.2 Where to Run This

| Option | Cost | GPU | Time |
|---|---|---|---|
| Google Colab (free) | $0 | T4 (15GB) | 2-3 hours |
| Google Colab Pro | ~$10/month | A100 (40GB) | 30-45 min |
| RunPod | ~$0.50/hour | A10G (24GB) | 1-2 hours |
| Lambda Labs | ~$0.75/hour | A10 (24GB) | 1-2 hours |

A 1.5B model with LoRA fits comfortably on a free Colab T4. No need to spend money unless you want faster iteration.

### Day 3 Deliverable

A trained LoRA adapter saved to HuggingFace Hub (private repo). You can load it in ~10 lines of code on any machine.

---

## Phase 4: Evaluation & Demo (Day 4)

### 4.1 Create a Held-Out Test Set

Set aside 30-40 claims that were NOT in the training data. Include a balanced mix: ~15 approvals, ~10 denials (various reasons), ~5-10 edge cases.

### 4.2 Evaluation Metrics

Run both Teacher and Student through the same test claims using the same real tool infrastructure.

```python
def evaluate_model(model_client, test_claims, label="Model"):
    results = []
    for claim in test_claims:
        trace = run_agent(claim["text"], claim["user_id"], model_client)

        # Extract from trace
        predicted_verdict = extract_verdict(trace)
        predicted_tools = extract_tool_calls(trace)
        predicted_payout = extract_payout(trace)

        results.append({
            "correct_verdict": predicted_verdict == claim["expected_verdict"],
            "correct_tools": predicted_tools == claim["expected_tools"],
            "correct_payout": abs(predicted_payout - claim["expected_payout"]) < 0.01,
            "num_steps": len([t for t in trace if t["role"] == "assistant"]),
            "latency_ms": measure_latency(trace),
        })

    return results
```

### 4.3 Comparison Table (Your Money Slide)

| Metric | Teacher (8B) | Student (1.5B + LoRA) |
|---|---|---|
| Verdict Accuracy | 95%+ (baseline) | Target: >85% |
| Tool Selection Accuracy | 95%+ (baseline) | Target: >85% |
| Payout Calculation Accuracy | 98%+ (baseline) | Target: >90% |
| Average Latency per Claim | ~8-12 seconds | Target: <3 seconds |
| Model Size | ~16 GB | ~3 GB |
| Can Run Locally (No API) | No | Yes |
| Estimated Cost per 1000 Claims | ~$5-10 (API) | ~$0.10 (compute) |

### 4.4 Error Analysis

Don't just report numbers. Categorize where the Student fails:
- Does it call the wrong tool? (tool selection error)
- Does it call the right tool with wrong arguments? (argument error)
- Does it reason correctly but format the verdict wrong? (format error)
- Does it hallucinate tool results instead of waiting for them? (hallucination — most serious)

This analysis shows Accenture you think like a researcher, not just an engineer.

### Day 4 Deliverable

A Jupyter notebook with the full evaluation, comparison tables, and error analysis. This becomes your demo.

---

## Project Repository Structure

```
insurance-claims-agent/
├── README.md                    # Project overview, results summary
├── tools/
│   ├── database.py              # SQLite policyholder DB + lookup_policy()
│   ├── rules.py                 # Rule engine + check_rules()
│   ├── calculator.py            # Payout calculator + calculate_payout()
│   └── tool_definitions.json    # Tool schemas for the LLM
├── agent/
│   ├── agent_loop.py            # ReAct-style execution orchestrator
│   ├── prompts.py               # System prompts for Teacher and Student
│   └── parsers.py               # Output parsing (extract actions, verdicts)
├── data/
│   ├── seed_claims.json         # 10-15 seed scenarios
│   ├── generate_variations.py   # Script to create claim variations via API
│   ├── generate_traces.py       # Script to run claims through Teacher pipeline
│   ├── train.jsonl              # 250-300 training traces
│   └── test.jsonl               # 30-40 held-out test claims
├── training/
│   ├── train_lora.py            # LoRA fine-tuning script (PEFT + TRL)
│   └── config.yaml              # Hyperparameters
├── evaluation/
│   ├── evaluate.py              # Run Teacher vs Student comparison
│   ├── metrics.py               # Accuracy, latency, error categorization
│   └── results_notebook.ipynb   # Final results with visualizations
└── demo/
    └── demo.py                  # Interactive demo: input a claim, see the trace
```

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Teacher model produces inconsistent traces | Bad training data | Validate each trace: does it contain valid tool calls? Does the verdict match what the tools returned? Discard broken traces. |
| Student hallucinates tool results | Demo failure | Add a format check in the agent loop: if the model produces an Observation instead of a Thought/Action, force-stop and flag it. |
| 250 examples isn't enough | Low accuracy | If Student accuracy is below 80%, generate 200 more examples focused on failure categories from error analysis. |
| LoRA fine-tune diverges | Wasted GPU time | Use conservative learning rate (2e-4), check loss curve every 50 steps. If loss spikes, reduce LR or increase warmup. |
| Pipeline works but demo is boring | Weak interview impact | Build a simple CLI or Streamlit demo that shows the reasoning trace step-by-step with color-coded tool calls and results. |

---

## Estimated Costs

| Item | Cost |
|---|---|
| Teacher API calls (300 traces × ~2K tokens each) | ~$3-8 |
| Claim variation generation | ~$2-5 |
| GPU for LoRA training (Colab free tier) | $0 |
| Total | **~$5-13** |

---

## What to Say in the Interview

**On Agentic Systems:** "The model doesn't just generate text that looks like tool calls — it actually integrates with a real database and rule engine. Each tool call executes, and the real result feeds back into the model's reasoning. Every step in the chain is verifiable against ground truth."

**On Distillation:** "I used the large Teacher model to generate structured reasoning traces, then trained a 1.5B Student model to replicate that reasoning pattern using LoRA. The Student achieves [X]% of the Teacher's accuracy at a fraction of the cost and latency."

**On Why SLMs Matter:** "In regulated industries, you can't always send client data to an external API. A compact model that runs on-premise or on edge devices solves both the cost and the compliance problem simultaneously."

**On Experimental Rigour:** "I held out 40 test claims, measured tool selection accuracy, verdict accuracy, and payout correctness separately, and did an error analysis categorizing where the Student model fails compared to the Teacher."
