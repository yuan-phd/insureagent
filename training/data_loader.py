"""
training/data_loader.py
Load and format training data for InsureAgent LoRA fine-tuning.

Supports two backends:
  - local:      reads from a local .jsonl file (default, used in Colab)
  - databricks: reads from a Delta table via PySpark (production)

Backend is selected via the --backend argument in train.py.
Switch requires no code change — only config.yaml and --backend flag.
"""

import json
from collections import Counter
from datasets import Dataset


# ── BACKEND: LOCAL ────────────────────────────────────────────────────────────

def _load_from_local(path: str) -> list[dict]:
    """Load traces from a local .jsonl file."""
    with open(path) as f:
        data = [json.loads(line) for line in f if line.strip()]
    print(f"[local] Loaded {len(data)} traces from {path}")
    return data


# ── BACKEND: DATABRICKS ───────────────────────────────────────────────────────

def _load_from_databricks(delta_path: str) -> list[dict]:
    """
    Load traces from a Databricks Delta table via PySpark.

    Expects the Delta table to have a column 'conversations' containing
    JSON-serialised conversation lists (same schema as local .jsonl).

    Requires:
        - PySpark session active (automatically available on Databricks cluster)
        - delta_path set in config.yaml under data.delta_path
          e.g. "dbfs:/insureagent/train/v2"

    To write training data to Delta (one-time setup):
        spark.createDataFrame(data).write.format("delta").save(delta_path)
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise ImportError(
            "PySpark not available. "
            "Use --backend local for Colab/local environments, "
            "or run on a Databricks cluster for --backend databricks."
        )

    spark = SparkSession.builder.getOrCreate()
    print(f"[databricks] Reading Delta table: {delta_path}")

    df = spark.read.format("delta").load(delta_path)
    rows = df.collect()

    data = []
    for row in rows:
        conversations = row["conversations"]
        # Delta stores the list as a string if written from jsonl
        if isinstance(conversations, str):
            conversations = json.loads(conversations)
        data.append({"conversations": conversations})

    print(f"[databricks] Loaded {len(data)} traces from {delta_path}")
    return data


# ── SHARED UTILITIES ──────────────────────────────────────────────────────────

def get_verdict_distribution(data: list[dict]) -> dict:
    """Count APPROVED / DENIED distribution in a trace list."""
    verdicts = Counter()
    for d in data:
        for msg in reversed(d["conversations"]):
            if msg["role"] == "assistant" and "Verdict:" in msg["content"]:
                if "APPROVED" in msg["content"]:
                    verdicts["APPROVED"] += 1
                elif "DENIED" in msg["content"]:
                    verdicts["DENIED"] += 1
                break
    return dict(verdicts)


def _format_trace(example: dict, tokenizer) -> dict:
    """Apply chat template to a single trace."""
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def build_dataset(cfg: dict, tokenizer, backend: str = "local") -> Dataset:
    """
    Load traces and return a HuggingFace Dataset formatted for SFT.

    Args:
        cfg:      full config dict loaded from config.yaml
        tokenizer: HuggingFace tokenizer (used for chat template)
        backend:  "local" (default) or "databricks"

    Returns:
        HuggingFace Dataset with a "text" column ready for SFTTrainer
    """
    if backend == "databricks":
        data = _load_from_databricks(cfg["data"]["delta_path"])
    elif backend == "local":
        data = _load_from_local(cfg["data"]["train_path"])
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'local' or 'databricks'.")

    dist = get_verdict_distribution(data)
    print(f"Verdict distribution: {dist}")

    dataset = Dataset.from_list(data)
    dataset = dataset.map(lambda ex: _format_trace(ex, tokenizer))
    return dataset