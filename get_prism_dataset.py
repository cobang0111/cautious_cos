from datasets import load_dataset
import os

out_dir = "data/prism_raw"
os.makedirs(out_dir, exist_ok=True)

survey_ds = load_dataset("HannahRoseKirk/prism-alignment", "survey")
conv_ds = load_dataset("HannahRoseKirk/prism-alignment", "conversations")

survey_ds["train"].to_json(
    f"{out_dir}/survey.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)

conv_ds["train"].to_json(
    f"{out_dir}/conversations.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)

print("saved:", out_dir)