"""
Phase 1, Step 7: essay-level 80/20 train/test split, stratified by topic within
each class label. Locked into dataset/splits.json as the single source of truth.

Records flagged "excluded" (stub-sourced essays with no genuine content - see
build_dataset.py) are filtered out before splitting; they stay in essays.jsonl for
audit purposes but never enter train/test.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "dataset" / "essays.jsonl"
SPLITS_PATH = ROOT / "dataset" / "splits.json"

TEST_FRACTION = 0.2
SEED = 42


# Entry point: excludes stub records, groups the rest by (label, topic), shuffles each
# group with a fixed seed, peels off TEST_FRACTION into test/the rest into train, and
# writes splits.json - the single locked split every later phase reads from.
def main():
    all_records = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in all_records if not r.get("excluded")]
    excluded_ids = [r["id"] for r in all_records if r.get("excluded")]

    print("=" * 60)
    print("STEP 7: TRAIN/TEST SPLIT")
    print("=" * 60)
    print(f"{len(all_records)} total records, {len(excluded_ids)} excluded (stub-sourced): {excluded_ids}")
    print(f"{len(records)} usable records for splitting")

    rng = random.Random(SEED)

    groups = defaultdict(list)
    for r in records:
        groups[(r["label"], r.get("topic") or "other")].append(r["id"])

    train_ids, test_ids = [], []
    single_member_groups = []

    for (label, topic), ids in groups.items():
        ids = ids[:]
        rng.shuffle(ids)
        n_test = round(len(ids) * TEST_FRACTION)
        if len(ids) == 1:
            train_ids.extend(ids)
            single_member_groups.append((label, topic))
            continue
        n_test = max(n_test, 0)
        test_ids.extend(ids[:n_test])
        train_ids.extend(ids[n_test:])

    split = {"train": sorted(train_ids), "test": sorted(test_ids)}
    with SPLITS_PATH.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    print(f"Wrote {SPLITS_PATH.relative_to(ROOT)}")
    print(f"train: {len(train_ids)}, test: {len(test_ids)}")

    by_label = defaultdict(lambda: {"train": 0, "test": 0})
    id_to_label = {r["id"]: r["label"] for r in records}
    for eid in train_ids:
        by_label[id_to_label[eid]]["train"] += 1
    for eid in test_ids:
        by_label[id_to_label[eid]]["test"] += 1

    print("\nSplit counts by class:")
    for label in ("human", "ai", "hybrid"):
        counts = by_label[label]
        print(f"  {label}: train={counts['train']}, test={counts['test']}")

    if single_member_groups:
        print(f"\n{len(single_member_groups)} (label, topic) group(s) had only 1 essay and were kept entirely in train "
              "(can't be split):")
        for label, topic in single_member_groups:
            print(f"  - {label} / {topic}")


if __name__ == "__main__":
    main()
