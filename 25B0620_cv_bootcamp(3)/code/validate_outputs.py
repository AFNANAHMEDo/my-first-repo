from pathlib import Path
import csv, json, re, sys

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "predictions"
PRED = ["t1_clean_ref.csv", "t1_noisy_ce.csv", "t2_early_stop.csv",
        "t5_baseline_synth.csv", "t5_best_synth.csv"]
DYN = ["epoch", "train_acc_correct", "train_acc_mislabeled", "val_acc_noisy", "val_acc_clean"]

def read(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{path.name}: empty")
    return [[x.strip() for x in r] for r in rows if any(x.strip() for x in r)]

def pred_file(path):
    rows = read(path)
    assert [x.lower() for x in rows[0]] == ["id", "pred"], f"{path.name}: header must be id,pred"
    body = rows[1:]
    assert len(body) == 10000, f"{path.name}: expected 10000 rows, got {len(body)}"
    ids, preds = [], []
    for r in body:
        assert len(r) == 2, f"{path.name}: each row must have 2 columns"
        i, p = int(r[0]), int(r[1])
        assert 0 <= i < 10000 and 0 <= p < 10, f"{path.name}: out-of-range id/pred"
        ids.append(i); preds.append(p)
    assert sorted(ids) == list(range(10000)), f"{path.name}: IDs must be 0..9999 exactly once"

def main():
    for name in PRED:
        pred_file(P / name)
    t4 = sorted(x.name for x in P.glob("t4_*.csv"))
    assert 1 <= len(t4) <= 3, "need 1-3 t4_*.csv files"
    for name in t4:
        pred_file(P / name)
    dyn = read(P / "t2_dynamics.csv")
    assert [x.lower() for x in dyn[0]] == DYN, "invalid t2_dynamics.csv header"
    assert 1 <= len(dyn)-1 <= 60, "Task 2 must contain 1-60 epochs"
    epochs = [int(r[0]) for r in dyn[1:]]
    assert epochs in (list(range(len(epochs))), list(range(1, len(epochs)+1))), "epoch sequence has gaps"
    for row in dyn[1:]:
        assert len(row) == 5 and all(0 <= float(v) <= 100 for v in row[1:]), "invalid dynamics row"
    scores = read(P / "t3_noise_scores.csv")
    assert [x.lower() for x in scores[0]] == ["index", "score"], "invalid Task 3 header"
    body = scores[1:]
    idx = [int(r[0]) for r in body]
    assert len(body) == 45000 and len(set(idx)) == 45000, "Task 3 must have 45000 unique indices"
    assert all(0 <= i < 50000 for i in idx), "Task 3 index outside dataset range"
    assert all(len(r) == 2 and float(r[1]) == float(r[1]) for r in body), "invalid Task 3 score"
    res = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    assert res.get("roll_no") and res.get("name"), "results.json identity missing"
    assert (ROOT / "report.md").is_file() and (ROOT / "code/README.md").is_file()
    print("Packaged output-format checks passed. This does not verify model accuracy, training provenance, or official checker compatibility.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(1)
