"""Download or reproduce the fixed Increment Lab analysis in an isolated directory."""
import argparse, json, math, shutil
from pathlib import Path
from uplift_lab.data import download

ROOT = Path(__file__).resolve().parent
FIELDS = ["primary", "comparisons", "policies", "budgets", "quintiles"]

def equivalent(a, b):
    if isinstance(b, dict):
        return isinstance(a, dict) and a.keys() == b.keys() and all(equivalent(a[k], v) for k, v in b.items())
    if isinstance(b, list):
        return isinstance(a, list) and len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if isinstance(b, float):
        return isinstance(a, (int, float)) and not isinstance(a, bool) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return type(a) is type(b) and a == b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "reproduce"])
    parser.add_argument("--tag", default="verification")
    args = parser.parse_args()
    if args.command == "download":
        _, record = download(ROOT)
        print(json.dumps(record, indent=2))
        return
    if not args.tag.replace("-", "").replace("_", "").isalnum():
        parser.error("Use an alphanumeric tag")
    target = ROOT / "reproductions" / args.tag
    if target.exists():
        parser.error("This reproduction already exists. Choose a new tag to preserve it.")
    if not (ROOT / "data/raw/hillstrom.csv.gz").exists():
        parser.error("Run python run.py download first")
    target.mkdir(parents=True)
    shutil.copytree(ROOT / "src", target / "src")
    shutil.copy2(ROOT / "analysis_plan.json", target / "analysis_plan.json")
    (target / "data/raw").mkdir(parents=True)
    shutil.copy2(ROOT / "data/raw/hillstrom.csv.gz", target / "data/raw/hillstrom.csv.gz")
    from uplift_lab.pipeline import run
    run(target, target / "reports/evaluation")
    expected = json.loads((ROOT / "reports/evaluation/results.json").read_text())
    actual = json.loads((target / "reports/evaluation/results.json").read_text())
    checks = {key: equivalent(actual[key], expected[key]) for key in FIELDS}
    (target / "comparison.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit("Mismatch. Preserved the output for investigation.")

if __name__ == "__main__":
    main()
