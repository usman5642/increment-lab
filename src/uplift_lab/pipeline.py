"""Run the locked independent-holdout analysis, refusing result overwrite."""

from pathlib import Path
import argparse, hashlib, json, platform, time
import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from .data import load, FEATURES
from .models import fit_candidates, score
from .metrics import aipw, transformed_outcome, top_policy, bootstrap_policies, curve


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")


def heuristic(frame):
    rank = (
        frame.reset_index(drop=True)
        .sort_values(
            ["mens", "recency", "history"],
            ascending=[False, True, False],
            kind="stable",
        )
        .index.to_numpy()
    )
    s = np.zeros(len(frame))
    s[rank] = np.arange(len(frame), 0, -1)
    return s


def run(root, output):
    root = Path(root)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists() or (out / "evaluation_started.json").exists():
        raise FileExistsError(
            "Use a new output directory for exact reproduction; never overwrite an evaluation."
        )
    plan = json.loads((root / "analysis_plan.json").read_text())
    seed = plan["seed"]
    parts, audit = load(root, seed)
    train, test = parts["train"], parts["test"]
    rates = train.groupby("treatment").visit.mean()
    nt = len(test)
    se = np.sqrt(0.4 * 0.6 * (rates[0] + rates[1]) / (nt / 2))
    power = {
        "training_arm_rates": {str(k): float(v) for k, v in rates.items()},
        "approximate_mdd_per_1000": float(
            1000 * (norm.ppf(0.975) + norm.ppf(0.8)) * se
        ),
        "method": "Approximate HT contrast variance under near-uniform outcome rates across targeting groups; not exact AIPW power",
        "alpha": 0.05,
        "power": 0.8,
    }
    code = {p.name: digest(p) for p in (root / "src/uplift_lab").glob("*.py")}
    lock = {
        "plan_sha256": digest(root / "analysis_plan.json"),
        "code_sha256": code,
        "data_sha256": audit["sha256"],
        "seed": seed,
        "train_ids_sha256": hashlib.sha256(
            train.source_row.to_numpy().tobytes()
        ).hexdigest(),
        "test_ids_sha256": hashlib.sha256(
            test.source_row.to_numpy().tobytes()
        ).hexdigest(),
        "created_before_fit": True,
        "power": power,
        "python": platform.python_version(),
    }
    write(out / "plan_lock.json", lock)
    write(out / "data_audit.json", audit)
    print("Plan and power note locked before fitting.", flush=True)
    started = time.perf_counter()
    with threadpool_limits(limits=3):
        bundle = fit_candidates(train, seed)
    (root / "models").mkdir(exist_ok=True)
    joblib.dump(bundle, root / "models/frozen.joblib")
    write(
        out / "evaluation_started.json",
        {
            "model_sha256": digest(root / "models/frozen.joblib"),
            "plan_sha256": lock["plan_sha256"],
            "models_fitted_on": "50% training rows only",
        },
    )
    predictions = score(bundle, test)
    mu0 = predictions.pop("mu0")
    mu1 = predictions.pop("mu1")
    predictions["Heuristic"] = heuristic(test)
    y = test.visit.to_numpy()
    t = test.treatment.to_numpy()
    dr = aipw(y, t, mu0, mu1)
    ht = transformed_outcome(y, t)
    # Secondary conversion and spend use design-IPW, not visit nuisance probabilities.
    signals = np.c_[
        dr,
        ht,
        transformed_outcome(test.conversion, t),
        transformed_outcome(test.spend, t),
    ]
    print("Models frozen. Computing held-out paired uncertainty.", flush=True)
    names, observed, draws = bootstrap_policies(
        predictions, t, signals, draws=2000, seed=seed + 8
    )

    def ci(values, point):
        lo, hi = np.quantile(values, [0.025, 0.975])
        return {
            "estimate": float(point * 1000),
            "low": float(lo * 1000),
            "high": float(hi * 1000),
        }

    rows = []
    for j, name in enumerate(names):
        rows.append(
            {
                "policy": name,
                **ci(draws[:, j, 0], observed[j, 0]),
                "ht_estimate": float(observed[j, 1] * 1000),
                "conversion": ci(draws[:, j, 2], observed[j, 2]),
                "spend": ci(draws[:, j, 3], observed[j, 3]),
            }
        )
    primary_idx = names.index("T-logistic")
    comparisons = []
    for comparator in ["Random", "Response ranking", "Heuristic"]:
        k = names.index(comparator)
        comparisons.append(
            {
                "comparator": comparator,
                "confirmatory": comparator == "Random",
                **ci(
                    draws[:, primary_idx, 0] - draws[:, k, 0],
                    observed[primary_idx, 0] - observed[k, 0],
                ),
            }
        )
    budgets = []
    for q in np.linspace(0, 1, 21):
        rec = {"capacity": float(q), "confirmatory_capacity": bool(np.isclose(q, 0.4))}
        for name, s in predictions.items():
            rec[name] = float(np.mean(top_policy(s, q) * dr) * 1000)
        rec["Random"] = float(int(np.floor(nt * q + 0.5)) / nt * dr.mean() * 1000)
        budgets.append(rec)
    sorted_idx = np.argsort(-predictions["T-logistic"], kind="stable")
    bins = []
    for b, idx in enumerate(np.array_split(sorted_idx, 5)):
        z = dr[idx]
        sem = z.std(ddof=1) / np.sqrt(len(idx))
        bins.append(
            {
                "quintile": b + 1,
                "rows": len(idx),
                "predicted": float(predictions["T-logistic"][idx].mean() * 1000),
                "observed_aipw": float(z.mean() * 1000),
                "low": float((z.mean() - 1.96 * sem) * 1000),
                "high": float((z.mean() + 1.96 * sem) * 1000),
            }
        )
    _, _, area = curve(predictions["T-logistic"], y, t)
    result = {
        "project": "Increment Lab",
        "design": plan["design"],
        "units": "incremental visits per 1000 eligible customers",
        "capacity": 0.4,
        "primary_learner": "T-logistic",
        "evaluation_rows": nt,
        "training_rows": len(train),
        "primary": comparisons[0],
        "comparisons": comparisons,
        "policies": rows,
        "budgets": budgets,
        "quintiles": bins,
        "training_balance": audit["training_balance"],
        "power": power,
        "centred_ht_gain_area": area,
        "uncertainty": "95% arm-stratified percentile bootstrap; 2000 paired draws with re-ranking; conditional on fixed models; excludes training variability. Secondary intervals are pointwise, not multiplicity-adjusted.",
        "secondary_units": {
            "conversion": "incremental conversions per 1000 eligible customers",
            "spend": "incremental recorded spend currency units per 1000 eligible customers; not profit",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "data_sha256": audit["sha256"],
        "model_sha256": digest(root / "models/frozen.joblib"),
        "limitations": [
            "Historical 2008 email experiment; no claim about present-day campaigns.",
            "No individual counterfactual effect is observed.",
            "The 40% capacity is illustrative, not a tuned budget.",
            "No campaign costs or margins; no actual ROI claim.",
            "New geographic or temporal populations require new validation.",
        ],
    }
    write(out / "results.json", result)
    pd.DataFrame(
        [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows]
    ).to_csv(out / "policy_results.csv", index=False)
    write(
        out / "completion.json",
        {
            "results_sha256": digest(out / "results.json"),
            "status": "completed",
            "no_model_selected_on_test": True,
        },
    )
    print(
        json.dumps(
            {
                "primary": result["primary"],
                "rows": nt,
                "seconds": result["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--output", default="reports/evaluation")
    a = p.parse_args()
    run(a.root, a.output)
