"""Fetch the latest results and line-ups, refit the model and write the pages into the personal site."""
import shutil
from pathlib import Path

import numpy as np

import fetch
import model
import report

HERE = Path(__file__).parent
SITE = HERE.parent  # the personal-site repo this generator lives in
OUT = SITE / "maly-fotbal"


def main():
    print("Downloading results and line-ups...")
    matches = fetch.load_matches()
    ids, names = model.teams_of(matches)
    print(f"  {sum(m['hg'] is not None for m in matches)} of {len(matches)} matches played")

    print("Fitting model and simulating the season...")
    forecast = model.season_forecast(matches, ids)
    effects = model.attendance_effects(forecast["fit"])
    print("Running walk-forward check...")
    backtest = model.backtest(matches, ids)

    (OUT / "model").mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(report.render_forecast(matches, ids, names, forecast), encoding="utf-8")
    (OUT / "model" / "index.html").write_text(
        report.render_model(matches, ids, names, forecast, effects, backtest), encoding="utf-8")
    shutil.copy(HERE / "static" / "maly-fotbal.css", OUT / "maly-fotbal.css")

    print("\nPredicted final order (expected points):")
    for rank, k in enumerate(np.argsort(-forecast["exp_points"]), 1):
        print(f"  {rank:2d}. {names[ids[k]]:<22s} {forecast['exp_points'][k]:5.1f}")
    print(f"\nWrote {OUT}/ (index.html, model/index.html, maly-fotbal.css)")
    print(f"Nothing is committed: review and push the site repo yourself.")


if __name__ == "__main__":
    main()
