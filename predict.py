"""Standalone scoring: load a saved uplift ensemble and produce, for every
e-mail, whether to send an uplift (discount) mailing.

This script does NOT train anything -- it reuses models persisted by
``script.py`` (``models/uplift_ensemble.joblib``), so scoring can be run
independently and repeatedly on fresh data.

Usage
-----
    python predict.py                                   # uses ./data + saved model
    python predict.py --comm comm.csv --purch purchases.csv \
                      --models models/uplift_ensemble.joblib \
                      --out outputs/send_recommendations.csv

The output CSV has one row per e-mail:
    email, mean_predicted_uplift, has_upcoming_trip, recommend_send

Clients who already bought a tour but have not travelled yet (trip starts on/
after the cutoff) are excluded from the recommendation (recommend_send=False).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Reuse the exact data-loading / feature / recommendation logic from the trainer
# so scoring stays perfectly consistent with training.
import script as trainer  # noqa: E402
from features import build_dataset  # noqa: E402
from uplift_model import load_ensemble  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comm", default=None, help="communications CSV path")
    parser.add_argument("--purch", default=None, help="purchases CSV path")
    parser.add_argument("--models", default=os.path.join(trainer.MODEL_DIR,
                                                         "uplift_ensemble.joblib"),
                        help="path to the saved ensemble")
    parser.add_argument("--out", default=os.path.join(trainer.OUT_DIR,
                                                      "send_recommendations.csv"),
                        help="where to write the per-email recommendations")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--sample-rows", type=int, default=3_000_000)
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--send-threshold", type=float, default=None,
                        help="override the send cutoff; default = value saved with "
                             "the model (optimal threshold from training).")
    args = parser.parse_args()

    trainer.setup_logging(args.log_level, args.log_file)
    logger = trainer.logger
    logger.info("scoring with saved ensemble | cutoff=%s | models=%s",
                trainer.CUTOFF.date(), args.models)

    try:
        if not os.path.exists(args.models):
            raise FileNotFoundError(
                f"model file not found: {args.models}. Train first with "
                f"`python script.py` to create it.")

        payload = load_ensemble(args.models)

        comm, purch = trainer.load_or_generate(
            args.comm, args.purch, sep=args.sep, chunksize=args.chunksize,
            sample_rows=args.sample_rows, sample_frac=args.sample_frac)
        comm, purch = trainer.enforce_cutoff(comm, purch)

        X, _treatment, _target, _num, _cat, meta = build_dataset(comm, purch)

        # Use the optimal threshold saved with the model (or a manual override).
        threshold = (args.send_threshold if args.send_threshold is not None
                     else float(payload.get("extra", {}).get("send_threshold", 0.0)))
        logger.info("send threshold = %.5f (%s)", threshold,
                    "manual" if args.send_threshold is not None else "from saved model")

        reco = trainer.build_send_recommendations(payload["members"], X, meta,
                                                  comm, purch, threshold=threshold)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        reco.to_csv(args.out, index=False)

        n_send = int(reco["recommend_send"].sum())
        n_excl = int(reco["has_upcoming_trip"].sum())
        logger.info("wrote recommendations for %d emails -> %s", len(reco), args.out)
        logger.info("  recommend SEND: %d (%.1f%%) | excluded upcoming-trip: %d",
                    n_send, 100 * n_send / max(1, len(reco)), n_excl)
        logger.info("top of recommendation table:\n%s",
                    reco.head(10).to_string(index=False))
    except Exception:
        logger.exception("SCORING ABORTED due to an unhandled error")
        sys.exit(1)


if __name__ == "__main__":
    main()
