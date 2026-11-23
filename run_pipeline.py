"""Run the complete project end-to-end.

    python run_pipeline.py                     # everything (first run downloads ~210 MB)
    python run_pipeline.py --skip-download     # data already in data/raw
    python run_pipeline.py --skip-download --skip-av   # only the speed-control study (~2 min)

Stages: download -> preprocess -> train risk models -> collision-avoidance simulation ->
figures -> Level-2 + RC-car evaluation -> report
"""
import argparse
import time
import warnings

warnings.filterwarnings("ignore")

from src import config as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-av", action="store_true", help="skip the Level-2 and RC-car evaluation (~4 min)")
    ap.add_argument("--mc", type=int, default=500, help="Monte-Carlo runs per synthetic scenario")
    ap.add_argument("--av-runs", type=int, default=30, help="random variants per Level-2 scenario family")
    args = ap.parse_args()
    t0 = time.time()

    def stage(name):
        print(f"\n{'=' * 70}\n{name}  (t+{time.time() - t0:.0f}s)\n{'=' * 70}")

    if not args.skip_download:
        stage("1/7  Download NGSIM data")
        from src import download_data
        for s in C.SITES:
            download_data.download_site(s)

    stage("2/7  Preprocess trajectories and compute surrogate safety measures")
    from src import preprocess
    for s in C.SITES:
        preprocess.run(s)

    stage("3/7  Train and evaluate risk-prediction models")
    from src import train_models
    train_models.main()

    stage("4/7  Closed-loop collision-avoidance simulation")
    from src import collision_avoidance
    collision_avoidance.run_all(n_mc=args.mc)

    stage("5/7  Figures")
    from src import visualize
    visualize.make_all()

    if not args.skip_av:
        stage("6/7  Level-2 driving (steering, lane changes, lights, pedestrians) and RC-car evaluation")
        from src.av import evaluate
        evaluate.run(args.av_runs)
        import demo2d
        demo2d.save_report_figure()
        from rc_car import evaluate as rc_eval
        rc_eval.main()

    stage("7/7  Report")
    from src import report
    report.build()

    print(f"\nFinished in {time.time() - t0:.0f}s. See reports/.")


if __name__ == "__main__":
    main()
