"""Resumable local job: recipe embeddings, frozen test predictions, three diagnostics.

Launch: python scripts/run_frozen_test.py --run-dir artifacts/frozen_test --detach
Monitor: python scripts/run_frozen_test.py --run-dir artifacts/frozen_test --status
Restart with the launch command; completed, checksummed work is reused.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def active(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
    return False


def command_for(args):
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--run-dir", str(args.run_dir), "--config", str(args.config),
               "--split-dir", str(args.split_dir), "--embedding-dir", str(args.embedding_dir),
               "--device", args.device]
    if args.skip_embeddings:
        command.append("--skip-embeddings")
    for flag, value in (("--max-quarters", args.max_quarters), ("--max-users", args.max_users)):
        if value is not None:
            command.extend([flag, str(value)])
    return command


def show_status(args):
    status = read_json(args.run_dir / "status.json")
    if status is None:
        print(f"No status yet: {args.run_dir}")
        return
    print(f"Stage: {status['stage']} | {'running' if active(args.run_dir) else 'not running'}")
    print(status.get("detail", ""))
    print(f"Completed: {status.get('completed_units', 0)}/{status.get('total_units', '?')} model-quarters")
    for model, count in status.get("models", {}).items():
        print(f"  {model:20s} {count}/{status.get('quarters_per_model', '?')} quarters")
    if status["stage"] == "embeddings":
        manifest = read_json(args.run_dir / "manifest.json") or {}
        progress = read_json(Path(manifest.get("embedding_dir", args.embedding_dir)) / "progress.json")
        if progress:
            n = progress["identity"]["rows"]
            for field, item in progress["fields"].items():
                print(f"  {field:20s} {item['rows']:,}/{n:,} recipes")
    print(f"Updated: {status.get('updated_at')} | PID: {status.get('pid')}")
    print(f"Log: {args.run_dir / 'run.log'}")
    if status.get("child_log"):
        print(f"Current worker log: {status['child_log']}")
    if status.get("error"):
        print(f"Error: {status['error']}")


def valid_record(path, run_dir):
    from remy.checkpoints import file_sha256
    record = read_json(path)
    if not record:
        return None
    try:
        if all(file_sha256(run_dir / name) == digest for name, digest in record["files"].items()):
            return record
    except OSError:
        pass
    return None


def run_job(args):
    from remy.checkpoints import atomic_json, atomic_parquet, atomic_text, file_sha256
    from remy.experiment_config import load_config

    lock = (args.run_dir / "run.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("A job is already running in this directory")
    status = {"stage": "starting", "pid": os.getpid(), "started_at": now(), "completed_units": 0}

    def update(stage, detail, **extra):
        status.update(stage=stage, detail=detail, updated_at=now(), **extra)
        atomic_json(args.run_dir / "status.json", status)

    def child(command, logfile, stage, detail):
        logfile.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{now()}] {detail}; log: {logfile}", flush=True)
        with logfile.open("a") as log:
            log.write(f"\n[{now()}] {shlex.join(command)}\n")
            log.flush()
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            update(stage, detail, child_pid=process.pid, child_log=str(logfile))
            try:
                while process.poll() is None:
                    time.sleep(3)
                    update(stage, detail)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            update(stage, detail, child_pid=None)
            if process.returncode:
                raise RuntimeError(f"Worker exited with code {process.returncode}; see {logfile}")

    def aggregate(models):
        import pandas as pd
        from remy.evaluate import METRICS
        from remy.diagnostics import DIAGNOSTIC_METRICS
        rows = []
        counts = {name: 0 for name in models}
        for path in sorted((args.run_dir / "units").glob("*/*/metrics.json")):
            record = valid_record(path, args.run_dir)
            if record:
                rows.append(record["row"])
                counts[record["row"]["model"]] += 1
        if rows:
            frame = pd.DataFrame(rows).sort_values(["start", "model"])
            atomic_text(args.run_dir / "quarter_metrics.csv", frame.to_csv(index=False))
            means = frame.groupby("model")[METRICS + DIAGNOSTIC_METRICS].mean().reindex(models)
            means.insert(0, "completed_quarters", [counts[name] for name in means.index])
            atomic_text(args.run_dir / "summary.csv", means.to_csv())
        status.update(completed_units=len(rows), models=counts)

    try:
        snapshot = args.run_dir / "config.toml"
        if not snapshot.exists():
            atomic_text(snapshot, args.config.read_text())
        config = load_config(snapshot)
        source_files = [*sorted((ROOT / "src/remy").glob("*.py")),
                        *sorted((ROOT / "src/remy/model").glob("*.py")),
                        *(ROOT / "scripts" / name for name in ("embed_recipes.py", "run_experiments.py", "run_frozen_test.py"))]
        identity = {"config": config, "split": "test", "max_quarters": args.max_quarters,
                    "max_users": args.max_users, "split_dir": str(args.split_dir),
                    "embedding_dir": str(args.embedding_dir), "device": args.device,
                    "skip_embeddings": args.skip_embeddings,
                    "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in source_files},
                    "data_sha256": {name: file_sha256(args.split_dir / name) for name in
                                    ("recipes.parquet", "interactions.parquet", "windows.parquet")}}
        from importlib.metadata import version
        identity["versions"] = {name: version(name) for name in
                                ("numpy", "pandas", "scipy", "scikit-learn", "torch", "lightgbm",
                                 "xgboost", "sentence-transformers", "transformers")}
        manifest = args.run_dir / "manifest.json"
        if manifest.exists() and read_json(manifest) != identity:
            raise ValueError("The code, frozen settings or dataset changed. Use a new run directory rather than mixing results.")
        atomic_json(manifest, identity)
        def verify_sources():
            if any(file_sha256(ROOT / name) != digest for name, digest in identity["source_sha256"].items()):
                raise ValueError("Model/evaluation code changed during this job; completed work is preserved")
        atomic_text(args.run_dir / "resume_command.txt", shlex.join(command_for(args) + ["--detach"]) + "\n")
        from remy.splits import RollingSplits
        splits = RollingSplits(args.split_dir)
        windows = splits.windows[splits.windows.split.eq("test")].iloc[:args.max_quarters]
        if windows.empty:
            raise ValueError("No test quarters found")
        models = list(config["models"])
        status.update(total_units=len(models) * len(windows), quarters_per_model=len(windows))
        aggregate(models)
        update("starting", "Checking completed work")
        if not args.skip_embeddings:
            child([sys.executable, "-u", str(ROOT / "scripts/embed_recipes.py"),
                   "--split-dir", str(args.split_dir), "--output-dir", str(args.embedding_dir),
                   "--device", args.device], args.run_dir / "embeddings.log", "embeddings", "Generating recipe embeddings")
        embedding_identity = {name: file_sha256(args.embedding_dir / name)
                              for name in ("id.npy", "name.npy", "ingredients.npy", "vectors.npy")}
        saved_identity = args.run_dir / "embedding_identity.json"
        if saved_identity.exists() and read_json(saved_identity) != embedding_identity:
            raise ValueError("Saved embeddings changed; use a new run directory")
        atomic_json(saved_identity, embedding_identity)
        from remy.model.embedding import load_embeddings
        from remy.diagnostics import RecipeSimilarity, evaluate_diagnostics
        import numpy as np
        import pandas as pd
        similarity = RecipeSimilarity(load_embeddings(directory=args.embedding_dir))
        if not set(splits.recipes.id) <= set(similarity.rows.index):
            raise ValueError("Embeddings do not cover the split catalog")
        settings = config.get("diagnostics", {})
        threshold = settings.get("substitute_threshold", 0.8)
        for window in windows.itertuples(index=False):
            quarter = splits._make_quarter(window.start, window.end)
            stamp = str(window.start.date())
            for name in models:
                verify_sources()
                unit = args.run_dir / "units" / name / stamp
                unit.mkdir(parents=True, exist_ok=True)
                if valid_record(unit / "metrics.json", args.run_dir):
                    print(f"[{now()}] Reusing {name} {stamp}: predictions and metrics complete", flush=True)
                    continue
                base = valid_record(unit / "base.json", args.run_dir)
                if base is None:
                    attempts = unit / "attempts"
                    attempt_number = len(list(attempts.glob("*"))) + 1
                    attempt = attempts / f"{attempt_number:04d}"
                    output = attempt / "results.csv"
                    command = [sys.executable, "-u", str(ROOT / "scripts/run_experiments.py"),
                               "--config", str(snapshot), "--split", "test", "--models", name,
                               "--quarter", stamp, "--split-dir", str(args.split_dir),
                               "--embedding-dir", str(args.embedding_dir), "--output", str(output), "--no-plots"]
                    if args.max_users is not None:
                        command.extend(["--max-users", str(args.max_users)])
                    logfile = args.run_dir / "logs" / f"{stamp}_{name}_{attempt_number:04d}.log"
                    child(command, logfile, "models", f"Fitting/scoring {name} at {stamp}")
                    verify_sources()
                    prediction = attempt / "recommendations" / stamp / f"{name}.parquet"
                    row = json.loads(pd.read_csv(output).to_json(orient="records"))[0]
                    base = {"row": row, "prediction": str(prediction.relative_to(args.run_dir)),
                            "files": {str(p.relative_to(args.run_dir)): file_sha256(p) for p in (output, prediction)}}
                    atomic_json(unit / "base.json", base)
                else:
                    print(f"[{now()}] Reusing predictions for {name} {stamp}", flush=True)
                update("metrics", f"Computing diagnostics for {name} at {stamp}", child_pid=None)
                predictions = pd.read_parquet(args.run_dir / base["prediction"])
                scored = quarter
                if args.max_users is not None:
                    from dataclasses import replace
                    scored = replace(quarter, outcomes=quarter.outcomes[quarter.outcomes.user_id.isin(predictions.user_id)])
                metrics, per_user, matches, top_exposed = evaluate_diagnostics(scored, predictions, similarity, threshold)
                if not np.isclose(metrics.pop("Recall@10"), base["row"]["Recall@10"], atol=1e-9):
                    raise ValueError("Saved predictions disagree with the original exact Recall@10")
                atomic_parquet(unit / "per_user.parquet", per_user)
                atomic_parquet(unit / "matches.parquet", matches)
                record = {"row": {**base["row"], **metrics}, "threshold": threshold,
                          "top_exposed": [[int(recipe), int(count)] for recipe, count in top_exposed],
                          "files": {**base["files"], **{str(p.relative_to(args.run_dir)): file_sha256(p)
                                    for p in (unit / "per_user.parquet", unit / "matches.parquet")}}}
                atomic_json(unit / "metrics.json", record)
                aggregate(models)
                update("models", f"Completed {name} at {stamp}")
                print(f"[{now()}] Completed {status['completed_units']}/{status['total_units']}: {name} {stamp}", flush=True)
        aggregate(models)
        update("complete", "All frozen models and diagnostics complete", finished_at=now(), child_pid=None)
        print(f"[{now()}] COMPLETE. Summary: {args.run_dir / 'summary.csv'}", flush=True)
    except BaseException as error:
        update("failed", "Stopped; completed checkpoints are preserved", error=str(error), child_pid=None)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/frozen_test")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/frozen_models.toml")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data/processed/rolling")
    parser.add_argument("--embedding-dir", type=Path, default=ROOT / "data/processed/embeddings/minilm_joint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true", help="Use existing complete arrays (e.g. for integration checks)")
    parser.add_argument("--max-quarters", type=int, help="Only for reduced-data checks; omitted for the full job")
    parser.add_argument("--max-users", type=int, help="Only for reduced-data checks; omitted for the full job")
    args = parser.parse_args()
    for name in ("run_dir", "config", "split_dir", "embedding_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if any(value is not None and value < 1 for value in (args.max_quarters, args.max_users)):
        parser.error("max-quarters and max-users must be positive")
    if args.status:
        show_status(args)
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.detach:
        if active(args.run_dir):
            print("This job is already running.")
            show_status(args)
            return
        command = command_for(args)
        if shutil.which("caffeinate"):
            command = [shutil.which("caffeinate"), "-i", *command]
        with (args.run_dir / "run.log").open("a") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, cwd=ROOT, start_new_session=True)
        print(f"Launched supervisor PID {process.pid}")
        print(f"Monitor: {shlex.join([sys.executable, str(Path(__file__).resolve()), '--run-dir', str(args.run_dir), '--status'])}")
        print(f"Live log: {args.run_dir / 'run.log'}")
    else:
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        run_job(args)


if __name__ == "__main__":
    main()
