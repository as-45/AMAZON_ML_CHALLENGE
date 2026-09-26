"""Run the v4 pipeline on Modal cloud machines.

  run      : CPU machine (16 CPUs, 128 GB) for every stage except "ce"
  run_gpu  : GPU machine (1x H100, 64 GB) for the cross-encoder stage "ce"

Data lives in the Modal volume "amazon-ml-work" (the cleaned work/*.parquet files).
Results (models, candidates, output/*.tsv) are written back to the same volume.

Usage, from code/business_entity_resolution/:
    python -m modal deploy modal_app.py        # once, after any code change
    python launch.py "--stage dump"
    python launch.py "--stage ce" --gpu
    python launch.py "--stage train --sample 800000"
    python launch.py "--stage eval"
    python launch.py "--stage predict"
Then download the outputs:
    python -m modal volume get amazon-ml-work output/matching_results.tsv output/matching_results.tsv --force
    python -m modal volume get amazon-ml-work output/candidate_pairs.tsv output/candidate_pairs.tsv --force
"""
import modal

app = modal.App("amazon-ml-v4")
volume = modal.Volume.from_name("amazon-ml-work")

base = modal.Image.debian_slim(python_version="3.12").pip_install(
    "polars==1.44.2", "pyarrow==25.0.1", "Unidecode==1.4.0", "rapidfuzz==3.14.6",
    "numpy==2.4.4", "lightgbm==4.7.0", "scikit-learn==1.8.0", "scipy==1.17.1",
    "sparse_dot_topn==1.2.0",
)
image = base.add_local_dir("src", remote_path="/root/src")
gpu_image = (base.pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu126")
             .add_local_dir("src", remote_path="/root/src"))


def _run(args: list):
    import subprocess
    import sys

    cmd = [sys.executable, "-u", "-m", "src.model_v4", *args, "--work", "/work", "--out", "/work/output"]
    print("running:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True, cwd="/root")
    finally:
        volume.commit()   # save whatever was written, even if the run failed


@app.function(image=image, volumes={"/work": volume}, cpu=16.0, memory=131072, timeout=6 * 3600)
def run(args: list):
    _run(args)


@app.function(image=gpu_image, volumes={"/work": volume}, gpu="H100", cpu=8.0, memory=65536, timeout=6 * 3600)
def run_gpu(args: list):
    _run(args)


@app.local_entrypoint()
def main(args: str):
    run.remote(args.split())