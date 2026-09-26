"""Start a stage on the deployed Modal app and return immediately (it keeps running in the cloud).

    python launch.py "--stage train --sample 800000"      # CPU machine
    python launch.py "--stage ce" --gpu                   # GPU machine (cross-encoder)

Watch progress and logs at https://modal.com/apps (app "amazon-ml-v4").
"""
import sys

import modal

if len(sys.argv) < 2:
    sys.exit('usage: python launch.py "--stage NAME [options]" [--gpu]')
args = sys.argv[1].split()
fn_name = "run_gpu" if "--gpu" in sys.argv[2:] else "run"
fn = modal.Function.from_name("amazon-ml-v4", fn_name)
call = fn.spawn(args)
print(f"started {fn_name} {' '.join(args)}  (call id {call.object_id})")
print("It runs in the cloud now. Watch the logs at https://modal.com/apps")