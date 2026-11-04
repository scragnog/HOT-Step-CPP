"""Foreground process deadline; never leave a reference inference running."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from common import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).with_name('capture.py'))]
    for key, value in vars(args).items():
        command.extend(['--' + key, str(value.resolve())])
    started = time.perf_counter()
    with (args.out / 'capture.log').open('w', encoding='utf-8') as log:
        try:
            result = subprocess.run(command, cwd=args.upstream, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=1800)
            code = result.returncode
            status = 'finished' if code == 0 else 'failed'
        except subprocess.TimeoutExpired:
            # subprocess.run kills and waits for its exact child on timeout.
            code, status = 124, 'timeout'
    receipt = dict(status=status, exit_code=code, elapsed_seconds=time.perf_counter()-started,
                   deadline_seconds=1800, command=command)
    write_json(args.out / 'run.json', receipt)
    print(receipt, flush=True)
    print((args.out / 'capture.log').read_text(encoding='utf-8'), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
