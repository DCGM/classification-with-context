"""Forward interrupts and preserve exit status for a systemd-managed training attempt."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    command = [sys.executable, '-u', str(args.root / 'launch.py')]
    if args.resume:
        command.append('--resume')
    process = subprocess.Popen(command)
    def forward(signum, frame):
        if process.poll() is None:
            process.send_signal(signum)
    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    code = process.wait()
    temporary = args.root / 'managed-exit.writing'
    temporary.write_text(json.dumps({'exit_code': code,
                         'finished': datetime.datetime.now().astimezone().isoformat()}) + '\n')
    os.replace(temporary, args.root / 'managed-exit.json')
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    sys.exit(main())
