import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    executable, backend = sys.argv[1:]
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = str(Path(directory) / "refined.flst")
        uninterrupted = subprocess.check_output(
            [executable, backend, "continuous", checkpoint], text=True, timeout=30
        ).strip()
        with subprocess.Popen(
            [executable, backend, "interrupt", checkpoint],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        ) as process:
            try:
                checkpoint_receipt = process.stdout.readline().strip()
                assert checkpoint_receipt.startswith("checkpoint="), checkpoint_receipt
            finally:
                process.kill()
                assert process.wait(timeout=10) != 0
        resumed = subprocess.check_output(
            [executable, backend, "resume", checkpoint], text=True, timeout=30
        ).strip()
        assert resumed.startswith("final=") and resumed == uninterrupted, (uninterrupted, resumed)
        print(backend, checkpoint_receipt, resumed)


if __name__ == "__main__":
    main()
