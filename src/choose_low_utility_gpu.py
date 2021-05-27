import numpy as np
import os, sys
import subprocess
from termcolor import colored


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    exit(0)


def choose_low_utility_gpu(limit=5000):
    status, output = subprocess.getstatusoutput("nvidia-smi | grep %")
    if status != 0:
        eprint("command error")

    memory = np.array(
        [line.split("|")[2].split()[0][:-3] for line in output.split("\n")]
    ).astype(np.uint16)
    gpu_ID = np.argsort(memory)[0]

    if memory[gpu_ID] > limit:
        eprint("No empty Graphic Card Available")

    print(colored(f"Using GPU: {gpu_ID}", "green"))
    return gpu_ID


if __name__ == "__main__":
    print(choose_low_utility_gpu())
