import argparse
import subprocess
import sys


def run(command):
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["C1", "C2", "C3", "C4", "C5"])
    parser.add_argument("--total_steps", type=int, default=25000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--amp_dtype", choices=["fp16", "bf16", "none"], default="fp16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--source_shuffle_diagnostic", action="store_true")
    args = parser.parse_args()

    for config_name in args.configs:
        command = [
            sys.executable,
            "-m",
            "src.train",
            "--config_name",
            config_name,
            "--total_steps",
            str(args.total_steps),
            "--batch_size",
            str(args.batch_size),
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--amp_dtype",
            args.amp_dtype,
        ]
        if args.use_wandb:
            command.append("--use_wandb")
        if args.source_shuffle_diagnostic:
            command.append("--source_shuffle_diagnostic")
        run(command)

    if args.evaluate:
        run(
            [
                sys.executable,
                "-m",
                "src.utils",
                "--configs",
                *args.configs,
                "--eval_batch_size",
                str(args.eval_batch_size),
                "--amp_dtype",
                args.amp_dtype,
            ]
        )


if __name__ == "__main__":
    main()
