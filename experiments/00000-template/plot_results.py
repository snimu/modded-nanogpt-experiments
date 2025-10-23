
import sys
import argparse
import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def get_val_losses(
        header_numbers: list[int | str] | dict[int | str, str],
        filename: str,
        average_over: dict[str, tuple[str, int]] | None = None,
):
    with open(filename, "r") as f:
        lines = f.readlines()

    if isinstance(header_numbers, dict):
        descriptions = list(header_numbers.values())
        header_numbers = list(header_numbers.keys())
    else:
        descriptions = ["" for _ in header_numbers]

    parsed = {hnum: {"step": [], "time": [], "loss": []} for hnum in header_numbers}
    for hnum in header_numbers:
        extract= False
        for line in lines:
            if line.strip() == f"## {hnum}":
                extract = True
                continue
            if extract and line.startswith("##"):
                break
            if extract and line.startswith("step:") and "val_loss" in line:
                parsed[hnum]["loss"].append(float(line.split()[1].split("val_loss:")[-1]))
                parsed[hnum]["step"].append(int(line.split("step:")[1].split("/")[0]))
                parsed[hnum]["time"].append(float(line.split("train_time:")[1].split("ms")[0]) / 1000)
    
    if average_over is not None:
        header_numbers = list(average_over.keys())
        descriptions = ["" for _ in header_numbers]
        new_parsed = {}
        for hnum in average_over:
            group = average_over[hnum]
            steps = parsed[group[0]]["step"]
            times = np.array([parsed[header]["time"] for header in group])
            losses = np.array([parsed[header]["loss"] for header in group])
            new_parsed[hnum] = {
                "loss": np.mean(losses, axis=0),
                "step": steps,
                "time": np.mean(times, axis=0),
            }
        parsed = new_parsed
    return parsed, header_numbers, descriptions


def plot_val_loss(
        header_numbers: list[int | str] | dict[int | str, str],
        filename: str,
        x_axis: str = "step",
        average_over: dict[str, tuple[str, int]] | None = None,
):
    parsed, header_numbers, descriptions = get_val_losses(header_numbers, filename, average_over)

    for i, hnum in enumerate(header_numbers):
        description = f": {descriptions[i]}" if descriptions[i] else ""
        plt.plot(parsed[hnum][x_axis], parsed[hnum]["loss"], label=f"{hnum}{description}")
    plt.xlabel("step" if x_axis == "step" else "time (s)")
    plt.ylabel("val_loss")
    plt.legend()
    plt.grid()
    plt.show()


def get_final_val_losses(filename: str, header_numbers: list[int | str]):
    parsed, header_numbers, _ = get_val_losses(header_numbers, filename)
    final_losses = [parsed[hnum]["loss"][-1] for hnum in header_numbers]
    return final_losses


def get_final_times(filename: str, header_numbers: list[int | str]):
    parsed, header_numbers, _ = get_val_losses(header_numbers, filename)
    final_times = [parsed[hnum]["time"][-1] for hnum in header_numbers]
    return final_times


def test_mean_below(losses, threshold=2.92, alpha=0.05):
    """
    One-sided t-test for H0: mu = threshold vs H1: mu < threshold.
    Also returns the (1-alpha) upper confidence bound for the mean.
    """
    losses = np.asarray(losses, dtype=float)
    n = losses.size
    xbar = losses.mean()
    s = losses.std(ddof=1)

    # One-sided t-test (lower-tailed)
    t_res = stats.ttest_1samp(losses, popmean=threshold, alternative='less')
    t_stat, p_value = t_res.statistic, t_res.pvalue

    # (1 - alpha) upper bound for the mean (one-sided CI)
    # mean <= xbar + t_{1-alpha, df} * s/sqrt(n)
    t_crit = stats.t.ppf(1 - alpha, df=n-1)
    upper_bound = xbar + t_crit * s / np.sqrt(n)

    decision = "REJECT H0 (mean < threshold)" if p_value < alpha else "FAIL TO REJECT H0"

    return {
        "n": n,
        "sample_mean": xbar,
        "sample_std": s,
        "t_stat": t_stat,
        "p_value": p_value,
        "alpha": alpha,
        "decision": decision,
        "upper_conf_bound_mean": upper_bound,  # (1-alpha) one-sided upper bound
        "threshold": threshold
    }


def extract_vallosses(path: str, name: str, offset: int = 0):
    files = sorted(os.listdir(path))
    results = ""
    for i, file in enumerate(files):
        title = f"## {name}-{i + offset}"
        with open(os.path.join(path, file), "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if "val_loss" in line and line.startswith("step:") and line.strip()]
        trace = '\n'.join(lines)
        results += f"{title}\n\n{trace}\n\n"

    return results


def get_all_final_losses_and_times(path_to_results: str) -> dict[str, dict[Literal['loss', 'time'], float]]:
    # Find subdirectories under path_to_results (not under CWD)
    with os.scandir(path_to_results) as it:
        subdirs = sorted([e.name for e in it if e.is_dir()])

    # Collect all the results into one file
    fulltext = ""
    for subdir in subdirs:
        fulltext += extract_vallosses(
            path=os.path.join(path_to_results, subdir),
            name=subdir
        ) + "\n\n"

    results_file = os.path.join(path_to_results, "extracted_vallosses")
    with open(results_file, "w") as f:
        f.write(fulltext)

    # Extract val_losses and times
    results = {}
    sep = f"{'-'*60}\n{'-'*60}"
    formatted_results = f"\n\n{sep}"
    for subdir in subdirs:
        subdir_path = os.path.join(path_to_results, subdir)
        try:
            run_files = [e for e in os.scandir(subdir_path) if e.is_file()]
        except FileNotFoundError:
            run_files = []
        headers = [f"{subdir}-{i}" for i in range(len(run_files))]
        if not headers:
            continue
        parsed, _, _ = get_val_losses(headers, results_file)
        final_losses = []
        final_times = []
        for h in headers:
            # Only include runs that produced values
            if parsed[h]["loss"]:
                final_losses.append(parsed[h]["loss"][-1])
            if parsed[h]["time"]:
                final_times.append(parsed[h]["time"][-1])
        if not final_losses or not final_times:
            continue
        loss = float(np.mean(final_losses))
        time = float(np.mean(final_times))
        results[subdir] = {
            "loss": round(loss, 4),
            "time": round(time, 2),
            "n": len(final_losses),
            "p": test_mean_below(final_losses)["p_value"],
        }
        formatted_results += (
            f"\n{subdir} —— Loss: {results[subdir]['loss']}, "
            f"Time: {results[subdir]['time']}, n: {results[subdir]['n']}, "
            f"p: {results[subdir]['p']}"
        )
    formatted_results += f"\n{sep}\n\n"
    return results, formatted_results


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-losses", action="store_true")
    parser.add_argument("--print-final-stats", action="store_true")
    parser.add_argument("--path", type=str, default="", help="The subdir of logs if --extract-losses, logs if --print-final-stats")
    parser.add_argument("--name", type=str, default="", help="The name if --extract-losses")
    parser.add_argument("--offset", type=int, default=0, help="The offset if --extract-losses")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.extract_losses:
        vallosses = extract_vallosses(path=args.path, name=args.name, offset=args.offset)
        with open(os.path.join("logs", args.path, "vallosses.md"), "w") as f:
            f.write(vallosses)
    if args.print_final_stats:
        import rich
        print()
        _, formatted_results = get_all_final_losses_and_times(args.path)
        rich.print(formatted_results)
        print()
    if args.extract_losses or args.print_final_stats:
        sys.exit(0)  # only perform the freeform code if nothing else is done