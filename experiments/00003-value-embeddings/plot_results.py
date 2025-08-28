
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


if __name__ == "__main__":
    # plot_val_loss(
    #     header_numbers={
    #         # 0: "rm layer 0",
    #         # 1: "rm layer 1",
    #         # 2: "rm layer 2",
    #         # 3: "rm layer 13",
    #         # 4: "rm layer 14",
    #         # 5: "rm layer 15",
    #         # 6: "rm layers 0, 1",
    #         # 7: "rm layers 0, 1, 2",
    #         # 8: "rm layers 0, 13",
    #         # 9: "rm layers 1, 14",
    #         # 10: "rm layers 2, 15",
    #         # 11: "shared valemb 012-131415",
    #         # 12: "shared valemb 01-1415",
    #         # 13: "baseline",
    #         # 14: "new valemb 15",
    #         # 15: "new valemb 3-15",
    #         # "13-1": "Baseline 1",
    #         # 16: "new valemb 3-14-4-15",
    #         # 17: "new valemb 3-13-4-14-5-15",
    #         # 18: "new valemb 3-12-4-13-5-14-6-15",
    #         # 19: "new valemb 3-11-4-12-5-13-6-14-7-15",
    #         # 20: "shifted valemb 123-131415",
    #         # 21: "shifted valemb 234-131415",
    #         "22-0": "baseline record 1",
    #         "22-1": "baseline record 2",
    #         "22-2": "baseline record 3",
    #         "22-3": "baseline record 4",
    #         "22-4": "baseline record 5",
    #         "23-0": "new valemb 3-15 record 1",
    #         "23-1": "new valemb 3-15 record 2",
    #         "23-2": "new valemb 3-15 record 3",
    #         "23-3": "new valemb 3-15 record 4",
    #         "23-4": "new valemb 3-15 record 5",
    #         "24-0": "new valemb 3-14-4-15 record 1",
    #         "24-1": "new valemb 3-14-4-15 record 2",
    #         "24-2": "new valemb 3-14-4-15 record 3",
    #         "24-3": "new valemb 3-14-4-15 record 4",
    #         "24-4": "new valemb 3-14-4-15 record 5",
    #     },
    #     average_over={
    #         "baseline": ("22-0", "22-1", "22-2", "22-3", "22-4"),
    #         "new valemb 3-15": ("23-0", "23-1", "23-2", "23-3", "23-4"),
    #         "new valemb 3-14-4-15": ("24-0", "24-1", "24-2", "24-3", "24-4"),
    #     },
    #     filename="results.md",
    #     x_axis="time",
    # )

    from rich import print
    print("## Baseline")
    print("\n### Final val_losses")
    final_losses=get_final_val_losses("t-test-results.md", [f"25 {hnum}" for hnum in range(28)])
    print(f"\n#### Final val_losses\n\n```python\n{final_losses}\n```")
    final_losses = np.array(final_losses)

    print("\n#### Simple statistics\n")
    print(f"- Mean:     {np.mean(final_losses):.3f} ± {np.std(final_losses):.3f}")
    print(f"- Median:   {np.median(final_losses):.3f}")
    print(f"- Min:      {np.min(final_losses):.3f}")
    print(f"- Max:      {np.max(final_losses):.3f}")
    print(f"- # < 2.92: {np.sum(final_losses < 2.92).item():.0f}")

    print("\n#### T-test results\n")
    ttest_results = test_mean_below(final_losses)
    printable_ttest_results = r"{" + "\n    " + ",\n    ".join([f'"{k}": {v}' for k, v in ttest_results.items()]) + "\n" + r"}"
    print(f"```python\n{printable_ttest_results}\n```")

    print("\n### Run times\n")
    final_times = get_final_times("t-test-results.md", [f"25 {hnum}" for hnum in range(28)])
    fts = [f"{t:.3f}" for t in final_times]
    print(f"\n#### Run times\n\n```python\n{fts}\n```")
    final_times = np.array(final_times)
    print("\n#### Simple statistics\n")
    print(f"\nMean:     {np.mean(final_times):.3f} ± {np.std(final_times):.3f}")
    print(f"Median:   {np.median(final_times):.3f}")
    print(f"Min:      {np.min(final_times):.3f}")
    print(f"Max:      {np.max(final_times):.3f}")

    print("\n## Two new value embeddings")
    print("\n### Final val_losses")
    final_losses=get_final_val_losses("t-test-results.md", [f"26 {hnum}" for hnum in range(37)])
    print(f"\n#### Final val_losses\n\n```python\n{final_losses}\n```")
    final_losses = np.array(final_losses)

    print("\n#### Simple statistics\n")
    print(f"- Mean:     {np.mean(final_losses):.3f} ± {np.std(final_losses):.3f}")
    print(f"- Median:   {np.median(final_losses):.3f}")
    print(f"- Min:      {np.min(final_losses):.3f}")
    print(f"- Max:      {np.max(final_losses):.3f}")
    print(f"- # < 2.92: {np.sum(final_losses < 2.92).item():.0f}")

    print("\n#### T-test results\n")
    ttest_results = test_mean_below(final_losses)
    printable_ttest_results = r"{" + "\n    " + ",\n    ".join([f'"{k}": {v}' for k, v in ttest_results.items()]) + "\n" + r"}"
    print(f"```python\n{printable_ttest_results}\n```")

    print("\n### Run times\n")
    final_times = get_final_times("t-test-results.md", [f"26 {hnum}" for hnum in range(37)])
    fts = [f"{t:.3f}" for t in final_times]
    print(f"\n#### Run times\n\n```python\n{fts}\n```")
    final_times = np.array(final_times)
    print("\n#### Simple statistics\n")
    print(f"\nMean:     {np.mean(final_times):.3f} ± {np.std(final_times):.3f}")
    print(f"Median:   {np.median(final_times):.3f}")
    print(f"Min:      {np.min(final_times):.3f}")
    print(f"Max:      {np.max(final_times):.3f}")
