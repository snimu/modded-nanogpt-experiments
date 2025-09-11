
import ast
from typing import Mapping, Sequence, Optional, Literal
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass


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


@dataclass
class Stats:
    mean: float
    median: float
    std: float
    min: float
    max: float


def get_stats(values: list[float| int]) -> Stats:
    return Stats(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values)),
        min=min(values),
        max=max(values),
    )


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


def _plot_percent_weights(
    series: Mapping[str, Sequence[float]],
    x: Optional[Sequence] = None,
    xaxis: Literal["layer", "step"] = "layer",
    kind: str = "bar",      # "bar" or "area"
    title: Optional[str] = None,
):
    """
    Plot normalized percentage weights per position from a dict of label -> sequence.
    Handles negatives by normalizing with sum of absolute values at each position.

    Parameters
    ----------
    series : dict[str, Sequence[float]]
        Keys are legend labels; values are equal-length sequences of numbers.
    x : Sequence | None
        X positions/labels. If None, uses range(n_points).
    kind : {"bar","area"}
        "bar" -> 100% stacked bars; "area" -> 100% stacked area plot.
    title : str | None
        Optional plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if not series:
        raise ValueError("`series` cannot be empty.")

    labels = list(series.keys())
    data = np.array([series[k] for k in labels], dtype=float)  # shape: (n_series, n_points)

    if data.ndim != 2:
        raise ValueError("Each value in `series` must be a 1D sequence of equal length.")

    n_series, n_points = data.shape
    if x is None:
        x = np.arange(n_points)

    x = np.array(x)
    if x.size != n_points:
        raise ValueError(f"`x` must have length {n_points}, got {x.size}.")

    # Normalize by column using sum of absolute values (handles negatives)
    denom = np.sum(np.abs(data), axis=0)  # shape: (n_points,)
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.divide(
            np.abs(data), denom,
            out=np.zeros_like(data),
            where=denom != 0
        )  # each column sums to 1 (or 0 if denom==0)

    fig, ax = plt.subplots()

    if kind.lower() == "area":
        ax.stackplot(x, *weights, labels=labels)
    elif kind.lower() == "bar":
        bottom = np.zeros(n_points)
        for i, lab in enumerate(labels):
            ax.bar(x, weights[i], bottom=bottom, label=lab)
            bottom += weights[i]
    else:
        raise ValueError("`kind` must be 'bar' or 'area'.")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Share")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_xlabel(xaxis)
    ax.set_title(title or "Normalized component shares (|value| / sum |values|)")
    ax.legend(title="Series", ncol=min(len(labels), 3))
    fig.tight_layout()
    plt.show()


def _plot_abs_weigths(
        series: Mapping[str, Sequence[float]],
        xaxis: Literal["layer", "step"] = "layer",
        normalize: bool = True,
        title: str | None = None,
):
    if not series:
        raise ValueError("`series` cannot be empty.")
    labels = list(series.keys())
    data = np.array([series[k] for k in labels], dtype=float)  # shape: (n_series, n_points)
    if normalize:
        denom = np.sum(np.abs(data), axis=0)  # shape: (n_points,)
        with np.errstate(divide='ignore', invalid='ignore'):
            data = np.divide(
                data, denom,
                out=np.zeros_like(data),
                where=denom != 0
            )  # each column sums to 1 (or 0 if denom==0)
    for label, d in zip(labels, data):
        plt.plot(d, label=label)
    plt.xlabel(xaxis)
    plt.ylabel("Weight")
    plt.legend()
    plt.grid()
    if title:
        plt.title(title)
    plt.show()


def plot_x0_lambdas(
        filename: str,
        header_numbers: int | str | list[int | str],
        kind: Literal["bar", "area", "plot"] = "area",
        xaxis: Literal["layer", "step"] = "layer",
        layer_idx: int | None = None,
        normalize: bool = True,
        d_dx: bool = False,
        title: str | None = None,
        x_norms: list[float] | None = None,
):
    if xaxis == "step" and layer_idx is None:
        raise ValueError("`layer_idx` must be specified if `xaxis` is 'step'.")
    header_numbers = header_numbers if isinstance(header_numbers, list) else [header_numbers]
    with open(filename, "r") as f:
        lines = f.readlines()
    all_lambdas = [
        "x_lambdas", "x00_lambdas", "x01_lambdas", "x02_lambdas",
        "x03_lambdas", "x04_lambdas"
    ]
    results = {h: None for h in header_numbers}
    for header_number in header_numbers:
        filtered_lines = []
        extract = False
        for line in lines:
            if line.strip() == f"## {header_number}":
                extract = True
                continue
            if extract and line.startswith("##"):
                extract = False
                break
            if extract and line.startswith("step:"):
                filtered_lines.append(line)

        lambdas = [l for l in all_lambdas if l in "".join(filtered_lines)]
        parsed = {l: [] for l in lambdas}
        for line in filtered_lines:
            for l in lambdas:
                if l in line:
                    parsed[l].append(ast.literal_eval(line.split(f"{l}:")[-1].strip()))
        for l in lambdas:
            parsed[l] = np.array(parsed[l])
            if xaxis == "layer":
                parsed[l] = parsed[l][-1, :]
            else:
                parsed[l] = parsed[l][:, -1]
        results[header_number] = parsed

    if len(header_numbers) == 1:
        parsed = results[header_numbers[0]]
    else:
        parsed = {l: [] for l in lambdas}
        for l in lambdas:
            parsed[l] = [results[h][l].tolist() for h in header_numbers]
            parsed[l] = np.mean(np.array(parsed[l]), axis=0)
    if x_norms is not None:
        # The norm isn't the RMS norm; it includes the mean. So since x == x00 at the start,
        # and x01 should be similar, I'll just divide all the norms by the first one.
        x_norms = np.array(x_norms) / x_norms[0]
        for l in lambdas:
            parsed[l] = parsed[l] * x_norms
    if d_dx:
        for l in lambdas:
            parsed[l] = parsed[l][1:] - parsed[l][:-1]
    for l in lambdas:
        parsed[l] = parsed[l].tolist()
    if kind == "plot":
        _plot_abs_weigths(parsed, xaxis=xaxis, normalize=normalize, title=title)
    else:
        _plot_percent_weights(parsed, xaxis=xaxis, kind=kind)


if __name__ == "__main__":
    kind = "plot"
    d_dx = False
    normalize = True
    xaxis = "layer"
    layer_idx = 0
    x_norms = [8192., 323584., 544768., 501760., 477184., 362496., 499712., 411648., 430080., 602112., 573440., 606208., 520192., 630784., 724992., 782336.]

    x0s = [0, "0-1", "0-2", "0-3", "0-4"]
    # for x0 in x0s:
    #     plot_x0_lambdas(
    #         filename="results.md",
    #         header_numbers=x0,
    #         kind=kind,
    #         xaxis=xaxis,
    #         layer_idx=layer_idx,
    #         normalize=normalize,
    #         d_dx=d_dx,
    #     )
    # plot_x0_lambdas(
    #     filename="results.md",
    #     header_numbers=x0s,
    #     kind=kind,
    #     xaxis=xaxis,
    #     layer_idx=layer_idx,
    #     normalize=normalize,
    #     d_dx=d_dx,
    #     title="Mean over 5 runs",
    #     x_norms=x_norms,
    # )
    # plot_x0_lambdas(
    #     filename="results.md",
    #     header_numbers=[1, "1-1", "1-2"],
    #     kind=kind,
    #     xaxis="layer",
    #     layer_idx=layer_idx,
    #     normalize=False,
    #     d_dx=d_dx,
    # )
    # plot_x0_lambdas(
    #     filename="results.md",
    #     header_numbers=[3, "3-1", "3-2", "3-3", "3-4"],
    #     kind=kind,
    #     xaxis="layer",
    #     layer_idx=layer_idx,
    #     normalize=False,
    #     d_dx=d_dx,
    # )
    # plot_val_loss(
    #     header_numbers={
    #         # 0: "x00-x01",
    #         # 1: "x00-x01-x02",
    #         # 2: "x00-x01-x02-x03",
    #         # 3: "x00-x01-x02-x03-x04",
    #         # "999-0": "Baseline 1",
    #         # "999-1": "Baseline 2",
    #         "0-record-0": "x00-x01",
    #         "0-record-1": "x00-x01",
    #         "0-record-2": "x00-x01",
    #         "0-record-3": "x00-x01",
    #         "0-record-4": "x00-x01",
    #         "1-record-0": "x00-x01-x02",
    #         "1-record-1": "x00-x01-x02",
    #         "1-record-2": "x00-x01-x02",
    #         "1-record-3": "x00-x01-x02",
    #         "1-record-4": "x00-x01-x02",
    #         "999-record-0": "Baseline 1",
    #         "999-record-1": "Baseline 2",
    #         "999-record-2": "Baseline 3",
    #         "999-record-3": "Baseline 4",
    #         "999-record-4": "Baseline 5",
    #     },
    #     average_over={
    #         "+ x01": ("0-record-0", "0-record-1", "0-record-2", "0-record-3", "0-record-4"),
    #         "+ x01, x02": ("1-record-0", "1-record-1", "1-record-2", "1-record-3", "1-record-4"),
    #         "baseline": ("999-record-0", "999-record-1", "999-record-2", "999-record-3", "999-record-4"),
    #     },
    #     filename="results.md",
    #     x_axis="step",
    # )
    # plot_val_loss(
    #     filename="t-test-results.md",
    #     header_numbers=["999 0", "999 1"],
    #     average_over={"999": ["999 0", "999 1"]},
    #     x_axis="time",
    # )
    # plot_val_loss(
    #     filename="results.md",
    #     header_numbers=[f"{i} {j}" for i in [60, 61, 62] for j in range(3)],
    #     average_over={
    #         60: [f"60 {i}" for i in range(3)],
    #         61: [f"61 {i}" for i in range(3)],
    #         62: [f"62 {i}" for i in range(3)],
    #     },
    #     x_axis="step",
    # )

    def make_title(title: str):
        barrier = "\n" + "-" * (len(title) + 2) + "\n"
        return barrier + f" {title} " + barrier

    losses = get_final_val_losses(
        filename="t-test-results.md",
        header_numbers=[f"692 {i}" for i in range(18)],
    )
    times = get_final_times(
        filename="t-test-results.md",
        header_numbers=[f"692 {i}" for i in range(18)],
    )
    from rich import print
    print(make_title("692: Record attempt (5690 steps)"))
    print(test_mean_below(losses=losses))
    print("\nLoss stats 692:\n")
    print(get_stats(losses))
    print("\nTime stats 692:\n")
    print(get_stats(times))
    
    
    losses = get_final_val_losses(
        filename="t-test-results.md",
        header_numbers=[f"693 {i}" for i in range(17)],
    )
    times = get_final_times(
        filename="t-test-results.md",
        header_numbers=[f"693 {i}" for i in range(17)],
    )
    print(make_title("693: Inverse ve sharing"))
    print(test_mean_below(losses=losses))
    print("\nLoss stats 693:\n")
    print(get_stats(losses))
    print("\nTime stats 693:\n")
    print(get_stats(times))
