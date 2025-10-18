
import math
import ast
import sys
import argparse
import os
from collections import defaultdict
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


def get_val_losses_and_lambdas(
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

    parsed = {hnum: {"step": [], "time": [], "loss": [], "x-lambda": [], "lambdas": [], "skip-layers": []} for hnum in header_numbers}
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
                parsed[hnum]["x-lambda"].append(float(line.split("x-lambda:")[1].split("lambdas")[0].strip()))
                parsed[hnum]["lambdas"].append(ast.literal_eval(line.split("lambdas:")[1].split("skip-layers")[0].strip()))
                parsed[hnum]["skip-layers"].append(ast.literal_eval(line.split("skip-layers:")[1].strip()))
    
        if parsed[hnum]["skip-layers"]:
            parsed[hnum]["skip-layers"] = parsed[hnum]["skip-layers"][0]  # this is otherwise redundant

    if average_over is not None:
        header_numbers = list(average_over.keys())
        descriptions = ["" for _ in header_numbers]
        new_parsed = {}
        for hnum in average_over:
            group = average_over[hnum]
            steps = parsed[group[0]]["step"]
            times = np.array([parsed[header]["time"] for header in group])
            losses = np.array([parsed[header]["loss"] for header in group])
            x_lambdas = parsed[group[0]]["x-lambda"]
            lambdas = np.array([parsed[header]["lambdas"] for header in group])
            skip_layers = np.array([parsed[header]["skip-layers"] for header in group])
            new_parsed[hnum] = {
                "loss": np.mean(losses, axis=0),
                "step": steps,
                "time": np.mean(times, axis=0),
                "x-lambda": x_lambdas,
                "lambdas": np.mean(lambdas, axis=0),
                "skip_layers": skip_layers,
            }
        parsed = new_parsed
    return parsed, header_numbers, descriptions


def get_val_losses_and_lambdas_record(
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

    parsed = {hnum: {"step": [], "time": [], "loss": [], "x-lambda": [], "lambdas": []} for hnum in header_numbers}
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
                parsed[hnum]["x-lambda"].append(float(line.split("lx:")[1].split("ls")[0].strip()))
                parsed[hnum]["lambdas"].append(ast.literal_eval(line.split("ls:")[1].strip()))
    
    if average_over is not None:
        header_numbers = list(average_over.keys())
        descriptions = ["" for _ in header_numbers]
        new_parsed = {}
        for hnum in average_over:
            group = average_over[hnum]
            steps = parsed[group[0]]["step"]
            times = np.array([parsed[header]["time"] for header in group])
            losses = np.array([parsed[header]["loss"] for header in group])
            x_lambdas = np.array([parsed[header]["x-lambda"] for header in group])
            lambdas = np.array([parsed[header]["lambdas"] for header in group])
            new_parsed[hnum] = {
                "loss": np.mean(losses, axis=0).tolist(),
                "step": steps,
                "time": np.mean(times, axis=0).tolist(),
                "x-lambda": np.mean(x_lambdas, axis=0).tolist(),
                "lambdas": np.mean(lambdas, axis=0).tolist(),
                "all-times": [parsed[header]["time"] for header in group],
                "all-losses": [parsed[header]["loss"] for header in group],
                "all-x-lambdas": [parsed[header]["x-lambda"] for header in group],
                "all-lambdas": [parsed[header]["lambdas"] for header in group],
            }
        parsed = new_parsed
    return parsed, header_numbers, descriptions


def plot_lambdas_record(
        header_numbers: list[int | str] | dict[int | str, str],
        filename: str,
        plot_all: bool = False,
        x_axis: Literal["step", "time"] = "step",
):
    parsed, _, _ = get_val_losses_and_lambdas_record(
        filename=filename,
        header_numbers=header_numbers,
        average_over={"all": header_numbers},
    )
    parsed = parsed["all"]  # artifact of averaging
    colors = plt.get_cmap("tab10").colors
    plt.plot(parsed[x_axis], parsed["x-lambda"], color=colors[0], marker="^", label="x-lambda", lw=2)
    if plot_all:
        for x_lambdas in parsed["all-x-lambdas"]:
            plt.plot(parsed[x_axis], x_lambdas, color=colors[0], alpha=0.3)
    
    plt.plot(parsed[x_axis], parsed["lambdas"], color=colors[1], marker="o", label="skip-lambda", lw=2)
    if plot_all:
        for lambdas in parsed["all-lambdas"]:
            plt.plot(parsed[x_axis], lambdas, color=colors[1], alpha=0.3)
    plt.grid()
    plt.legend()
    plt.xlabel("Training Step" if x_axis == "step" else "Training Time")
    plt.ylabel("Lambda Value")
    plt.show()


def plot_val_loss(
        header_numbers: list[int | str] | dict[int | str, str],
        filename: str,
        x_axis: str = "step",
        average_over: dict[str, tuple[str, int]] | None = None,
        legend: bool = True,
        title: str | None = None,
):
    parsed, header_numbers, descriptions = get_val_losses(header_numbers, filename, average_over)

    for i, hnum in enumerate(header_numbers):
        description = f": {descriptions[i]}" if descriptions[i] else ""
        plt.plot(parsed[hnum][x_axis], parsed[hnum]["loss"], label=f"{hnum}{description}")
    plt.xlabel("step" if x_axis == "step" else "time (s)")
    plt.ylabel("val_loss")
    if legend:
        plt.legend()
    if title:
        plt.title(title)
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


def plot_final_losses_over_names(
        filename: str,
        header_numbers: list[int | str],
        names: list[str] | None = None,
        xaxis_label: str | None = None,
        yaxis_label: str | None = None,
):
    if names is not None:
        assert isinstance(names, list), f"{type(names)=}"
        assert len(names) == len(header_numbers), f"{len(names)=}, {len(header_numbers)=}"
    else:
        names = [str(hnum) for hnum in header_numbers]
    
    losses = get_final_val_losses(filename, header_numbers)

    # Annotate each bar with its loss value (rounded to 4 decimals)
    bars = plt.bar(names, losses)
    plt.xticks(rotation=50, ha='right')
    plt.axhline(2.92, color='red', linewidth=1)
    plt.ylim(2.9, 2.93)
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    y_offset = 0.02 * (ymax - ymin)
    for rect, loss in zip(bars, losses):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + y_offset,
            f"{loss:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    if xaxis_label:
        plt.xlabel(xaxis_label)
    if yaxis_label:
        plt.ylabel(yaxis_label)
    plt.show()


def plot_final_losses_over_names_by_method(
        filename: str,
        method_to_header_numbers: dict[str, list[str | int]],
        x_labels: list[str],
        x_axis_label: str = "layer",
        draw_target_loss: bool = True,
) -> None:
    l0 = len(method_to_header_numbers[list(method_to_header_numbers.keys())[0]])
    assert all(len(method_to_header_numbers[method]) == l0 for method in method_to_header_numbers)
    assert len(x_labels) == l0
    
    markers = ['o', '^', 's', 'D', 'x', '*', '.', '+']
    for method, marker in zip(method_to_header_numbers, markers, strict=False):
        losses = get_final_val_losses(filename, method_to_header_numbers[method])
        plt.plot(x_labels, losses, marker=marker, label=method)
    plt.grid()
    plt.ylabel("Final validation loss")
    plt.xlabel(x_axis_label.capitalize())
    plt.legend()
    if draw_target_loss:
        plt.axhline(2.92, color='red', linewidth=1)
    plt.show()


def extract_lambdas_and_layers_for_results_multiple(
        filename: str,
        header_numbers: list[int | str],
) -> dict[str, dict[Literal["lambdas", "layers"], list[float] | list[int]]]:
    with open(filename, "r") as f:
        lines = f.readlines()

    if isinstance(header_numbers, dict):
        header_numbers = list(header_numbers.keys())

    results = {}
    for hnum in header_numbers:
        extract= False
        for line in lines:
            if line.strip() == f"## {hnum}":
                extract = True
                continue
            if extract and line.startswith("##"):
                break
            if extract and line.startswith("step:") and "val_loss" in line:
                line = line.split("lambdas: ")[1]
                split = line.split(" skip-layers: ")
                lambdas = ast.literal_eval(split[0].strip())
                layers = ast.literal_eval(split[1].strip())
                results[hnum] = dict(lambdas=lambdas, layers=layers)

    return results


def plot_lambdas_over_layer_multiple(
        filename: str,
        header_numbers: list[int | str],
        max_layers: int = 15,
):
    results = extract_lambdas_and_layers_for_results_multiple(filename, header_numbers)

    # Gather lambdas per layer
    layer_to_lambda: dict[int, list[float]] = defaultdict(list)
    for info in results.values():
        for lam, layer in zip(info["lambdas"], info["layers"]):
            layer_to_lambda[int(layer)].append(float(lam))

    # Keep only layers that have any data
    if not layer_to_lambda:
        raise ValueError("No lambda data found for the given inputs.")

    x = sorted(k for k in layer_to_lambda.keys() if 0 <= k < max_layers)
    data = [layer_to_lambda[layer] for layer in x]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Violin plot at exact layer positions
    parts = ax.violinplot(
        data,
        positions=x,
        showmeans=True,
        showmedians=True,
        showextrema=True,
        widths=0.8
    )
    # Make violins slightly transparent
    for body in parts['bodies']:
        body.set_alpha(0.6)

    # Scatter raw points with small horizontal jitter so density is visible
    for xi, ys in zip(x, data):
        jitter = (np.random.rand(len(ys)) - 0.5) * 0.6  # +/- 0.3 jitter
        ax.scatter(np.full(len(ys), xi) + jitter, ys, s=12, alpha=0.6)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Lambda")
    ax.set_xticks(x)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.show()


def get_first_step_and_time_below_target_loss(
        filename: str,
        header_number: str | int,
) -> tuple[int, float]:
    parsed, _, _ = get_val_losses(filename=filename, header_numbers=[header_number])
    parsed = parsed[header_number]
    is_below = [loss < 2.92 for loss in parsed["loss"]]
    
    try:
        idx_first_below = is_below.index(True)
    except ValueError:
        return -1, -1.0

    if idx_first_below == 0:
        return 0, 0.0
    
    idx0 = idx_first_below - 1
    idx1 = idx_first_below
    loss0 = parsed["loss"][idx0]
    loss1 = parsed["loss"][idx1]

    step0 = parsed["step"][idx0]
    step1 = parsed["step"][idx1]
    time0 = parsed["time"][idx0]
    time1 = parsed["time"][idx1]
    threshold = 2.92

    delta_loss = loss1 - loss0
    if delta_loss == 0:
        return step1, time1

    fraction = (threshold - loss0) / delta_loss
    fraction = min(max(fraction, 0.0), 1.0)
    interpolated_step = step0 + fraction * (step1 - step0)
    interpolated_time = time0 + fraction * (time1 - time0)

    return math.ceil(round(interpolated_step)), interpolated_time


def get_hnum_crossing_target_loss_sorted_by_time():
    results = {
        hnum: get_first_step_and_time_below_target_loss("results-multiple.md", hnum)
        for hnum in [
            f"8000-add-skip-multiple-{num_skips}-method-{method}-0"
            for method in ["btw", "wtb", "htl", "lth", "random"]
            for num_skips in range(2, 15)
        ]
    }
    results = {k: v for k, v in results.items() if v[0] >= 0 and v[1] >= 0.0}
    sorted_hnums = sorted(
        list(results.keys()),
        key=lambda hnum: results[hnum][1]  # sort by time
    )
    sorted_info = []
    for hnum in sorted_hnums:
        method = f"{hnum.split('-method-')[1].split('-')[0]}"
        num_layers = int(f"{hnum.split('multiple-')[1].split('-method')[0]}")
        sorted_info.append(
            dict(
                num_layers=num_layers,
                method={
                    "btw": "Best to Worst",
                    "wtb": "Worst to Best",
                    "htl": "Late to Early",
                    "lth": "Early to Late",
                    "random": "Random",
                }[method],
                time=round(results[hnum][1], 4),
                step=results[hnum][0],
            )
        )
    return sorted_info


def get_losses_and_times_sorted_by_category(
        filename: str,
        header_numbers: list[int | str] | Literal["all", "htl", "lth", "btw", "wtb", "random"],
        by: Literal["distance_between_layers", "sorting_method", "num_layers", "distance_from_output"],
) -> dict[str | int, list[str]]:
    if isinstance(header_numbers, str):
        with open(filename, "r") as f:
            lines = f.readlines()
        hnums = []
        for line in lines:
            if line.strip().startswith("##"):
                hnums.append(line.replace("##", "").strip())
        if header_numbers != "all":
            hnums = [hnum for hnum in hnums if header_numbers in hnum]
        header_numbers = hnums
    data = extract_lambdas_and_layers_for_results_multiple(
        filename=filename,
        header_numbers=header_numbers,
    )
    if by == "distance_between_layers":
        times = get_final_times(filename, header_numbers)
        losses = get_final_val_losses(filename, header_numbers)
        mean_dists = []
        for hnum in header_numbers:
            layers = sorted(data[hnum]["layers"])
            dists = [l2 - l1 for l1, l2 in zip(layers[:-1], layers[1:])]
            mean_dists.append(sum(dists) / len(dists))
        dist_to_data = {
            d: dict(time=times[i], loss=losses[i], hnum=[header_numbers[i]])
            for i, d in enumerate(mean_dists)
        }
        return {d: dist_to_data[d] for d in sorted(mean_dists)}  # sort the dict
    elif by == "sorting_method":
        assert all(isinstance(hnum, str) for hnum in header_numbers)
        methods = {hnum.split("method")[1].split("-")[1] for hnum in header_numbers}
        results = {}
        for method in methods:
            hnums = [hnum for hnum in header_numbers if method in hnum]
            times = get_final_times(filename, hnums)
            losses = get_final_val_losses(filename, hnums)
            results[method] = dict(
                time=sum(times) / len(times),
                loss=sum(losses) / len(losses),
                hnums=hnums,
            )
        return results
    elif by == "num_layers":
        results = {}
        nums = [int(hnum.split("multiple-")[1].split("-")[0]) for hnum in header_numbers]
        for num in nums:
            hnums = [hnum for hnum in header_numbers if f"-{num}-" in hnum]
            times = get_final_times(filename, hnums)
            losses = get_final_val_losses(filename, hnums)
            results[num] = dict(
                time=sum(times)/len(times),
                loss=sum(losses)/len(losses),
                hnums=hnums,
            )
        return results
    elif by == "distance_from_output":
        times = get_final_times(filename, header_numbers)
        losses = get_final_val_losses(filename, header_numbers)
        mean_dists = []
        for hnum in header_numbers:
            layers = sorted(data[hnum]["layers"])
            dists = [15 - layer for layer in layers]
            mean_dists.append(sum(dists) / len(dists))
        dist_to_data = {
            d: dict(time=times[i], loss=losses[i], hnum=[header_numbers[i]])
            for i, d in enumerate(mean_dists)
        }
        return {d: dist_to_data[d] for d in sorted(mean_dists)}  # sort the dict
    raise ValueError(
        f"{by=} is not a valid value; "
        "must be 'distance_between_layers' or 'sorting_method'"
    )


def plot_final_val_loss_by_category(
        filename: str,
        header_numbers: list[int | str] | Literal["all", "htl", "lth", "btw", "wtb", "random"],
        by: Literal["distance_between_layers", "sorting_method", "num_layers", "distance_from_output"],
        do_print: bool = False,
):
    results = get_losses_and_times_sorted_by_category(
        filename=filename,
        header_numbers=header_numbers,
        by=by,
    )

    if do_print:
        import rich
        print(f"BY {by.upper()}:")
        rich.print(results)

    if by == "sorting_method":
        keys = sorted(list(results.keys()), key=lambda x: ["htl", "lth", "btw", "wtb", "random"].index(x))
    else:
        keys = sorted(list(results.keys()))
    if by == "sorting_method":
        x = [
            {
                "btw": "Best to Worst",
                "wtb": "Worst to Best",
                "htl": "Late to Early",
                "lth": "Early to Late",
                "random": "Random",
            }[key]
            for key in keys
        ]
    else:
        x = keys
    y = [results[dpt]["loss"] for dpt in keys]
    if by == "sorting_method":
        plt.bar(x, y)
    else:
        plt.plot(x, y)
    plt.xlabel(by.replace("_", " ").capitalize())
    plt.ylabel("Final val loss")
    plt.grid()
    if header_numbers == "all" and by == "sorting_method":
        plt.ylim(2.918, 2.9205)
    elif header_numbers == "all" and by == "num_layers":
        plt.ylim(2.9185, 2.92)
    plt.show()


def plot_loss_by_sorting_method_over_num_layers(
        methods: list[Literal["btw", "wtb", "htl", "lth", "random", "all"]] = "all",
        take_average: bool = False,
):
    if methods == "all" or methods == ["all"]:
        methods = ["btw", "wtb", "htl", "lth", "random"]
    else:
        methods = list(set(methods))
    assert all(method in ["btw", "wtb", "htl", "lth", "random"] for method in methods), f"{methods=}"
    by_meth = get_losses_and_times_sorted_by_category(
        filename="results-multiple.md",
        header_numbers="all",
        by="sorting_method",
    )
    if take_average:
        layer_sums: defaultdict[int, float] = defaultdict(float)
        layer_counts: defaultdict[int, int] = defaultdict(int)
    for method in methods:
        results = by_meth[method]
        by_layer = get_losses_and_times_sorted_by_category(
            filename="results-multiple.md",
            header_numbers=results["hnums"],
            by="num_layers",
        )
        x = sorted(list(by_layer.keys()))
        y = [by_layer[layer]["loss"] for layer in x]
        if take_average:
            for layer, loss in zip(x, y):
                layer_sums[layer] += loss
                layer_counts[layer] += 1
        else:
            label = {
                "btw": "Best to Worst",
                "wtb": "Worst to Best",
                "htl": "Late to Early",
                "lth": "Early to Late",
                "random": "Random",
            }[method]
            plt.plot(x, y, label=label)
    if take_average:
        x = sorted(layer_sums.keys())
        y = [layer_sums[layer] / layer_counts[layer] for layer in x]
        plt.plot(x, y, label="Average")
    
    plt.grid()
    plt.legend()
    plt.xlabel("Number of Layers skipped to Output")
    plt.ylabel("Final validation loss")
    plt.show()


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

    # plot_val_loss(
    #     filename="results.md",
    #     header_numbers=[f"3 {i}" for i in range(5)] + [f"0 {i}" for i in range(5)],
    #     average_over={
    #         "emb-to-lmhead-skip": [f"3 {i}" for i in range(5)],
    #         "baseline": [f"0 {i}" for i in range(5)],
    #     },
    #     x_axis="time",
    # )
    # plot_val_loss(
    #     filename="results.md",
    #     header_numbers=[f"{i} {j}" for i in [16, 18] for j in range(4)],
    #     average_over={
    #         "16": ["16 0", "16 1", "16 2", "16 3"],
    #         "18": ["18 0", "18 1", "18 2", "18 3"],
    #     },
    #     x_axis="step",
    # )
    # plot_val_loss(
    #     filename="results.md",
    #     header_numbers=["16 0 (long)", "17 0 (long)", "16 1 (long)", "17 1 (long)"],
    #     average_over={
    #         "baseline": ["16 0 (long)", "16 1 (long)"],
    #         "add x00 to output latent": ["17 0 (long)", "17 1 (long)"],
    #     }
    # )
    # plot_final_losses_over_names(
    #     filename="results.md",
    #     header_numbers=[f"5001-add-normed-skip{i}-to-x-out-0" for i in range(15)],
    #     names=[f"Layer {i}" for i in range(15)],
    #     xaxis_label="Skip from layer",
    #     yaxis_label="Final validation loss",
    # )
    # plot_val_loss(
    #     filename="results.md",
    #     header_numbers=[2312, "0 0"],
    #     x_axis="time",
    # )
    # plot_val_loss(
    #     header_numbers=["5001-add-normed-skip11-to-x-out-0"],
    #     filename="results.md",
    #     x_axis="step",
    #     legend=False,
    # )

    # plot_val_loss(
    #     filename="results.md",
    #     header_numbers=["7000-updated-record-0", "7001-add-skip11-from-updated-record-0"],
    #     average_over={
    #         "Baseline PR#137": ["7000-updated-record-0"],
    #         "+ skip from layer 11": ["7001-add-skip11-from-updated-record-0"],
    #     },
    #     x_axis="time",
    # )

    # # CONCAT VS ADD
    # plot_final_losses_over_names_by_method(
    #     filename="results.md",
    #     method_to_header_numbers={
    #         "concat": [f"23{i:02d}" for i in range(15)],
    #         "add": [f"5001-add-normed-skip{i}-to-x-out-0" for i in range(15)],
    #     },
    #     x_labels=[str(i) for i in range(15)],
    #     x_axis_label="Layer",
    # )

    # # CONCAT ONLY
    # plot_final_losses_over_names(
    #     filename="results.md",
    #     header_numbers=[21, 22] + [f"23{i:02d}" for i in range(15)] + [24],
    #     names=["Input emb", "Extra emb"] + [f"Layer {i}" for i in range(15)] + ["Layer 15"],
    #     yaxis_label="Final validation loss",
    # )

    # # VAL LOSSES CONCAT
    # plot_val_loss(
    #     filename="results.md",
    #     header_numbers=[2312],
    #     average_over={"Concat layer 12 outputs to output latents": [2312]},
    #     x_axis="time",
    #     legend=False,
    # )

    # # T-TEST ANALYSIS
    # hnums = [f"7002-add-skip11-record-from-updated-record-5550steps-{i}" for i in range(22)]
    # losses = get_final_val_losses(
    #     filename="results-t-test.md",
    #     header_numbers=hnums,
    # )
    # loss_stats = test_mean_below(losses)
    # times = get_final_times(
    #     filename="results-t-test.md",
    #     header_numbers=hnums,
    # )
    # time_stats = test_mean_below(times)

    # print(losses)
    # print(
    #     f"- Mean: {loss_stats['sample_mean']}\n",
    #     f"- Std: {loss_stats['sample_std']}\n",
    #     f"- P-value: {loss_stats['p_value']}\n",
    # )
    # print(times)
    # print(
    #     f"- Mean: {time_stats['sample_mean']}\n",
    #     f"- Std: {time_stats['sample_std']}\n",
    # )

    # # SORT THE HEADER NUMBERS BY LOSS
    # hnums = [f"5001-add-normed-skip{i}-to-x-out-0" for i in range(15)]
    # losses = get_final_val_losses("results.md", hnums)
    # indices = np.argsort(losses).tolist()
    # sorted_hnums = []
    # for idx in indices:
    #     sorted_hnums.append(hnums[idx])
    # print(indices)
    # print(sorted_hnums)

    # # MULITPLE LAMBDAS
    # results = get_val_losses_and_lambdas(
    #     filename="results-multiple.md",
    #     header_numbers=["8000-add-skip-multiple-7-method-btw-0"],
    # )[0]
    # import rich
    # rich.print(results)

    # # FINAL LOSS BY CATEGORY
    # plot_final_val_loss_by_category(
    #     filename="results-multiple.md",
    #     header_numbers="all",  # [f"8000-add-skip-multiple-10-method-{meth}-0" for meth in ["btw", "wtb", "htl", "lth", "random"]],
    #     by="distance_between_layers",
    #     do_print=True,
    # )
    # plot_final_val_loss_by_category(
    #     filename="results-multiple.md",
    #     header_numbers="all",
    #     by="distance_from_output",
    #     do_print=False,
    # )
    # plot_final_val_loss_by_category(
    #     filename="results-multiple.md",
    #     header_numbers="all",
    #     by="num_layers",
    # )

    # # VAL LOSSES
    # for num_skips in range(2, 15):
    #     plot_val_loss(
    #         filename="results-multiple.md",
    #         header_numbers=[
    #             f"8000-add-skip-multiple-{num_skips}-method-{method}-0"
    #             for method in ["btw", "wtb", "htl", "lth", "random"]
    #         ],
    #         average_over={
    #             "Best to Worst": [f"8000-add-skip-multiple-{num_skips}-method-btw-0"],
    #             "Worst to Best": [f"8000-add-skip-multiple-{num_skips}-method-wtb-0"],
    #             "Late to Early": [f"8000-add-skip-multiple-{num_skips}-method-htl-0"],
    #             "Early to Late": [f"8000-add-skip-multiple-{num_skips}-method-lth-0"],
    #             "Random": [f"8000-add-skip-multiple-{num_skips}-method-random-0"],
    #         },
    #         title=f"{num_skips} Layers skipped to Output",
    #     )

    # # LOSS OVER LAYERS BY METHOD
    # plot_loss_by_sorting_method_over_num_layers(
    #     methods=["all"],
    #     take_average=True,
    # )

    # # MISSING HNUMS
    # with open("results-multiple.md", "r") as f:
    #     lines = f.readlines()
    # hnums = []
    # for line in lines:
    #     if line.strip().startswith("##"):
    #         hnums.append(line.replace("##", "").strip())
    # target_hnums = [
    #     f"8000-add-skip-multiple-{num_skips}-method-{method}-0"
    #     for method in ["btw", "wtb", "htl", "lth", "random"]
    #     for num_skips in range(2, 15)
    # ]
    # print(set(target_hnums) - set(hnums))

    # LAMBDAS
    # parsed, _, _ = get_val_losses_and_lambdas_record(
    #     filename="results-t-test.md",
    #     header_numbers=[f"7002-add-skip11-record-from-updated-record-5550steps-{i}" for i in range(22)],
    #     average_over={"all": [f"7002-add-skip11-record-from-updated-record-5550steps-{i}" for i in range(22)]},
    # )
    # import rich
    # rich.print(parsed)

    # plot_lambdas_record(
    #     filename="results-t-test.md",
    #     header_numbers=[f"7002-add-skip11-record-from-updated-record-5550steps-{i}" for i in range(22)],
    #     plot_all=True,
    # )

    # plot_final_val_loss_by_category(
    #     filename="results-multiple.md",
    #     header_numbers="all",
    #     by="sorting_method"
    # )

    # plot_lambdas_over_layer_multiple(
    #     "results-multiple.md",
    #     header_numbers=[
    #         f"8000-add-skip-multiple-{num_skips}-method-{method}-0"
    #         for method in ["btw", "wtb", "htl", "lth", "random"]
    #         for num_skips in range(2, 14)
    #     ]
    # )

    # plot_val_loss(
    #     filename="results.md",
    #     header_numbers=[
    #         f"7007-add-skip11-record-from-updated-record-fixed-alpha-{i}" for i in range(5)
    #     ] + ["7001-add-skip11-from-updated-record-0"] + [f"7000-updated-record-{i}" for i in range(5)],
    #     average_over={
    #         "Baseline w/o skip": [f"7000-updated-record-{i}" for i in range(5)],
    #         "Baseline w/ skip": ["7001-add-skip11-from-updated-record-0"],
    #         "Fixed lambda_x": [f"7007-add-skip11-record-from-updated-record-fixed-alpha-{i}" for i in range(5)],
    #     },
    #     x_axis="step",
    # )

    # plot_val_loss(
    #     filename="results.md",
    #     header_numbers=[
    #         f"7006-add-skip11-record-from-updated-record-detach-latents-{i}" for i in range(5)
    #     ] + ["7001-add-skip11-from-updated-record-0"] + [f"7000-updated-record-{i}" for i in range(5)],
    #     average_over={
    #         "Baseline w/o skip": [f"7000-updated-record-{i}" for i in range(5)],
    #         "Baseline w/ skip": ["7001-add-skip11-from-updated-record-0"],
    #         "Detached Latents": [f"7006-add-skip11-record-from-updated-record-detach-latents-{i}" for i in range(5)],
    #     },
    #     x_axis="step",
    # )

    # plot_val_loss(
    #     filename="results.md",
    #     header_numbers=[
    #         f"7007-add-skip11-record-from-updated-record-fixed-alpha-{i}" for i in range(5)
    #     ] + ["7001-add-skip11-from-updated-record-0"] + [f"7000-updated-record-{i}" for i in range(5)],
    #     average_over={
    #         "Baseline w/o skip": [f"7000-updated-record-{i}" for i in range(5)],
    #         "Baseline w/ skip": ["7001-add-skip11-from-updated-record-0"],
    #         "Fixed lamba_x": [f"7007-add-skip11-record-from-updated-record-fixed-alpha-{i}" for i in range(5)],
    #     },
    #     x_axis="step",
    # )

    plot_val_loss(
        filename="results.md",
        header_numbers=[
            f"7012-add-skip11-record-from-updated-record-relu-linear-{i}" for i in range(5)
        ] + ["7001-add-skip11-from-updated-record-0"] + [f"7000-updated-record-{i}" for i in range(5)],
        average_over={
            "Baseline w/o skip": [f"7000-updated-record-{i}" for i in range(5)],
            "Baseline w/ skip": ["7001-add-skip11-from-updated-record-0"],
            "ReLU + Linear Layer": [f"7012-add-skip11-record-from-updated-record-relu-linear-{i}" for i in range(5)],
        },
        x_axis="step",
    )

    # plot_val_loss(
    #     filename="results.md",
    #     header_numbers=[
    #         f"7009-add-skip11-record-from-updated-record-relu-{i}" for i in range(5)
    #     ] + [
    #         f"7011-add-skip11-record-from-updated-record-tanh-{i}" for i in range(5)
    #     ] + ["7001-add-skip11-from-updated-record-0"] + [f"7000-updated-record-{i}" for i in range(5)],
    #     average_over={
    #         "Baseline w/o skip": [f"7000-updated-record-{i}" for i in range(5)],
    #         "Baseline w/ skip": ["7001-add-skip11-from-updated-record-0"],
    #         "ReLU": [f"7009-add-skip11-record-from-updated-record-relu-{i}" for i in range(5)],
    #         "TanH": [f"7011-add-skip11-record-from-updated-record-tanh-{i}" for i in range(5)],
    #     },
    #     x_axis="step",
    # )

    

    # plot_val_loss(
    #     filename="results.md",
    #     header_numbers=["7005-add-skip11-record-from-updated-record-norm-sum-norm-0"]
    #         + ["7002-add-skip11-record-from-updated-record-0"],
    #     average_over={
    #         "Baseline w/ skip (Record)": ["7002-add-skip11-record-from-updated-record-0"],
    #         "Norm-sum-norm": ["7005-add-skip11-record-from-updated-record-norm-sum-norm-0"],
    #     },
    #     x_axis="time",
    # )
