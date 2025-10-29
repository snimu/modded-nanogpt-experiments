
import re
import sys
import argparse
import os
from typing import Literal, Callable, Iterable, Union

import seaborn as sns
import polars as pl
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
        title: str | None = None
):
    parsed, header_numbers, descriptions = get_val_losses(header_numbers, filename, average_over)

    for i, hnum in enumerate(header_numbers):
        description = f": {descriptions[i]}" if descriptions[i] else ""
        plt.plot(parsed[hnum][x_axis], parsed[hnum]["loss"], label=f"{hnum}{description}")
    plt.xlabel("step" if x_axis == "step" else "time (s)")
    plt.ylabel("val_loss")
    if title:
        plt.title(title)
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
        results[subdir] = {"loss": round(loss, 4), "time": round(time, 2)}
        formatted_results += f"\n{subdir} —— Loss: {results[subdir]['loss']}, Time: {results[subdir]['time']}"

    formatted_results += f"\n{sep}\n\n"
    return results, formatted_results


# --- helper to parse num_ve / num_embs_per_ve from a header ---
_num_re = re.compile(r"num_ve(\d+).*?num_embs_per_ve(\d+)")

def _parse_ve_and_embs(header: str) -> tuple[int, int]:
    m = _num_re.search(header)
    if not m:
        raise ValueError(f"Could not parse num_ve/num_embs_per_ve from header: {header}")
    return int(m.group(1)), int(m.group(2))


def plot_ve_embs_heatmap(
    headers: Iterable[str],
    filename: str,
    *,
    reducer: Union[str, Callable[[np.ndarray], float]] = "final",  # 'final' | 'min' | callable
    annotate: bool = True,
    fmt: str = ".3g",
    cmap: str = "viridis",
    title: str | None = None,
    as_percent_over: Literal["none", "row", "col"] = "none",
    get_val_losses_fn=None,  # inject your existing function (or it will import from globals)
):
    """
    Build a heatmap of val_loss with rows=num_ve and cols=num_embs_per_ve.
    Uses Polars for the table and seaborn/matplotlib for plotting.

    Parameters
    ----------
    headers : iterable of str
        Header names present in the markdown file; each must encode num_ve & num_embs_per_ve.
    filename : str
        Path to the markdown log file.
    reducer : {'final', 'min'} or callable
        How to collapse a run's loss curve to a scalar.
    as_percent_over : {'none', 'row', 'col'}
        'none' -> plot raw reduced values.
        'row'  -> within each row (fixed num_ve), divide by that row's max (ignoring NaN),
                 then *100, so every row peaks at 100.
        'col'  -> within each column (fixed num_embs_per_ve), same idea but column-wise.
    """

    # Reuse your parser
    if get_val_losses_fn is None:
        # assumes get_val_losses is in scope
        get_val_losses_fn = get_val_losses

    parsed, _, _ = get_val_losses_fn(list(headers), filename, average_over=None)

    def _reduce_losses(losses: list[float]) -> float:
        if not losses:  # missing data
            return float("nan")
        arr = np.asarray(losses, dtype=float)
        if isinstance(reducer, str):
            if reducer == "final":
                return float(arr[-1])
            elif reducer == "min":
                return float(np.min(arr))
            else:
                raise ValueError("reducer must be 'final', 'min', or a callable")
        else:
            return float(reducer(arr))

    # Build long-form rows via Polars
    rows = []
    for h in headers:
        ve, embs = _parse_ve_and_embs(h)
        value = _reduce_losses(parsed[h]["loss"])
        rows.append({"num_ve": ve, "num_embs_per_ve": embs, "val": value})

    df_long = pl.DataFrame(rows)

    # Pivot to wide matrix (rows=num_ve, cols=num_embs_per_ve)
    df_pivot = (
        df_long
        .pivot(values="val", index="num_ve", on="num_embs_per_ve", aggregate_function="mean")
        .sort("num_ve")
    )

    # Ensure columns are ordered numerically (Polars names pivot columns by their value)
    col_nums = sorted([int(c) for c in df_pivot.columns if c != "num_ve"])
    ordered_cols = ["num_ve"] + [str(c) for c in col_nums]
    df_pivot = df_pivot.select(ordered_cols)

    # Prepare matrix & ticks for seaborn (no pandas used)
    mat = df_pivot.drop("num_ve").to_numpy()  # shape [n_rows, n_cols], dtype could be float + NaN
    ytick = df_pivot["num_ve"].to_list()
    xtick = col_nums

    # --- NEW BLOCK: row/col percentage normalization ---
    if as_percent_over != "none":
        mat_norm = mat.astype(float).copy()

        if as_percent_over == "row":
            # For each row: divide by row max (ignoring NaN), then *100
            # If a row is all NaN, it stays NaN.
            for i in range(mat_norm.shape[0]):
                row = mat_norm[i, :]
                # compute max ignoring NaN
                valid = ~np.isnan(row)
                if np.any(valid):
                    row_max = np.nanmax(row)
                    # avoid divide-by-zero (if row_max == 0.0, just leave row as-is)
                    if row_max != 0.0 and not np.isnan(row_max):
                        mat_norm[i, valid] = (row[valid] / row_max) * 100.0
                # else: row all NaN -> leave as NaN

        elif as_percent_over == "col":
            # For each col: divide by col max (ignoring NaN), then *100
            for j in range(mat_norm.shape[1]):
                col = mat_norm[:, j]
                valid = ~np.isnan(col)
                if np.any(valid):
                    col_max = np.nanmax(col)
                    if col_max != 0.0 and not np.isnan(col_max):
                        mat_norm[valid, j] = (col[valid] / col_max) * 100.0
                # else: entire column NaN -> leave as NaN
        else:
            raise ValueError("as_percent_over must be 'none', 'row', or 'col'")

        # Replace mat with normalized version for plotting
        mat = mat_norm

        # Also update df_pivot so the returned frame matches what was plotted
        # We'll rebuild df_pivot's non-index columns from mat.
        # df_pivot currently has columns ["num_ve", <col_nums...>]
        new_cols = {"num_ve": df_pivot["num_ve"]}
        for j, col_name in enumerate([str(c) for c in col_nums]):
            new_cols[col_name] = mat[:, j]
        df_pivot = pl.DataFrame(new_cols)

    # Plot
    plt.figure(figsize=(1.2 * max(4, len(xtick)), 1.0 * max(3, len(ytick))))
    mask = np.isnan(mat)
    ax = sns.heatmap(
        mat,
        annot=annotate,
        fmt=fmt,
        cmap=cmap,
        cbar=True,
        mask=mask,
        xticklabels=xtick,
        yticklabels=ytick,
    )
    ax.set_xlabel("Number of Embedding Modules per Value Embedding")
    ax.set_ylabel("Number of Value Embeddings (each shared between to layers)")

    # Default title logic, now aware of reducer + normalization
    if title is None:
        red_txt = reducer if isinstance(reducer, str) else getattr(reducer, "__name__", "custom")
        if as_percent_over == "none":
            title = f"{red_txt.capitalize()} validation loss"
        elif as_percent_over == "row":
            title = f"{red_txt.capitalize()} validation loss - % of row max"
        elif as_percent_over == "col":
            title = f"{red_txt.capitalize()} validation loss - % of column max"
        else:
            # shouldn't happen, but fallback
            title = f"{red_txt.capitalize()} validation loss"

    ax.set_title(title)

    plt.tight_layout()
    plt.show()

    # Return the Polars pivot (helpful for downstream use/tests)
    return df_pivot


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

    # # MULTIPLE EMBS VIA PROJECTION
    # plot_val_loss(
    #     filename="results.md",
    #     header_numbers=[
    #         "00000-extra-embs-num_ve4-num_embs_per_ve1-0",
    #         "00001-multi-emb-via-projection-0",
    #         "00002-multi-small-emb-up-projected-embed_dim_div2-0",
    #         "00002-multi-small-emb-up-projected-embed_dim_div4-0",
    #         "00002-multi-small-emb-up-projected-embed_dim_div8-0",
    #         "00002-multi-small-emb-up-projected-embed_dim_div16-0",
    #     ],
    #     average_over={
    #         "baseline": ["00000-extra-embs-num_ve4-num_embs_per_ve1-0"],
    #         "1024->1024 projection": ["00001-multi-emb-via-projection-0"],
    #         "512->1024 projection": ["00002-multi-small-emb-up-projected-embed_dim_div2-0"],
    #         "256->1024 projection": ["00002-multi-small-emb-up-projected-embed_dim_div4-0"],
    #         "128->1024 projection": ["00002-multi-small-emb-up-projected-embed_dim_div8-0"],
    #         "64->1024 projection": ["00002-multi-small-emb-up-projected-embed_dim_div16-0"],
    #     },
    #     x_axis="time",
    # )
    # plot_val_loss(
    #     filename="results.md",
    #     header_numbers=[
    #         "00000-extra-embs-num_ve4-num_embs_per_ve1-0",
    #         "00003-multi-emb-via-projection-relu-0",
    #         "00004-multi-small-emb-up-projected-relu-embed_dim_div2-0",
    #         "00004-multi-small-emb-up-projected-relu-embed_dim_div4-0",
    #         "00004-multi-small-emb-up-projected-relu-embed_dim_div8-0",
    #         "00004-multi-small-emb-up-projected-relu-embed_dim_div16-0",
    #     ],
    #     average_over={
    #         "baseline": ["00000-extra-embs-num_ve4-num_embs_per_ve1-0"],
    #         "1024->1024 projection w/ ReLU": ["00003-multi-emb-via-projection-relu-0"],
    #         "512->1024 projection w/ ReLU": ["00004-multi-small-emb-up-projected-relu-embed_dim_div2-0"],
    #         "256->1024 projection w/ ReLU": ["00004-multi-small-emb-up-projected-relu-embed_dim_div4-0"],
    #         "128->1024 projection w/ ReLU": ["00004-multi-small-emb-up-projected-relu-embed_dim_div8-0"],
    #         "64->1024 projection w/ ReLU": ["00004-multi-small-emb-up-projected-relu-embed_dim_div16-0"],
    #     },
    #     x_axis="time",
    # )

    # # LOSS OVER NUM VE
    # plot_val_loss(
    #     filename="results.md",
    #     header_numbers=[f"00000-extra-embs-num_ve{i}-num_embs_per_ve{j}-0" for i in range(1, 5) for j in range(1, 5)],
    #     average_over={
    #         "1 ve": [f"00000-extra-embs-num_ve1-num_embs_per_ve{j}-0" for j in range(1, 5)],
    #         "2 ve": [f"00000-extra-embs-num_ve2-num_embs_per_ve{j}-0" for j in range(1, 5)],
    #         "3 ve": [f"00000-extra-embs-num_ve3-num_embs_per_ve{j}-0" for j in range(1, 5)],
    #         "4 ve": [f"00000-extra-embs-num_ve4-num_embs_per_ve{j}-0" for j in range(1, 5)],
    #     },
    # )

    # # LOSS OVER NUM EMS PER VE
    # plot_val_loss(
    #     filename="results.md",
    #     header_numbers=[f"00000-extra-embs-num_ve{i}-num_embs_per_ve{j}-0" for i in range(1, 5) for j in range(1, 5)],
    #     average_over={
    #         "1 emb per ve": [f"00000-extra-embs-num_ve{i}-num_embs_per_ve1-0" for i in range(1, 5)],
    #         "2 emb per ve": [f"00000-extra-embs-num_ve{i}-num_embs_per_ve2-0" for i in range(1, 5)],
    #         "3 emb per ve": [f"00000-extra-embs-num_ve{i}-num_embs_per_ve3-0" for i in range(1, 5)],
    #         "4 emb per ve": [f"00000-extra-embs-num_ve{i}-num_embs_per_ve4-0" for i in range(1, 5)],
    #     },
    #     x_axis="time",
    # )

    # # HEATMAP OF FINAL / BEST VAL LOSS OVER NUM VE AND NUM EMB PER VE
    # headers = [f"00000-extra-embs-num_ve{i}-num_embs_per_ve{j}-0"
    #        for i in range(1, 5) for j in range(1, 6)]

    # plot_ve_embs_heatmap(
    #     headers, filename="results.md", reducer="final", as_percent_over="none", fmt=".5g"
    # )
    # plot_ve_embs_heatmap(
    #     headers, filename="results.md", reducer="final", as_percent_over="row", fmt=".5g"
    # )
    # plot_ve_embs_heatmap(
    #     headers, filename="results.md", reducer="final", as_percent_over="col", fmt=".5g"
    # )

    # VAL LOSSES ALL RUNS
    for num_ve in range(1, 5):
        plot_val_loss(
            filename="results.md",
            header_numbers=[f"00000-extra-embs-num_ve{num_ve}-num_embs_per_ve{j}-0" for j in range(1, 6)],
            average_over={
                f"# Embeddings per ve Layer: {j}": [f"00000-extra-embs-num_ve{num_ve}-num_embs_per_ve{j}-0"]
                for j in range(1, 6)
            },
            x_axis="time",
            title=f"{num_ve} Value Embedding Layer" + ("s" if num_ve > 1 else ""),
        )