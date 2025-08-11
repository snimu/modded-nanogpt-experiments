
import ast
import math
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


def _plot_unet_lambdas(
        header_numbers: list[int | str] | dict[int | str, str],
        filename: str,
):
    with open(filename, "r") as f:
        lines = f.readlines()

    if isinstance(header_numbers, dict):
        descriptions = list(header_numbers.values())
        header_numbers = list(header_numbers.keys())
    else:
        descriptions = ["" for _ in header_numbers]

    parsed = {hnum: {"step": [], "l-2-11": [], "l-4-10": [], "l-6-9": []} for hnum in header_numbers}
    for hnum in header_numbers:
        extract= False
        for line in lines:
            if line.strip() == f"## {hnum}":
                extract = True
                continue
            if extract and line.startswith("##"):
                break
            if extract and line.startswith("step:") and "unet_lambda" in line:
                parsed[hnum]["l-2-11"].append(float(line.split("unet_lambda_2_11:")[-1].split()[0].strip()))
                parsed[hnum]["l-4-10"].append(float(line.split("unet_lambda_4_10:")[-1].split()[0].strip()))
                parsed[hnum]["l-6-9"].append(float(line.split("unet_lambda_6_9:")[-1].split()[0].strip()))
                parsed[hnum]["step"].append(int(line.split("step:")[1].split("/")[0]))
    
    for i, hnum in enumerate(header_numbers):
        description = f": {descriptions[i]}" if descriptions[i] else ""
        plt.plot(parsed[hnum]["step"], parsed[hnum]["l-2-11"], label=f"unet-lambda-2-11{description}")
        plt.plot(parsed[hnum]["step"], parsed[hnum]["l-4-10"], label=f"unet-lambda-4-10{description}")
        plt.plot(parsed[hnum]["step"], parsed[hnum]["l-6-9"], label=f"unet-lambda-6-9{description}")
    plt.xlabel("step")
    plt.ylabel("unet lambda")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid()
    plt.show()


def _plot_x0_lambdas(
        header_numbers: list[int | str] | dict[int | str, str],
        filename: str,
        layers: list[int],
        norm: bool = False,
):
    with open(filename, "r") as f:
        lines = f.readlines()

    if isinstance(header_numbers, dict):
        header_numbers = list(header_numbers.keys())

    parsed = {hnum: {"step": [], "l0": [], "l1": []} for hnum in header_numbers}
    for hnum in header_numbers:
        extract = False
        for line in lines:
            if line.strip() == f"## {hnum}":
                extract = True
                continue
            if extract and line.startswith("##"):
                break
            if extract and line.startswith("step:"):
                if "x0_lambdas_0" in line:
                    parsed[hnum]["step"].append(int(line.split("step:")[1].split("/")[0]))
                    parsed[hnum]["l0"].append(ast.literal_eval(line.split("x0_lambdas_0:")[-1].strip()))
                elif "x0_lambdas_1" in line:
                    parsed[hnum]["l1"].append(ast.literal_eval(line.split("x0_lambdas_1:")[-1].strip()))

    # arrays + optional normalization
    for hnum in header_numbers:
        parsed[hnum]["l0"] = np.array(parsed[hnum]["l0"])
        parsed[hnum]["l1"] = np.array(parsed[hnum]["l1"])
        if norm:
            norm_val = np.abs(parsed[hnum]["l0"]) + np.abs(parsed[hnum]["l1"])
        else:
            norm_val = np.ones_like(parsed[hnum]["l0"])
        parsed[hnum]["l0"] = parsed[hnum]["l0"] / norm_val
        parsed[hnum]["l1"] = parsed[hnum]["l1"] / norm_val

    # ---- 3 columns, stack rows ----
    layers_sorted = sorted(layers)
    n = len(layers_sorted)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(4 * ncols, 3.2 * nrows))
    axs = np.atleast_2d(axs).ravel()

    for idx, layer in enumerate(layers_sorted):
        ax = axs[idx]
        for i, hnum in enumerate(header_numbers):
            ax.plot(
                parsed[hnum]["step"],
                parsed[hnum]["l0"][:, layer],
                label="x0-lambda",
                color="blue",
            )
            ax.plot(
                parsed[hnum]["step"],
                parsed[hnum]["l1"][:, layer],
                label="x-lambda",
                color="orange",
            )
        ax.set_title(f"Layer {layer}")
        ax.grid(True)
        ax.legend(fontsize="small")
        if norm:
            min_v = np.min(parsed[hnum]["l0"][:, layer])
            min_ve = np.min(parsed[hnum]["l1"][:, layer])
            ax.set_ylim(min(min_v, min_ve), 1)

    # hide any unused panes
    for j in range(n, nrows * ncols):
        axs[j].set_visible(False)

    fig.supxlabel("step")
    fig.supylabel(f"{'normed ' if norm else ''}lambdas")
    fig.tight_layout()
    plt.show()


def _plot_ve_lambdas(
        header_numbers: list[int | str] | dict[int | str, str],
        filename: str,
        layers: list[int],
        norm: bool = False,
):
    with open(filename, "r") as f:
        lines = f.readlines()

    if isinstance(header_numbers, dict):
        header_numbers = list(header_numbers.keys())

    parsed = {hnum: {"step": [], "v": [], "ve": []} for hnum in header_numbers}
    for hnum in header_numbers:
        extract = False
        for line in lines:
            if line.strip() == f"## {hnum}":
                extract = True
                continue
            if extract and line.startswith("##"):
                break
            if extract and line.startswith("step:"):
                if "ve_lambdas_0" in line:
                    parsed[hnum]["step"].append(int(line.split("step:")[1].split("/")[0]))
                    parsed[hnum]["v"].append(ast.literal_eval(line.split("ve_lambdas_0:")[-1].strip()))
                elif "ve_lambdas_1" in line:
                    parsed[hnum]["ve"].append(ast.literal_eval(line.split("ve_lambdas_1:")[-1].strip()))

    # arrays + optional normalization
    for hnum in header_numbers:
        parsed[hnum]["v"] = np.array(parsed[hnum]["v"])
        parsed[hnum]["ve"] = np.array(parsed[hnum]["ve"])
        if norm:
            norm_val = np.abs(parsed[hnum]["v"]) + np.abs(parsed[hnum]["ve"])
        else:
            norm_val = np.ones_like(parsed[hnum]["v"])
        parsed[hnum]["v"] = parsed[hnum]["v"] / norm_val
        parsed[hnum]["ve"] = parsed[hnum]["ve"] / norm_val

    # ---- 3 columns, stack rows ----
    layers_sorted = sorted(layers)
    n = len(layers_sorted)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(4 * ncols, 3.2 * nrows))
    axs = np.atleast_2d(axs).ravel()

    for idx, layer in enumerate(layers_sorted):
        ax = axs[idx]
        for i, hnum in enumerate(header_numbers):
            ax.plot(
                parsed[hnum]["step"],
                parsed[hnum]["v"][:, layer],
                label="v-lambda",
                color="blue",
            )
            ax.plot(
                parsed[hnum]["step"],
                parsed[hnum]["ve"][:, layer],
                label="ve-lambda",
                color="orange",
            )
        ax.set_title(f"Layer {layer}")
        ax.grid(True)
        ax.legend(fontsize="small")
        if norm:
            min_v = np.min(parsed[hnum]["v"][:, layer])
            min_ve = np.min(parsed[hnum]["ve"][:, layer])
            ax.set_ylim(min(min_v, min_ve), 1)

    # hide any unused panes
    for j in range(n, nrows * ncols):
        axs[j].set_visible(False)

    fig.supxlabel("step")
    fig.supylabel(f"{'normed ' if norm else ''}lambdas")
    fig.tight_layout()
    plt.show()


def plot_lambdas(
        which: Literal["unet", "x0", "ve"],
        header_numbers: list[int | str] | dict[int | str, str],
        filename: str,
        layers: list[int] | None = None,
        norm: bool = False,
):
    if which == "unet":
        _plot_unet_lambdas(header_numbers, filename)
    elif which == "x0":
        assert layers is not None
        _plot_x0_lambdas(header_numbers, filename, layers, norm)
    elif which == "ve":
        assert layers is not None
        _plot_ve_lambdas(header_numbers, filename, layers, norm)


def _plot_final_x0_lambdas(
        header_numbers: list[int | str] | dict[int | str, str],
        filename: str,
        norm: bool = False,
):
    with open(filename, "r") as f:
        lines = f.readlines()

    if isinstance(header_numbers, dict):
        header_numbers = list(header_numbers.keys())

    parsed = {hnum: {"step": [], "l0": [], "l1": []} for hnum in header_numbers}
    for hnum in header_numbers:
        extract = False
        for line in lines:
            if line.strip() == f"## {hnum}":
                extract = True
                continue
            if extract and line.startswith("##"):
                break
            if extract and line.startswith("step:"):
                if "x0_lambdas_0" in line:
                    parsed[hnum]["step"].append(int(line.split("step:")[1].split("/")[0]))
                    parsed[hnum]["l0"].append(ast.literal_eval(line.split("x0_lambdas_0:")[-1].strip()))
                elif "x0_lambdas_1" in line:
                    parsed[hnum]["l1"].append(ast.literal_eval(line.split("x0_lambdas_1:")[-1].strip()))

    # arrays + optional normalization
    for hnum in header_numbers:
        parsed[hnum]["l0"] = np.array(parsed[hnum]["l0"])
        parsed[hnum]["l1"] = np.array(parsed[hnum]["l1"])
        if norm:
            norm_val = np.abs(parsed[hnum]["l0"]) + np.abs(parsed[hnum]["l1"])
        else:
            norm_val = np.ones_like(parsed[hnum]["l0"])
        parsed[hnum]["l0"] = parsed[hnum]["l0"] / norm_val
        parsed[hnum]["l1"] = parsed[hnum]["l1"] / norm_val

        # extract the final lambdas per layer
        parsed[hnum]["l0"] = parsed[hnum]["l0"][-1, :]
        parsed[hnum]["l1"] = parsed[hnum]["l1"][-1, :]
    
    # plot the final lambdas
    for hnum in header_numbers:
        plt.plot(parsed[hnum]["l0"], label="x0-lambda final")
        plt.plot(parsed[hnum]["l1"], label="x-lambda final")
    plt.xlabel("layer")
    plt.ylabel(f"{'normed ' if norm else ''}lambda")
    plt.legend()
    plt.grid()
    plt.show()


def _plot_final_ve_lambdas(
        header_numbers: list[int | str] | dict[int | str, str],
        filename: str,
        norm: bool = True,
):
    with open(filename, "r") as f:
        lines = f.readlines()

    if isinstance(header_numbers, dict):
        header_numbers = list(header_numbers.keys())

    parsed = {hnum: {"step": [], "v": [], "ve": []} for hnum in header_numbers}
    for hnum in header_numbers:
        extract = False
        for line in lines:
            if line.strip() == f"## {hnum}":
                extract = True
                continue
            if extract and line.startswith("##"):
                break
            if extract and line.startswith("step:"):
                if "ve_lambdas_0" in line:
                    parsed[hnum]["step"].append(int(line.split("step:")[1].split("/")[0]))
                    parsed[hnum]["v"].append(ast.literal_eval(line.split("ve_lambdas_0:")[-1].strip()))
                elif "ve_lambdas_1" in line:
                    parsed[hnum]["ve"].append(ast.literal_eval(line.split("ve_lambdas_1:")[-1].strip()))

    # arrays + optional normalization
    for hnum in header_numbers:
        parsed[hnum]["v"] = np.array(parsed[hnum]["v"])
        parsed[hnum]["ve"] = np.array(parsed[hnum]["ve"])

        # extract the final lambdas per layer
        parsed[hnum]["v"] = parsed[hnum]["v"][-1, :]
        parsed[hnum]["ve"] = parsed[hnum]["ve"][-1, :]

        # normalize
        if norm:
            norm_val = np.abs(parsed[hnum]["v"]) + np.abs(parsed[hnum]["ve"])
        else:
            norm_val = np.ones_like(parsed[hnum]["v"])
        parsed[hnum]["v"] = parsed[hnum]["v"] / norm_val
        parsed[hnum]["ve"] = parsed[hnum]["ve"] / norm_val
    
    # plot the final lambdas
    for hnum in header_numbers:
        v_ve_x = np.array([0, 1, 2, 13, 14, 15])
        v_x = np.array([3, 4, 5, 6, 8, 9, 10, 11, 12])
        w = 0.4

        v_ve_y = [parsed[hnum]["v"][i] for i in v_ve_x]
        v_y = [parsed[hnum]["v"][i] for i in v_x]
        plt.bar(v_ve_x + w/2, v_ve_y, label="v-lambda, v+ve", color="blue", width=w)
        plt.bar(v_x + w/2, v_y, label="v-lambda, v only", color="blue", alpha=0.5, width=w)
        v_ve_y = [parsed[hnum]["ve"][i] for i in v_ve_x]
        v_y = [parsed[hnum]["ve"][i] for i in v_x]
        plt.bar(v_ve_x - w/2, v_ve_y, label="ve-lambda, ve only", color="orange", width=w)
        plt.bar(v_x - w/2, v_y, label="ve-lambda, v only", color="orange", alpha=0.5, width=w)
    plt.xlabel("layer")
    plt.ylabel(r"% v-lambda" if norm else "lambda")
    plt.legend()
    plt.grid()
    plt.show()


def plot_final_lambdas(
        which: Literal["x0", "ve"],
        header_numbers: list[int | str] | dict[int | str, str],
        filename: str,
        norm: bool = False,
):
    if which == "x0":
        _plot_final_x0_lambdas(header_numbers, filename, norm)
    elif which == "ve":
        _plot_final_ve_lambdas(header_numbers, filename, norm)


def plot_val_loss(
        header_numbers: list[int | str] | dict[int | str, str],
        filename: str,
        x_axis: Literal["step", "time"] = "step",
):
    with open(filename, "r") as f:
        lines = f.readlines()

    if isinstance(header_numbers, dict):
        header_numbers = list(header_numbers.keys())

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
    
    for hnum in header_numbers:
        plt.plot(parsed[hnum][x_axis], parsed[hnum]["loss"])
    plt.xlabel("step" if x_axis == "step" else "time (s)")
    plt.ylabel("val_loss")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # for layers in [
    #     [0, 1, 2, 3],
    #     [4, 5, 6, 7],
    #     [8, 9, 10, 11],
    #     [12, 13, 14, 15],
    # ]:
    #     plot_lambdas(
    #         which="ve",
    #         header_numbers=["2025-08-09-lambdas"],
    #         filename="results.md",
    #         layers=layers,
    #         norm=False,
    #     )

    # plot_final_lambdas(
    #     which="ve",
    #     header_numbers=["2025-08-09-lambdas"],
    #     filename="results.md",
    #     norm=True,
    # )

    # plot_lambdas("unet", ["2025-08-09-lambdas"], "results.md")

    plot_val_loss(["2025-08-09-lambdas"], "results.md", "step")
