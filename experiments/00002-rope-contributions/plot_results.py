
import ast
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt


def plot_final_lambdas(
        header_numbers: list[int | str] | dict[int | str, str],
        filename: str,
        k_or_v: Literal["k", "v"],
        norm: bool = True,
):
    with open(filename, "r") as f:
        lines = f.readlines()

    if isinstance(header_numbers, dict):
        header_numbers = list(header_numbers.keys())

    parsed = {
        hnum: {
            "step": [],
            f"{k_or_v}lambda_rope": [],
            f"{k_or_v}lambda_raw": [],
        }
        for hnum in header_numbers
    }
    for hnum in header_numbers:
        extract = False
        for line in lines:
            if line.strip() == f"## {hnum}":
                extract = True
                continue
            if extract and line.startswith("##"):
                break
            if extract and line.startswith("step:"):
                if f"{k_or_v}rope_lambda_0" in line:
                    parsed[hnum]["step"].append(int(line.split("step:")[1].split("/")[0]))
                    parsed[hnum][f"{k_or_v}lambda_rope"].append(ast.literal_eval(line.split(f"{k_or_v}rope_lambda_0:")[-1].strip()))
                elif f"{k_or_v}rope_lambda_1" in line:
                    parsed[hnum][f"{k_or_v}lambda_raw"].append(ast.literal_eval(line.split(f"{k_or_v}rope_lambda_1:")[-1].strip()))
    # arrays + optional normalization
    for hnum in header_numbers:
        parsed[hnum][f"{k_or_v}lambda_rope"] = np.array(parsed[hnum][f"{k_or_v}lambda_rope"])
        parsed[hnum][f"{k_or_v}lambda_raw"] = np.array(parsed[hnum][f"{k_or_v}lambda_raw"])

        # extract the final lambdas per layer
        parsed[hnum][f"{k_or_v}lambda_rope"] = parsed[hnum][f"{k_or_v}lambda_rope"][-1, :]
        parsed[hnum][f"{k_or_v}lambda_raw"] = parsed[hnum][f"{k_or_v}lambda_raw"][-1, :]

        # normalize
        if norm:
            norm_val = np.abs(parsed[hnum][f"{k_or_v}lambda_rope"]) + np.abs(parsed[hnum][f"{k_or_v}lambda_raw"])
        else:
            norm_val = np.ones_like(parsed[hnum][f"{k_or_v}lambda_rope"])
        parsed[hnum][f"{k_or_v}lambda_rope"] = parsed[hnum][f"{k_or_v}lambda_rope"] / norm_val
        parsed[hnum][f"{k_or_v}lambda_raw"] = parsed[hnum][f"{k_or_v}lambda_raw"] / norm_val
    
    # plot the final lambdas
    for hnum in header_numbers:
        plt.plot(parsed[hnum][f"{k_or_v}lambda_rope"], label=f"{k_or_v}-lambda, rope")
        plt.plot(parsed[hnum][f"{k_or_v}lambda_raw"], label=f"{k_or_v}-lambda, raw")
    plt.xlabel("layer")
    plt.ylabel(r"% lambda" if norm else "lambda")
    plt.legend()
    plt.grid()
    plt.show()


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
    # plot_final_lambdas(
    #     header_numbers=["2025-08-13-no-valemb-0-1-2"],
    #     filename="results.md",
    #     k_or_v="v",
    #     norm=True,
    # )
    plot_final_lambdas(
        header_numbers=["2025-08-09-measure-rope-contributions"],
        filename="results.md",
        k_or_v="k",
        norm=True,
    )
