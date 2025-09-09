
import matplotlib.pyplot as plt
import numpy as np


def plot_val_loss(
        header_numbers: list[int | str] | dict[int | str, str],
        filename: str,
        x_axis: str = "step",
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

    for i, hnum in enumerate(header_numbers):
        description = f": {descriptions[i]}" if descriptions[i] else ""
        plt.plot(parsed[hnum][x_axis], parsed[hnum]["loss"], label=f"{hnum}{description}")
    plt.xlabel("step" if x_axis == "step" else "time (s)")
    plt.ylabel("val_loss")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
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
    plot_val_loss(
        filename="results.md",
        header_numbers=["16 0 (long)", "17 0 (long)", "16 1 (long)", "17 1 (long)"],
        average_over={
            16: ["16 0 (long)", "16 1 (long)"],
            17: ["17 0 (long)", "17 1 (long)"],
        }
    )
