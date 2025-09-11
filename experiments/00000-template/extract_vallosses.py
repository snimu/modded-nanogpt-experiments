
import argparse
import os


def extract_vallosses(subdir: str, name: str, offset: int = 0):
    files = sorted(os.listdir("logs/" + subdir))
    results = ""
    for i, file in enumerate(files):
        title = f"## {name} {i + offset}"
        with open(os.path.join("logs/" + subdir, file), "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if "val_loss" in line and line.startswith("step:") and line.strip()]
        trace = '\n'.join(lines)
        results += f"{title}\n\n{trace}\n\n"

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdir", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    results = extract_vallosses(args.subdir, args.name, args.offset)

    with open(f"logs/{args.subdir}/vallosses-{args.name}.md", "w") as f:
        f.write(results)
