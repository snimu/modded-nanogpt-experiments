
import argparse
import os


def extract_vallosses(subdir: str, name: str, offset: int = 0):
    files = sorted(os.listdir(subdir))
    results = ""
    for i, file in enumerate(files):
        title = f"## {name} {i + offset}"
        with open(os.path.join(subdir, file), "r") as f:
            lines = f.readlines()
        lines = [line for line in lines if "val_loss" in line]

        results += f"{title}\n\n{'\n'.join(lines)}\n\n"

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdir", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    results = extract_vallosses(args.subdir, args.name, args.offset)

    with open(f"{args.subdir}/vallosses-{args.name}.md", "w") as f:
        f.write(results)
