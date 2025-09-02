
with open("predictions.txt", "r") as f:
    txt = f.read()

predictions = eval(txt.split("predictions=")[1])
tokens = eval(txt.split("\n")[0].split("tokens=")[1])

def topn_preds(n):
    p = dict()
    for k, v in predictions.items():
        newv = dict()
        for i, t in tokens.items():
            if n == 1:
                newv[i] = {t: list(v[i].values())[0]}
            else:
                newv[i] = {t: tuple(list(v[i].values())[:n])}
        p[k] = newv
    return p


def tabulate_preds(predictions, ntoks: int = 16, vectors: list[str] | None = None):
    from tabulate import tabulate
    results = {k: [] for k in ["vector"] + [f"t{i}" for i in range(ntoks)]}
    results["vector"].append("input")
    tokens = [list(v.keys())[0].replace("\n", r"\n").replace(" ", "_") for v in list(predictions["x-0"].values())][:ntoks]
    for i, t in enumerate(tokens):
        results[f"t{i}"].append(t)
    for k, v in predictions.items():
        if vectors is not None and k not in vectors:
            continue
        results["vector"].append(k)
        values = list(v.values())
        for i in range(ntoks):
            token = list(values[i].values())[0].replace("\n", r"\n").replace(" ", "_")
            results[f"t{i}"].append(token)
    return tabulate(results, headers="keys", tablefmt="github")
        
        

if __name__ == "__main__":
    from rich import print
    print(tabulate_preds(topn_preds(1), ntoks=12, vectors=["ve2_o2", "ve2_o15"]))
