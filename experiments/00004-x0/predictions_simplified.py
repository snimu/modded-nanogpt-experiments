
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

if __name__ == "__main__":
    from rich import print
    print(topn_preds(1))
