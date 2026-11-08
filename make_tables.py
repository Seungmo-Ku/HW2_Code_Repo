"""
논문 표 (results/*.csv, logs/analysis*.log -> results/tables.tex)
숫자만 채운 LaTeX 표. 캡션은 직접 작성 (TODO 로 남겨둠)

사용법: python make_tables.py
"""
import csv, os, re
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
UNITS = ["eojeol", "morpheme", "syllable"]
LABEL = {"eojeol": "Eojeol", "morpheme": "Morpheme", "syllable": "Syllable"}
GEN = {"claude-haiku-5-5": "Haiku", "gemini-3.5-flash-lite": "Gemini", "gpt-5.4-mini-2026-03-17": "GPT"}


def unseen(path):
    by = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["train_gen"] != r["test_gen"]:
            by[(r["unit"], int(r["n"]))][r["seed"]].append(float(r["auroc"]))
    return {k: np.array([np.mean(v) for v in s.values()]) * 100 for k, s in by.items()}

def optimal(res, unit, ns=range(1, 9)):
    m = {n: res[(unit, n)].mean() for n in ns}; s = {n: res[(unit, n)].std() for n in ns}
    b = max(m, key=m.get)
    return min(n for n in ns if m[n] >= m[b] - s[b]), b

def table(body, cols, label):
    return ("\\begin{table}[t]\n\\centering\n\\small\n"
            f"\\begin{{tabular}}{{{cols}}}\n\\hline\n{body}\\hline\n\\end{{tabular}}\n"
            f"\\caption{{TODO}}\n\\label{{{label}}}\n\\end{{table}}\n")


def t_main():
    res = unseen(os.path.join(RES, "main.csv"))
    opt = {u: optimal(res, u)[0] for u in UNITS}
    body = "$n$ & " + " & ".join(f"\\textbf{{{LABEL[u]}}}" for u in UNITS) + " \\\\\n\\hline\n"
    for n in range(1, 9):
        cells = []
        for u in UNITS:
            c = f"{res[(u, n)].mean():.1f} $\\pm$ {res[(u, n)].std():.1f}"
            cells.append(f"\\textbf{{{c}}}" if n == opt[u] else c)
        body += f"{n} & " + " & ".join(cells) + " \\\\\n"
    return table(body, "cccc", "tab:main")


def baseline(log):
    return re.search(r"AUROC = ([\d.]+) ± ([\d.]+)", open(os.path.join(HERE, "logs", log), encoding="utf-8").read()).groups()

def t_summary():
    body = "\\textbf{Method} & \\multicolumn{2}{c}{\\textbf{Original}} & \\multicolumn{2}{c}{\\textbf{Masked}} \\\\\n"
    body += " & Opt.\\ $n$ & AUROC & Opt.\\ $n$ & AUROC \\\\\n\\hline\n"
    raw, mask = unseen(os.path.join(RES, "main.csv")), unseen(os.path.join(RES, "mask.csv"))
    for u in UNITS:
        row = f"{LABEL[u]} $n$-gram"
        for res in (raw, mask):
            o, b = optimal(res, u)
            row += f" & {o} & {res[(u, b)].mean():.1f} $\\pm$ {res[(u, b)].std():.1f}"
        body += row + " \\\\\n"
    (m1, s1), (m2, s2) = baseline("full_katfish.log"), baseline("full_katfish_mask.log")
    body += f"KatFishNet (comma) & -- & {m1} $\\pm$ {s1} & -- & {m2} $\\pm$ {s2} \\\\\n"
    return table(body, "lcccc", "tab:summary")


def t_generators():
    by = defaultdict(list)
    res = unseen(os.path.join(RES, "main.csv"))
    opt = {u: optimal(res, u)[0] for u in UNITS}
    for r in csv.DictReader(open(os.path.join(RES, "main.csv"), encoding="utf-8")):
        if int(r["n"]) == opt[r["unit"]]:
            by[(r["unit"], r["train_gen"], r["test_gen"])].append(float(r["auroc"]) * 100)
    gens = list(GEN)
    body = "\\textbf{Unit} & \\textbf{Train $\\backslash$ Test} & " + " & ".join(GEN[g] for g in gens) + " \\\\\n\\hline\n"
    for u in UNITS:
        for k, gt in enumerate(gens):
            head = f"\\multirow{{3}}{{*}}{{{LABEL[u]} ($n$={opt[u]})}}" if k == 0 else ""
            cells = [f"{np.mean(by[(u, gt, ge)]):.1f}" for ge in gens]
            body += f"{head} & {GEN[gt]} & " + " & ".join(cells) + " \\\\\n"
        if u != UNITS[-1]:
            body += "\\hline\n"
    return "% \\usepackage{multirow} 필요\n" + table(body, "llccc", "tab:generators")


def t_cross():
    dirs = [("ours->ours", "Ours $\\rightarrow$ Ours"), ("ours->katfish", "Ours $\\rightarrow$ KatFish"),
            ("katfish->katfish", "KatFish $\\rightarrow$ KatFish"), ("katfish->ours", "KatFish $\\rightarrow$ Ours")]
    by = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(RES, "cross.csv"), encoding="utf-8")):
        by[(r["method"], int(r["n"]), r["direction"])].append(float(r["auroc"]) * 100)
    body = "\\textbf{Train $\\rightarrow$ Test} & \\textbf{Comma} & " + " & ".join(f"\\textbf{{{LABEL[u]}}}" for u in UNITS) + " \\\\\n\\hline\n"
    for d, name in dirs:
        cells = [f"{np.mean(by[('comma', 0, d)]):.1f}"]
        for u in UNITS:
            res = {(u, n): np.array(by[(u, n, d)]) for n in range(1, 9)}
            o, b = optimal(res, u)
            cells.append(f"{res[(u, b)].mean():.1f} ({o})")
        body += f"{name} & " + " & ".join(cells) + " \\\\\n"
    return "% 칸 = 해당 단위의 최고 AUROC (괄호 = 최적 n, 최고값의 1 std 이내 가장 작은 n)\n" + table(body, "lcccc", "tab:cross")


def t_bootstrap():
    pat = re.compile(r"(\w+) (\d) \(([\d.]+)\) - (\w+) (\d) \(([\d.]+)\) = ([+-][\d.]+)%p\s+95% CI \[([+-][\d.]+), ([+-][\d.]+)\]")
    rows = {}
    for tag, log in (("raw", "analysis.log"), ("mask", "analysis_mask.log")):
        for m in pat.finditer(open(os.path.join(HERE, "logs", log), encoding="utf-8").read()):
            key = (m[1], m[2], m[4], m[5])
            rows.setdefault(key, {})[tag] = f"{m[7]} [{m[8]}, {m[9]}]"
    body = "\\textbf{Comparison} & \\textbf{Original} & \\textbf{Masked} \\\\\n\\hline\n"
    for (u1, n1, u2, n2), v in rows.items():
        body += f"{LABEL[u1]}-{n1} vs {LABEL[u2]}-{n2} & {v['raw']} & {v['mask']} \\\\\n"
    return table(body, "lcc", "tab:bootstrap")


if __name__ == "__main__":
    out = os.path.join(RES, "tables.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("% make_tables.py 로 생성. 캡션(TODO)은 직접 작성\n\n")
        for t in (t_main(), t_summary(), t_generators(), t_cross(), t_bootstrap()):
            f.write(t + "\n")
    print(f"-> {out}")
