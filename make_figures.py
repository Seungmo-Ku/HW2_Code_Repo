"""
논문 그림 (results/*.csv -> figures/*.pdf, *.png)
  fig_main.pdf     : n 에 따른 unseen AUROC, 단위별 선 (seed 평균 ± std 음영), 최적 n 표시
  fig_learning.pdf : 학습 문서 수별 n-AUROC 곡선, 단위별 패널 (mask)
  fig_cross.pdf    : 교차 데이터셋 방향별 최고 AUROC (KatFishNet 쉼표 vs 단위별 n-gram)

사용법: python make_figures.py
"""
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES, OUT = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
UNITS = ["eojeol", "morpheme", "syllable"]
LABEL = {"eojeol": "Eojeol", "morpheme": "Morpheme", "syllable": "Syllable"}
# 범주 색 (dataviz 기본 팔레트 1–3번, 검증기 통과) + 마커로 이중 부호화
COLOR = {"eojeol": "#2a78d6", "morpheme": "#eb6834", "syllable": "#1baf7a"}
MARKER = {"eojeol": "o", "morpheme": "s", "syllable": "^"}
GRAY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"
SEQ = ["#b7d3f2", "#7fb0e8", "#4a8ddc", "#2a78d6", "#1b4f8f"]  # 학습량: 밝음 -> 어두움 (한 색상)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "lines.linewidth": 1.5, "lines.markersize": 4, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def unseen(path):
    # (unit, n) -> seed 별 unseen 평균 AUROC 배열 (%)
    by = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["train_gen"] != r["test_gen"]:
            by[(r["unit"], int(r["n"]))][r["seed"]].append(float(r["auroc"]))
    return {k: np.array([np.mean(v) for v in s.values()]) * 100 for k, s in by.items()}

def optimal(res, unit, ns):
    m = [res[(unit, n)].mean() for n in ns]; s = [res[(unit, n)].std() for n in ns]
    b = int(np.argmax(m))
    return min(n for n, x in zip(ns, m) if x >= m[b] - s[b])

def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)


def fig_main():
    res, ns = unseen(os.path.join(RES, "main.csv")), list(range(1, 9))
    fig, ax = plt.subplots(figsize=(3.15, 2.2))
    for u in UNITS:
        m = np.array([res[(u, n)].mean() for n in ns]); s = np.array([res[(u, n)].std() for n in ns])
        ax.fill_between(ns, m - s, m + s, color=COLOR[u], alpha=0.15, linewidth=0)
        ax.plot(ns, m, color=COLOR[u], marker=MARKER[u], label=LABEL[u])
        o = optimal(res, u, ns)
        ax.plot([o], [m[o - 1]], marker=MARKER[u], markersize=8, color=COLOR[u],
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.set_xlabel("$n$"); ax.set_ylabel("Unseen-generator AUROC (%)")
    ax.set_xticks(ns); ax.set_ylim(48, 101)
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig_main")


def fig_learning():
    ns, sizes = list(range(1, 9)), [25, 50, 100, 200]
    runs = [(str(t), unseen(os.path.join(RES, f"mask_tr{t}.csv"))) for t in sizes]
    runs.append(("797 (all)", unseen(os.path.join(RES, "mask.csv"))))
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.0), sharey=True)
    for ax, u in zip(axes, UNITS):
        for (name, res), c in zip(runs, SEQ):
            m = [res[(u, n)].mean() for n in ns]
            ax.plot(ns, m, color=c, marker=MARKER[u], markersize=3, linewidth=1.2, label=name)
            o = optimal(res, u, ns)
            ax.plot([o], [m[o - 1]], marker=MARKER[u], markersize=6, color=c,
                    markeredgecolor="white", markeredgewidth=1, zorder=5)
        ax.set_title(LABEL[u], color=INK); ax.set_xticks(ns); ax.set_xlabel("$n$")
    axes[0].set_ylabel("Unseen-generator AUROC (%)"); axes[0].set_ylim(48, 101)
    axes[-1].legend(title="Training docs", frameon=False, loc="lower left", title_fontsize=7)
    save(fig, "fig_learning")


def fig_cross():
    dirs = ["ours->ours", "ours->katfish", "katfish->katfish", "katfish->ours"]
    names = ["Ours→Ours", "Ours→KatFish", "KatFish→KatFish", "KatFish→Ours"]
    by = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(RES, "cross.csv"), encoding="utf-8")):
        by[(r["method"], int(r["n"]), r["direction"])].append(float(r["auroc"]) * 100)
    def best(method, d):
        return max(np.mean(v) for (mm, n, dd), v in by.items() if mm == method and dd == d)
    methods = [("comma", "KatFishNet (comma)", GRAY)] + [(u, f"{LABEL[u]} $n$-gram", COLOR[u]) for u in UNITS]
    fig, ax = plt.subplots(figsize=(3.15, 2.2))
    x, w = np.arange(len(dirs)), 0.19
    for k, (m, name, c) in enumerate(methods):
        vals = [best(m, d) for d in dirs]
        ax.bar(x + (k - 1.5) * w, vals, width=w - 0.02, color=c, label=name)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Best AUROC (%)"); ax.set_ylim(50, 100)
    ax.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.22))
    save(fig, "fig_cross")


if __name__ == "__main__":
    fig_main(); fig_learning(); fig_cross()
    print(f"-> {OUT}")
