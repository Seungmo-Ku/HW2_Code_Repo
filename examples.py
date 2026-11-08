"""
무작위 예시와 오분류 사례 추출 (논문 부록용)

experiment.py 와 같은 분할/모델로 seed 하나의 테스트 분할에서 단위별 최적 n 탐지기의 점수를 구한다.
테스트 항목 = (문서, 테스트 생성기, 사람/LLM) 한 글. 점수 = 그 생성기를 학습에 쓰지 않은 탐지기 2개의 평균 확률 (unseen).
판정 기준 0.5 (로지스틱 회귀 기본). 예시는 고정 seed 로 무작위 추출하며 직접 고르지 않는다.
  - 무작위 예시: 사람 글 2개, LLM 글 2개 (전체 테스트 항목에서)
  - 오분류 사례: 음절 탐지기(최고 단위)가 틀린 글 중 3개 + 단위별 오분류 수와 겹침
사람 글은 AI Hub 약관상 앞 90자만 저장한다 (전문은 문서 id 로 sample_data.py 결과에서 확인).

사용법: python examples.py [--mask] [--seed 42] [--sample_seed 0]
"""
import argparse, json, os, random, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from experiment import load, ngrams, TOKENIZERS, HERE

OPT = {"eojeol": 1, "morpheme": 2, "syllable": 4}


def main(args):
    ids, pairs = load(args)
    gens = list(pairs)
    rng = random.Random(args.seed)
    shuffled = ids[:]; rng.shuffle(shuffled)
    cut_at = int(len(shuffled) * 0.8)
    tr_ids, te_ids = shuffled[:cut_at], shuffled[cut_at:]

    toks = {u: {(g, i, k): TOKENIZERS[u](pairs[g][i][k]) for g in gens for i in ids for k in (0, 1)} for u in OPT}
    items = [(i, g, k) for g in gens for i in te_ids for k in (0, 1)]
    scores = {u: {it: [] for it in items} for u in OPT}
    for u, n in OPT.items():
        for g_tr in gens:
            keys = [(g_tr, i, k) for i in tr_ids for k in (0, 1)]
            vec = TfidfVectorizer(analyzer=lambda key: ngrams(toks[u][key], n), max_features=10000, min_df=2)
            clf = LogisticRegression(max_iter=2000, random_state=args.seed)
            clf.fit(vec.fit_transform(keys), [k for _, _, k in keys])
            for g_te in gens:
                if g_te == g_tr:
                    continue
                its = [(i, g_te, k) for i in te_ids for k in (0, 1)]
                p = clf.predict_proba(vec.transform([(g_te, i, k) for i, _, k in its]))[:, 1]
                for it, s in zip(its, p):
                    scores[u][it].append(s)
    score = {u: {it: float(np.mean(v)) for it, v in scores[u].items()} for u in OPT}
    wrong = {u: {it for it in items if (score[u][it] >= 0.5) != (it[2] == 1)} for u in OPT}

    short = {g: g.split("-")[0].replace("claude", "Haiku").replace("gemini", "Gemini").replace("gpt", "GPT") for g in gens}
    def row(it):
        i, g, k = it
        text = pairs[g][i][k]
        chars = len(text)
        if k == 0:  # 사람 글은 AI Hub 이용 약관상 재배포 불가 -> 앞 90자만 저장 (문서 id 로 sample_data.py 에서 복원 가능)
            text = text[:90] + "…"
        return {"doc": i, "generator": short[g], "label": "LLM" if k else "human", "chars": chars,
                "score": {u: round(score[u][it], 3) for u in OPT},
                "wrong": [u for u in OPT if it in wrong[u]], "text": text}

    print(f"seed {args.seed}{' (mask)' if args.mask else ''}: 테스트 항목 {len(items)}개 (문서 {len(te_ids)} x 생성기 {len(gens)} x 사람/LLM)")
    print("\n## 단위별 오분류 (기준 0.5)")
    for u in OPT:
        w = wrong[u]
        by = {f"{short[g]}-{'LLM' if k else 'human'}": sum(1 for it in w if it[1] == g and it[2] == k) for g in gens for k in (0, 1)}
        print(f"  {u} n={OPT[u]}: {len(w)}개 ({len(w) / len(items) * 100:.1f}%)  {by}")
        fn = sum(1 for it in w if it[2] == 1)
        ws = [score[u][it] for it in w]
        print(f"    LLM 글을 사람으로 판정 {fn / len(w) * 100:.0f}% / 오류 점수 중앙값 {np.median(ws):.2f}, "
              f"0.4–0.6 사이 {np.mean([abs(s - 0.5) < 0.1 for s in ws]) * 100:.0f}% / "
              f"점수 중앙값: LLM 글 {np.median([score[u][it] for it in items if it[2] == 1]):.2f}, "
              f"사람 글 {np.median([score[u][it] for it in items if it[2] == 0]):.2f}")
    allw = set.union(*wrong.values())
    print(f"  세 단위 모두 틀림: {len(set.intersection(*wrong.values()))}개 / 하나 이상 틀림: {len(allw)}개")
    # 오분류가 형식 단서(영문 포함)와 관련 있는지: 틀린 LLM 글 vs 전체 LLM 글의 영문 포함 비율
    has_en = lambda it: bool(re.search(r"[A-Za-z]", pairs[it[1]][it[0]][it[2]]))
    llm = [it for it in items if it[2] == 1]
    print(f"  영문 포함 비율: 전체 LLM 글 {np.mean([has_en(it) for it in llm]) * 100:.1f}%", end="")
    for u in OPT:
        wl = [it for it in wrong[u] if it[2] == 1]
        print(f" / {u} 가 틀린 LLM 글 {np.mean([has_en(it) for it in wl]) * 100:.1f}% ({len(wl)}개)", end="")
    print()

    srng = random.Random(args.sample_seed)
    picks = {"random_human": srng.sample([it for it in items if it[2] == 0], 2),
             "random_llm": srng.sample([it for it in items if it[2] == 1], 2),
             "syllable_errors": srng.sample(sorted(wrong["syllable"]), min(3, len(wrong["syllable"])))}
    out = {k: [row(it) for it in v] for k, v in picks.items()}
    for k, rows in out.items():
        print(f"\n## {k}")
        for r in rows:
            print(f"- 문서 {r['doc']} | {r['generator']} 짝 | 정답 {r['label']} | {r['chars']}자 | 점수 {r['score']} | 틀린 단위 {r['wrong']}")
            print(f"  {r['text']}")
    path = os.path.join(HERE, "results", f"examples{'_mask' if args.mask else ''}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--human", default=os.path.join(HERE, "data", "sampled.jsonl"))
    p.add_argument("--gen_dir", default=os.path.join(HERE, "data"))
    p.add_argument("--mask", action="store_true")
    p.add_argument("--seed", type=int, default=42, help="분할 seed (experiment.py 와 같음)")
    p.add_argument("--sample_seed", type=int, default=0, help="예시 추출 seed")
    main(p.parse_args())
