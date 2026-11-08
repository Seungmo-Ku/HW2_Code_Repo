"""
베이스라인: KatFishNet (Park et al., 2025) 의 쉼표 특징 5개 + 로지스틱 회귀를 우리 데이터에 적용
특징 정의는 katfishnet/lingustic_analysis/comma_feature_analysis.py 를 따름:
  1. 쉼표를 포함한 문장 비율          2. 문장별 (쉼표 수 / 형태소 수) 평균
  3. 쉼표의 문장 내 상대 위치 평균     4. 쉼표로 나뉜 분절 길이(형태소) 평균
  5. 쉼표 앞뒤 품사 쌍의 다양성 평균
원 코드와의 차이: 문장 분리(kss)와 형태소 분석(Kkma) 대신 Kiwi 를 사용
분할/평가 프로토콜은 experiment.py 와 동일 (문서 단위 80/20, 생성기 하나로 학습 -> 나머지 둘로 테스트)

사용법: python katfishnet_baseline.py [--mask]
"""
import argparse, os, random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from experiment import load, kiwi, HERE


def comma_features(text):
    sents = [[(t.form, t.tag) for t in kiwi.tokenize(s.text)] for s in kiwi.split_into_sents(text)]
    sents = [s for s in sents if s]
    if not sents:  # 빈 글
        return [0.0] * 5
    include, usage, rel_pos, seg_len, diversity = 0, [], [], [], []
    for s in sents:
        morphs = [m for m, _ in s]
        tags = [t for _, t in s]
        commas = [i for i, m in enumerate(morphs) if m == ","]
        if not commas:  # 원 코드와 같이 쉼표 없는 문장은 0 으로 채움
            usage.append(0); rel_pos.append(0); seg_len.append(0); diversity.append(0)
            continue
        include += 1
        usage.append(len(commas) / len(morphs))
        rel_pos.append(np.mean([c / len(morphs) for c in commas]))
        bounds = zip([-1] + commas, commas + [len(morphs)])
        seg_len.append(np.mean([end - start - 1 for start, end in bounds]))
        pairs = [(tags[i - 1], tags[i + 1]) for i in commas if 0 < i < len(tags) - 1]
        diversity.append(len(set(pairs)) / len(pairs) if pairs else 0)
    return [include / max(len(sents), 1), np.mean(usage), np.mean(rel_pos), np.mean(seg_len), np.mean(diversity)]


def main(args):
    ids, pairs = load(args)
    gens = list(pairs)
    feats = {(g, i, k): comma_features(pairs[g][i][k]) for g in gens for i in ids for k in (0, 1)}
    print(f"문서 {len(ids)}개 x 생성기 {len(gens)}개{' (mask)' if args.mask else ''}")

    unseen_by_seed = []
    for seed in args.seeds:
        rng = random.Random(seed)
        shuffled = ids[:]; rng.shuffle(shuffled)
        cut_at = int(len(shuffled) * 0.8)
        tr_ids, te_ids = shuffled[:cut_at], shuffled[cut_at:]
        unseen = []
        for g_tr in gens:
            X = [feats[(g_tr, i, k)] for i in tr_ids for k in (0, 1)]
            y = [k for i in tr_ids for k in (0, 1)]
            scaler = StandardScaler().fit(X)
            clf = LogisticRegression(random_state=seed).fit(scaler.transform(X), y)
            for g_te in gens:
                if g_te == g_tr:
                    continue
                Xt = [feats[(g_te, i, k)] for i in te_ids for k in (0, 1)]
                yt = [k for i in te_ids for k in (0, 1)]
                unseen.append(roc_auc_score(yt, clf.predict_proba(scaler.transform(Xt))[:, 1]))
        unseen_by_seed.append(np.mean(unseen))
    print(f"KatFishNet (comma) unseen AUROC = {np.mean(unseen_by_seed) * 100:.1f} ± {np.std(unseen_by_seed) * 100:.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--human", default=os.path.join(HERE, "data", "sampled.jsonl"))
    p.add_argument("--gen_dir", default=os.path.join(HERE, "data"))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--mask", action="store_true")
    main(p.parse_args())
