"""
본 실험: 분할 단위(어절/형태소/음절) x n(1~8) 에 따른 LLM 생성 한국어 탐지 성능 비교 (HW1 pilot.py 확장)

데이터: sample_data.py 의 사람 글 + generate.py 의 생성기별 LLM 글 (같은 문서 id 끼리 짝)
전처리: 짝마다 사람 글과 LLM 글을 모두 짧은 쪽 길이의 90% 에서 어절 경계로 자름 (길이 단서와 문장 끝 단서 제거)
분할: seed 마다 문서 id 를 80/20 으로 나눔. 같은 문서의 사람 글/LLM 글이 학습과 테스트에 나뉘어 들어가지 않음
평가: 생성기 하나로 학습 -> 세 생성기 각각으로 테스트 (AUROC). 학습에 안 쓴 두 생성기의 평균이 주 지표(unseen)
최적 n: 평균 AUROC 가 최고값의 1 표준편차(seed 간) 이내인 가장 작은 n (HW1 정의)

사용법: python experiment.py                      # 전체 (3 단위 x n 1~8 x seed 3 x 학습 생성기 3)
        python experiment.py --units eojeol --ns 1 2  # 일부만
"""
import argparse, csv, glob, json, os, random, re
import numpy as np
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
kiwi = Kiwi()

def syllables(text):
    # 음절 단위. 공백과 문장부호도 하나의 토큰으로 남겨 띄어쓰기/쉼표 신호를 보존
    return [ch if not ch.isspace() else "<SP>" for ch in text]

def morphemes(text):
    # 형태소 단위. '먹/VV' 처럼 품사 태그를 붙여 동음이의 형태소를 구분
    return [f"{t.form}/{t.tag}" for t in kiwi.tokenize(text)]

def eojeols(text):
    # 어절 단위: 공백 기준 분리
    return text.split()

TOKENIZERS = {"eojeol": eojeols, "morpheme": morphemes, "syllable": syllables}


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:  # 생성 중인 파일의 마지막 줄이 덜 써졌을 수 있음
                pass
    return rows

def normalize(text):
    return re.sub(r"\s+", " ", text).strip()

def mask_format(text):
    # 문체가 아닌 형식 단서 제거: LLM 은 요약만 받아 원문의 인용/수식/수치를 쓸 수 없으므로
    # 지운 자리에 공백을 넣으면 '풍공학 은' 처럼 새 띄어쓰기 단서가 생기므로 붙여서 지움
    text = re.sub(r"\([^()]*\)|\[[^\[\]]*\]", "", text)    # 괄호 내용 (인용, 영문 병기, 단위 등)
    text = re.sub(r"[A-Za-z][A-Za-z.&'\-]*", "E", text)    # 영문 단어는 모두 E 로 (조사 결합은 유지)
    text = re.sub(r"\d+(?:[.,]\d+)*", "0", text)           # 숫자는 모두 0 으로
    text = re.sub(r"E(?:[ ,]+E)+", "E", text)              # 'et al.' 처럼 연속된 영문은 하나로
    # 원문 추출 잡음: '연구결과 는' 처럼 떨어진 조사를 앞 어절에 붙임 ('이'는 관형사와 겹쳐 제외)
    text = re.sub(r" (은|는|을|를|의|에서|에|으로|로|와|과)(?=[\s.,]|$)", r"\1", text)
    return normalize(text)

def cut(text, length):
    # length 자 이하의 마지막 어절 경계에서 자름
    if len(text) <= length:
        return text
    head = text[:length + 1]  # 바로 뒤가 공백이면 length 자 전체를 쓸 수 있게 한 글자 더 봄
    space = head.rfind(" ")
    return head[:space] if space > 0 else head[:length]

def match_length(a, b, tol=0.1):
    # 짝의 두 글을 모두 짧은 쪽 길이의 (1 - tol) 에서 어절 경계로 자름
    # 한쪽만 자르면 '문장 중간에서 끝남' 이 그쪽 라벨의 단서가 되므로 둘 다 자름
    length = int(min(len(a), len(b)) * (1 - tol))
    return cut(a, length), cut(b, length)

def unwrap(text):
    # 생성 결과가 "['...']" 처럼 리스트 문자열로 감싸져 나온 경우 벗겨냄 (Haiku 1건)
    m = re.fullmatch(r"\s*\[\s*(['\"])(.*)\1\s*\]\s*", text, flags=re.S)
    return m.group(2) if m else text

def load(args):
    prep = mask_format if args.mask else normalize
    human = {r["id"]: prep(r["human_text"]) for r in read_jsonl(args.human)}
    gens = {}
    for path in sorted(glob.glob(os.path.join(args.gen_dir, "gen_*.jsonl"))):
        rows = read_jsonl(path)
        if rows:
            gens[rows[0]["model"]] = {r["id"]: prep(unwrap(r["text"])) for r in rows if r["text"].strip()}
    ids = sorted(set(human).intersection(*[set(g) for g in gens.values()]))
    # 생성기별 (사람 글, LLM 글) 짝을 같은 길이로 자름
    pairs = {}
    for g, texts in gens.items():
        pairs[g] = {}
        for i in ids:
            pairs[g][i] = match_length(human[i], texts[i])
    return ids, pairs

def ngrams(toks, n):
    return [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def main(args):
    ids, pairs = load(args)
    gens = list(pairs)
    print(f"문서 {len(ids)}개 x 생성기 {len(gens)}개: {', '.join(gens)}")
    for g in gens:
        h = np.mean([len(p[0]) for p in pairs[g].values()]); m = np.mean([len(p[1]) for p in pairs[g].values()])
        print(f"  {g}: 길이 맞춤 후 평균 사람 {h:.0f}자 / LLM {m:.0f}자")

    # 토큰화는 느리므로 (생성기, 문서, 사람/LLM) 별로 한 번만 해둠
    texts, index = [], {}
    for g in gens:
        for i in ids:
            for k, t in enumerate(pairs[g][i]):
                index[(g, i, k)] = len(texts); texts.append(t)
    cache = {u: [TOKENIZERS[u](t) for t in texts] for u in args.units}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = open(args.out, "w", newline="", encoding="utf-8")
    w = csv.writer(out)
    w.writerow(["unit", "n", "seed", "train_gen", "test_gen", "auroc", "n_features"])

    summary = {}
    for unit in args.units:
        for n in args.ns:
            unseen_by_seed = []
            for seed in args.seeds:
                rng = random.Random(seed)
                shuffled = ids[:]; rng.shuffle(shuffled)
                cut_at = int(len(shuffled) * 0.8)
                tr_ids, te_ids = shuffled[:cut_at], shuffled[cut_at:]
                if args.train_size:  # 학습 문서 수만 줄이고 테스트는 그대로 (학습량 곡선용)
                    tr_ids = tr_ids[:args.train_size]
                unseen = []
                for g_tr in gens:
                    tr = [index[(g_tr, i, k)] for i in tr_ids for k in (0, 1)]
                    y_tr = [k for i in tr_ids for k in (0, 1)]  # 0 = 사람, 1 = LLM
                    vec = TfidfVectorizer(analyzer=lambda j: ngrams(cache[unit][j], n),
                                          max_features=args.max_features, min_df=2)
                    clf = LogisticRegression(max_iter=2000, random_state=seed)
                    try:
                        clf.fit(vec.fit_transform(tr), y_tr)  # 학습 데이터에만 fit
                    except ValueError:  # 학습 문서가 적어 min_df 를 넘는 n-gram 이 없음 -> 우연 수준으로 기록
                        clf = None
                    for g_te in gens:
                        te = [index[(g_te, i, k)] for i in te_ids for k in (0, 1)]
                        y_te = [k for i in te_ids for k in (0, 1)]
                        auc = roc_auc_score(y_te, clf.predict_proba(vec.transform(te))[:, 1]) if clf else 0.5
                        w.writerow([unit, n, seed, g_tr, g_te, f"{auc:.4f}", len(vec.vocabulary_) if clf else 0])
                        if g_te != g_tr:
                            unseen.append(auc)
                unseen_by_seed.append(np.mean(unseen))
            mean, std = np.mean(unseen_by_seed), np.std(unseen_by_seed)
            summary[(unit, n)] = (mean, std)
            print(f"{unit:9s} n={n}  unseen AUROC = {mean * 100:5.1f} ± {std * 100:4.1f}", flush=True)
            out.flush()
    out.close()

    print("\n최적 n (최고값의 1 표준편차 이내인 가장 작은 n):")
    for unit in args.units:
        rows = [(n, *summary[(unit, n)]) for n in args.ns]
        best_n, best, best_std = max(rows, key=lambda r: r[1])
        opt = min(n for n, m, _ in rows if m >= best - best_std)
        print(f"  {unit:9s} 최고 n={best_n} ({best * 100:.1f})  ->  최적 n = {opt}")
    print(f"seed/생성기 쌍별 결과 -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--human", default=os.path.join(HERE, "data", "sampled.jsonl"))
    p.add_argument("--gen_dir", default=os.path.join(HERE, "data"))
    p.add_argument("--units", nargs="+", default=["eojeol", "morpheme", "syllable"])
    p.add_argument("--ns", nargs="+", type=int, default=list(range(1, 9)))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--max_features", type=int, default=10000)
    p.add_argument("--mask", action="store_true", help="괄호 내용/영문/숫자 등 형식 단서 제거")
    p.add_argument("--train_size", type=int, default=None, help="학습 문서 수 (기본: 전체의 80%%)")
    p.add_argument("--out", default=os.path.join(HERE, "results", "main.csv"))
    main(p.parse_args())
