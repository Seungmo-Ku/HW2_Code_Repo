"""
샘플링된 문서마다 (제목 + 요약)을 LLM에 주고, 사람 글과 비슷한 길이의 논문 단락을 생성한다.
세 제공자(openai / gemini / anthropic)를 같은 프롬프트로 호출하고,
샘플링 파라미터는 제공자 기본값을 그대로 쓴다 (Claude API는 temperature 지정을 받지 않으므로 통일).

API 키는 환경변수 또는 이 폴더의 .env 에서 읽는다:
  OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY

사용법:
  python generate.py --list_models openai               # 사용 가능한 모델 이름 확인
  python generate.py --model anthropic:claude-haiku-5-5 --limit 20
  python generate.py --model openai:<모델명> --limit 20
  python generate.py --model gemini:<모델명> --limit 20
이미 생성된 id는 건너뛰므로 중단 후 다시 실행하면 이어서 생성한다.
"""
import argparse, json, os, sys, time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM = "당신은 한국어 공학 논문을 쓰는 연구자입니다."
PROMPT = """다음은 한 공학 논문의 제목과, 그 논문 본문 중 한 단락의 요약입니다.
요약의 내용을 담은 논문 본문 단락을 한국어로 약 {length}자 분량으로 작성하세요.
제목, 소제목, 목록, 설명 없이 단락 본문만 출력하세요.

제목: {title}
요약: {summary}"""


def load_env():
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# 제공자별 호출: (생성 텍스트, 입력 토큰, 출력 토큰, 응답에 기록된 모델 이름)을 반환

def call_openai(model, prompt):
    from openai import OpenAI
    client = call_openai.client = getattr(call_openai, "client", None) or OpenAI()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens, r.model


def call_gemini(model, prompt):
    from google import genai
    from google.genai import types
    client = call_gemini.client = getattr(call_gemini, "client", None) or genai.Client()
    r = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM),
    )
    u = r.usage_metadata
    return r.text, u.prompt_token_count, u.candidates_token_count, r.model_version or model


def call_anthropic(model, prompt):
    import anthropic
    client = call_anthropic.client = getattr(call_anthropic, "client", None) or anthropic.Anthropic()
    r = client.messages.create(
        model=model, max_tokens=4096, system=SYSTEM,
        output_config={"effort": "low"},  # 단순 작문이라 사고량 최소화
        messages=[{"role": "user", "content": prompt}],
    )
    if r.stop_reason == "refusal":
        raise RuntimeError(f"refusal: {r.stop_details}")
    text = "".join(b.text for b in r.content if b.type == "text")
    return text, r.usage.input_tokens, r.usage.output_tokens, r.model


CALLERS = {"openai": call_openai, "gemini": call_gemini, "anthropic": call_anthropic}


def list_models(provider):
    if provider == "openai":
        from openai import OpenAI
        names = [m.id for m in OpenAI().models.list()]
    elif provider == "gemini":
        from google import genai
        client = genai.Client()  # 변수에 잡아두지 않으면 순회 중 닫힘
        names = [m.name for m in client.models.list()]
    else:
        import anthropic
        names = [m.id for m in anthropic.Anthropic().models.list()]
    print("\n".join(sorted(names)))


def main(args):
    load_env()
    if args.list_models:
        return list_models(args.list_models)

    provider, model = args.model.split(":", 1)
    call = CALLERS[provider]
    out = args.out or os.path.join(HERE, "data", f"gen_{provider}_{model.replace('/', '_')}.jsonl")

    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f}
    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    todo = [r for r in rows[:args.limit] if r["id"] not in done]
    print(f"{provider}:{model} | 대상 {min(args.limit, len(rows))}개, 이미 생성 {len(done)}개, 이번에 {len(todo)}개")

    tok_in = tok_out = n_ok = 0
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            prompt = PROMPT.format(length=len(r["human_text"]), title=r["title"], summary=r["summary"])
            for attempt in range(3):
                try:
                    text, n_in, n_out, served = call(model, prompt)
                    break
                except Exception as e:  # 네트워크/일시 오류는 잠깐 쉬고 재시도
                    print(f"  [{r['id']}] 오류 ({attempt + 1}/3): {e}", file=sys.stderr)
                    time.sleep(5 * (attempt + 1))
            else:
                continue
            tok_in += n_in; tok_out += n_out; n_ok += 1
            f.write(json.dumps({
                "id": r["id"], "provider": provider, "model": served, "text": (text or "").strip(),
                "target_len": len(r["human_text"]), "input_tokens": n_in, "output_tokens": n_out,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  누적 토큰 in {tok_in} / out {tok_out}")

    if n_ok:
        n = n_ok
        print(f"성공 {n_ok}/{len(todo)} | 문서당 평균 토큰: in {tok_in / n:.0f} / out {tok_out / n:.0f}  -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="provider:model_id  (provider = openai | gemini | anthropic)")
    p.add_argument("--input", default=os.path.join(HERE, "data", "sampled.jsonl"))
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=20, help="앞에서부터 몇 개를 생성할지")
    p.add_argument("--list_models", choices=list(CALLERS))
    main(p.parse_args())
