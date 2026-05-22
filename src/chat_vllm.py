"""
vllm 서버 기반 모델 비교 클라이언트

두 vllm 서버(포트 8000/8001)에 동시에 같은 질문을 보내고
응답을 나란히 출력한다.

서버가 먼저 실행 중이어야 한다:
  CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model quantized_models/SOLAR_10.7B_4bit_cond_A --quantization gptq --port 8000
  CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model quantized_models/SOLAR_10.7B_4bit_cond_C_v3 --quantization gptq --port 8001
"""

import sys
import threading
import requests

URL_A   = "http://localhost:8000/v1/chat/completions"
URL_B   = "http://localhost:8001/v1/chat/completions"
NAME_A  = "cond_A (랜덤 영어 calibration)"
NAME_B  = "cond_C_v3 (형태소 다양성 calibration)"
SYSTEM  = "당신은 도움이 되는 한국어 AI 어시스턴트입니다. 항상 한국어로 답변하세요."

def query(url: str, messages: list, result: dict, key: str,
          temperature: float = 0.7, max_tokens: int = 512):
    try:
        resp = requests.post(url, json={
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }, timeout=120)
        resp.raise_for_status()
        result[key] = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        # 400일 때 서버 응답 본문도 출력
        detail = ""
        if hasattr(e, "response") and e.response is not None:
            try: detail = e.response.json()
            except: detail = e.response.text[:200]
        result[key] = None
        result[f"{key}_err"] = f"[에러] {e}\n{detail}"

def ask_both(history_a: list, history_b: list, user_input: str,
             temperature: float, max_tokens: int):
    history_a.append({"role": "user", "content": user_input})
    history_b.append({"role": "user", "content": user_input})

    result = {}
    t_a = threading.Thread(target=query, args=(URL_A, history_a, result, "a", temperature, max_tokens))
    t_b = threading.Thread(target=query, args=(URL_B, history_b, result, "b", temperature, max_tokens))
    t_a.start(); t_b.start()
    t_a.join();  t_b.join()

    reply_a = result.get("a")
    reply_b = result.get("b")

    # 에러 시: 히스토리에서 이번 턴 제거 (오염 방지)
    if reply_a is None:
        history_a.pop()  # user 메시지 제거
        reply_a = result.get("a_err", "[없음]")
    else:
        history_a.append({"role": "assistant", "content": reply_a})

    if reply_b is None:
        history_b.pop()
        reply_b = result.get("b_err", "[없음]")
    else:
        history_b.append({"role": "assistant", "content": reply_b})

    return reply_a, reply_b

def check_servers():
    ok = True
    for name, url_base in [(NAME_A, "http://localhost:8000"), (NAME_B, "http://localhost:8001")]:
        try:
            r = requests.get(f"{url_base}/v1/models", timeout=3)
            model_id = r.json()["data"][0]["id"]
            print(f"  ✅ {name}  →  {model_id}")
        except Exception as e:
            print(f"  ❌ {name}  →  연결 실패: {e}")
            ok = False
    return ok

def main():
    temperature = 0.7
    max_tokens  = 512

    print("=" * 64)
    print("vllm 비교 클라이언트")
    print("=" * 64)
    print("서버 확인 중...")
    if not check_servers():
        print("\n서버를 먼저 실행하세요. README 참고.")
        sys.exit(1)

    print(f"\n온도: {temperature}  |  최대 토큰: {max_tokens}")
    print("명령: /quit  /clear  /temp <값>  /tokens <값>")
    print("=" * 64)

    history_a = [{"role": "system", "content": SYSTEM}]
    history_b = [{"role": "system", "content": SYSTEM}]

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("종료합니다.")
            break

        if user_input == "/clear":
            history_a = [{"role": "system", "content": SYSTEM}]
            history_b = [{"role": "system", "content": SYSTEM}]
            print("[대화 기록 초기화]")
            continue

        if user_input.startswith("/temp "):
            try:
                temperature = float(user_input.split()[1])
                print(f"[온도 변경] {temperature}")
            except:
                print("[사용법] /temp 0.7")
            continue

        if user_input.startswith("/tokens "):
            try:
                max_tokens = int(user_input.split()[1])
                print(f"[최대 토큰 변경] {max_tokens}")
            except:
                print("[사용법] /tokens 512")
            continue

        print("\n[응답 생성 중...]\n")
        reply_a, reply_b = ask_both(history_a, history_b, user_input, temperature, max_tokens)

        w = 62
        print("─" * w)
        print(f"▶ A: {NAME_A}")
        print("─" * w)
        print(reply_a)
        print("─" * w)
        print(f"▶ B: {NAME_B}")
        print("─" * w)
        print(reply_b)
        print("─" * w)

if __name__ == "__main__":
    main()
