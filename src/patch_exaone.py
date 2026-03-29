"""
EXAONE-3.5-7.8B-Instruct 캐시 파일 호환성 패치 (transformers 4.57.6 기준)
패치 대상: snapshot + 두 modules 캐시 디렉토리
"""
import os
import re
import ast

BASE = "/home/choihyun/workspace/.cache/huggingface"
SNAP_HASH = "553ea250b9a5317231459279d5847d6cf955b9aa"

SNAPSHOT_DIR = f"{BASE}/hub/models--LGAI-EXAONE--EXAONE-3.5-7.8B-Instruct/snapshots/{SNAP_HASH}"

MODULES_DIRS = [
    f"{BASE}/modules/transformers_modules/_553ea250b9a5317231459279d5847d6cf955b9aa",
    f"{BASE}/modules/transformers_modules/LGAI_hyphen_EXAONE/"
    f"EXAONE_hyphen_3_dot_5_hyphen_7_dot_8B_hyphen_Instruct/{SNAP_HASH}",
    f"{BASE}/modules/transformers_modules/LGAI-EXAONE/"
    f"EXAONE-3.5-7.8B-Instruct/{SNAP_HASH}",
]

STUBS = """\
# --- compatibility stubs (transformers 4.57.6) ---
import contextlib as _ctx
use_kernel_func_from_hub = lambda *a, **k: None
use_kernel_forward_from_hub = lambda *a, **k: (lambda cls: cls)
use_kernelized_func = lambda *a, **k: (lambda fn: fn)
@_ctx.contextmanager
def maybe_autocast(*a, **k):
    yield
def check_model_inputs(fn=None, *a, **k):
    if fn is not None and callable(fn):
        return fn
    return lambda f: f
# -------------------------------------------------
"""


def patch_configuration(path: str):
    with open(path) as f:
        src = f.read()
    changed = False
    if "from transformers.modeling_rope_utils import RopeParameters" in src:
        src = src.replace(
            "from transformers.modeling_rope_utils import RopeParameters\n", ""
        )
        changed = True
    if "rope_parameters: RopeParameters | None = None," in src:
        src = src.replace(
            "rope_parameters: RopeParameters | None = None,", "rope_parameters=None,"
        )
        changed = True
    if changed:
        with open(path, "w") as f:
            f.write(src)
        print(f"[패치] {os.path.basename(os.path.dirname(path))}/configuration_exaone.py")
    else:
        print(f"[스킵] configuration_exaone.py — 변경 불필요")


def patch_modeling(path: str):
    with open(path) as f:
        src = f.read()

    # 1. stubs 삽입/업데이트
    if "compatibility stubs" not in src:
        torch_import = src.find("\nimport torch")
        if torch_import == -1:
            torch_import = src.find("\nfrom torch")
        if torch_import != -1:
            insert_pos = src.find("\n", torch_import + 1)
            src = src[:insert_pos + 1] + "\n" + STUBS + "\n" + src[insert_pos + 1:]
        else:
            src = STUBS + "\n" + src
    else:
        # 기존 stubs 내 lambda 형태 업데이트
        src = src.replace(
            "use_kernel_forward_from_hub = lambda *a, **k: None",
            "use_kernel_forward_from_hub = lambda *a, **k: (lambda cls: cls)",
        )
        src = src.replace(
            "use_kernelized_func = lambda *a, **k: None",
            "use_kernelized_func = lambda *a, **k: (lambda fn: fn)",
        )

    # 2. @use_kernel*(...) 데코레이터 라인 제거 (@ 포함)
    src = re.sub(r"@use_kernel_forward_from_hub\([^)]*\)\n", "", src)
    src = re.sub(r"@use_kernelized_func\([^)]*\)\n", "", src)
    src = re.sub(r"@use_kernel_func_from_hub\([^)]*\)\n", "", src)

    # 3. 잔여 @def / @class 수정
    src = re.sub(r"^@(def |class )", r"\1", src, flags=re.MULTILINE)

    # 4. check_model_inputs decorator 형태 업데이트
    old_check = "def check_model_inputs(*a, **k):\n    pass"
    new_check = (
        "def check_model_inputs(fn=None, *a, **k):\n"
        "    if fn is not None and callable(fn):\n"
        "        return fn\n"
        "    return lambda f: f"
    )
    src = src.replace(old_check, new_check)

    # 5. ALL_ATTENTION_FUNCTIONS.get_interface() → .get() (transformers 4.57.6 호환)
    lines = src.split("\n")
    new_lines = []
    i = 0
    while i < len(lines):
        if "ALL_ATTENTION_FUNCTIONS.get_interface(" in lines[i]:
            new_lines.append(
                "        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get("
                "self.config._attn_implementation, eager_attention_forward)"
            )
            # skip next 2 lines (argument line and closing paren)
            i += 3
        else:
            new_lines.append(lines[i])
            i += 1
    src = "\n".join(new_lines)

    # 6. rope_parameters None 안전 접근
    src = src.replace(
        'self.config.rope_parameters["rope_type"]',
        '(self.config.rope_parameters or self.config.rope_scaling or {}).get("rope_type", "default")',
    )
    src = src.replace(
        'config.rope_parameters["rope_theta"]',
        '(config.rope_parameters or config.rope_scaling or {}).get("rope_theta", getattr(config, "rope_theta", 10000.0))',
    )

    with open(path, "w") as f:
        f.write(src)

    # 구문 검사
    try:
        ast.parse(src)
        print(f"[패치+OK] {path[-70:]}")
    except SyntaxError as e:
        print(f"[경고] 구문 오류: {e} — {path[-50:]}")


def patch_auto_docstring():
    path = (
        "/opt/conda/envs/llm-quant/lib/python3.11/site-packages/"
        "transformers/utils/auto_docstring.py"
    )
    if not os.path.exists(path):
        return
    with open(path) as f:
        src = f.read()
    if "isinstance(param_type, type(int | str))" in src:
        print("[스킵] auto_docstring.py — 이미 패치됨")
        return
    target = '        elif hasattr(param_type, "__module__"):'
    patch_line = (
        "        elif isinstance(param_type, type(int | str)):\n"
        "            param_type = str(param_type)\n"
    )
    if target in src:
        src = src.replace(target, patch_line + target)
        with open(path, "w") as f:
            f.write(src)
        print("[패치] auto_docstring.py 수정 완료")


if __name__ == "__main__":
    # 1. snapshot
    cfg = os.path.join(SNAPSHOT_DIR, "configuration_exaone.py")
    mdl = os.path.join(SNAPSHOT_DIR, "modeling_exaone.py")
    if os.path.exists(cfg):
        patch_configuration(cfg)
    if os.path.exists(mdl):
        patch_modeling(mdl)

    # 2. modules cache dirs
    for d in MODULES_DIRS:
        mdl2 = os.path.join(d, "modeling_exaone.py")
        cfg2 = os.path.join(d, "configuration_exaone.py")
        if os.path.exists(mdl2):
            patch_modeling(mdl2)
        if os.path.exists(cfg2):
            patch_configuration(cfg2)

    # 3. auto_docstring
    patch_auto_docstring()

    # 4. optimum from_dict: desc_act=True 시 act_group_aware=False 강제
    optimum_quant = "/opt/conda/envs/llm-quant/lib/python3.11/site-packages/optimum/gptq/quantizer.py"
    if os.path.exists(optimum_quant):
        with open(optimum_quant) as f:
            osrc = f.read()
        target = "        if config_dict.get('desc_act', False) and config_dict.get('act_group_aware', False):"
        fixed = "        if config_dict.get('desc_act', False):"
        if target in osrc:
            osrc = osrc.replace(target, fixed)
            with open(optimum_quant, "w") as f:
                f.write(osrc)
            print("[패치] optimum from_dict act_group_aware 수정 완료")
        elif fixed not in osrc:
            print("[경고] optimum quantizer.py 패치 대상 찾지 못함")

    print("\n모든 패치 완료!")
