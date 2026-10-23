"""checkpoint 路径解析：让脚本在两种布局下都能跑。

布局 A（开发仓库根目录）：  ./ckpt_v7.pt          与 ./shared_code/ckpt_v7.pt
布局 B（提交包内实验目录）：  ../../shared_code/ckpt_v7.pt

用法：
    from ckptutil import resolve_ckpt
    ck = torch.load(resolve_ckpt("ckpt_v7.pt"), weights_only=False)
"""
from pathlib import Path

_CANDIDATE_DIRS = (
    ".",                      # 当前工作目录
    "..",                     # 上一级
    "../shared_code",         # 提交包：paperX/experiments → paperX/shared_code
    "../../shared_code",      # 提交包：paperX/experiments → arxiv_submission/shared_code
    "shared_code",            # 仓库根
)


def resolve_ckpt(name, extra_dirs=()):
    """返回第一个存在的候选路径；都不存在则抛出带提示的 FileNotFoundError。"""
    p = Path(name)
    if p.is_absolute() and p.exists():
        return str(p)

    here = Path(__file__).resolve().parent
    dirs = list(extra_dirs) + list(_CANDIDATE_DIRS)
    for d in dirs:
        for base in (Path(d), here / d):
            cand = base / name
            if cand.exists():
                return str(cand)

    tried = [str(Path(d) / name) for d in dirs]
    raise FileNotFoundError(
        f"checkpoint {name!r} not found. Tried:\n  " + "\n  ".join(tried) +
        "\nPut the checkpoints in shared_code/ (see the package README).")


def out_path(name):
    """把新 checkpoint 写到 shared_code/（若存在）否则当前目录。"""
    here = Path(__file__).resolve().parent
    for d in ("shared_code", "../shared_code", "../../shared_code", "."):
        base = Path(d)
        if base.is_dir():
            return str(base / name)
    return name
