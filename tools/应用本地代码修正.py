"""Apply exact, audited text fixes when the Windows patch helper cannot edit copies."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(relative_path: str, old: str, new: str) -> None:
    path = ROOT / relative_path
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            "{} expected one match, found {}".format(relative_path, count)
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


def main():
    replace_once(
        "lib/models/monodetr/query_epipolar.py",
        """        epipolar_depth = epipolar_depth.clamp(
            min=1.0e-3,
            max=max_local_depth + 20.0,
        )
""",
        """        epipolar_depth = torch.minimum(
            epipolar_depth.clamp_min(1.0e-3),
            max_local_depth + 20.0,
        )
""",
    )
    print("本地代码修正已应用")


if __name__ == "__main__":
    main()

