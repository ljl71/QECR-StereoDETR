"""Validate 7518 KITTI result files and create a root-level submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


EXPECTED_COUNT = 7518
VALID_CLASSES = {"Car", "Pedestrian", "Cyclist"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--output_zip", required=True)
    return parser.parse_args()


def validate_results(result_dir):
    files = sorted(result_dir.glob("*.txt"))
    expected_names = [f"{index:06d}.txt" for index in range(EXPECTED_COUNT)]
    observed_names = [path.name for path in files]
    if observed_names != expected_names:
        missing = sorted(set(expected_names) - set(observed_names))
        extra = sorted(set(observed_names) - set(expected_names))
        raise RuntimeError(
            "KITTI结果文件不完整或编号异常；missing={} extra={}".format(
                missing[:10], extra[:10]
            )
        )

    detection_count = 0
    for path in files:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                fields = line.split()
                if len(fields) != 16:
                    raise RuntimeError(
                        f"{path}:{line_number} 不是KITTI 16字段检测格式"
                    )
                if fields[0] not in VALID_CLASSES:
                    raise RuntimeError(
                        f"{path}:{line_number} 非法类别 {fields[0]}"
                    )
                values = [float(value) for value in fields[1:]]
                if not all(math.isfinite(value) for value in values):
                    raise RuntimeError(
                        f"{path}:{line_number} 包含NaN或Inf"
                    )
                detection_count += 1
    return files, detection_count


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir).expanduser().resolve()
    output_zip = Path(args.output_zip).expanduser().resolve()
    if not result_dir.is_dir():
        raise FileNotFoundError(result_dir)
    if output_zip.exists():
        raise FileExistsError(
            f"拒绝覆盖已有提交包：{output_zip}"
        )

    files, detection_count = validate_results(result_dir)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)

    with ZipFile(output_zip, "r") as archive:
        names = archive.namelist()
        expected_names = [
            f"{index:06d}.txt" for index in range(EXPECTED_COUNT)
        ]
        if names != expected_names:
            raise RuntimeError("ZIP内部编号或顺序异常")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP完整性测试失败：{bad_member}")

    checksum = sha256(output_zip)
    checksum_file = Path(str(output_zip) + ".sha256")
    checksum_file.write_text(
        f"{checksum}  {output_zip.name}\n",
        encoding="utf-8",
    )
    print(
        "KITTI提交包检查通过：{}个文件，{}条检测".format(
            len(files), detection_count
        )
    )
    print(f"ZIP={output_zip}")
    print(f"SHA256={checksum}")


if __name__ == "__main__":
    main()
