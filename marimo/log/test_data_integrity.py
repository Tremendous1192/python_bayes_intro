# - 作成日: 2026-09-27。教材CSVの改行互換と内容改変・不正取得の拒否を検証する。
# - 実行: uv run --locked --group notebook python log/test_data_integrity.py
# - 制約: Notebook専用。試験ファイルはlog/_work内、通信はモックで遮断する。
import io
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
# - env.cmdの埋込コードを読み、通常の入口を実行せずデータ準備だけを検査する。
source_path = ROOT / "env.cmd"
head, separator, body = source_path.read_text(encoding="utf-8").partition("\n:__BAYES_PYTHON__\n")
assert separator and body, "Embedded Python is missing"
prepare_data = ModuleType("bayes_env")
prepare_data.__file__ = str(source_path)
sys.modules[prepare_data.__name__] = prepare_data
exec(compile("\n" * (head.count("\n") + 2) + body, str(source_path), "exec"), prepare_data.__dict__)
sys.path.insert(0, str(ROOT / "notebooks"))
import mod_load_data


def test_line_endings_and_tampering() -> None:
    """実際の3教材で改行互換と値・単独CRの改変拒否を確認する。"""
    sources = (ROOT / "data/sources.json").read_bytes()
    with tempfile.TemporaryDirectory(dir=ROOT / "log/_work") as temporary:
        folder = Path(temporary)
        (folder / "sources.json").write_bytes(sources)
        with patch.object(prepare_data, "DATA_ROOT", folder), patch.object(mod_load_data, "DATA_ROOT", folder):
            for newline in (b"\n", b"\r\n"):
                for name in json.loads(sources):
                    data = (ROOT / "data" / name).read_bytes().replace(b"\r\n", b"\n")
                    (folder / name).write_bytes(data.replace(b"\n", newline))
                prepare_data.prepare()
                for name in json.loads(sources):
                    assert not mod_load_data.load_data(name).empty
            target = folder / "iris.csv"
            valid = target.read_bytes()
            # - 数値変更・末尾の追加・単独CRは改行の互換処理で隠さない。
            for corrupted in (valid.replace(b"5.1", b"9.1", 1), valid + b"x", valid.replace(b"\r\n", b"\r", 1)):
                assert corrupted != valid
                target.write_bytes(corrupted)
                for operation in (prepare_data.prepare, lambda: mod_load_data.load_data("iris.csv")):
                    try:
                        operation()
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("Modified data was accepted")
                assert target.read_bytes() == corrupted


def test_download_bytes() -> None:
    """取得時は元バイト列を要求し、不正応答を保存しないことを確認する。"""
    with tempfile.TemporaryDirectory(dir=ROOT / "log/_work") as temporary:
        folder = Path(temporary)
        (folder / "sources.json").write_bytes((ROOT / "data/sources.json").read_bytes())
        for name in ("test_scores.csv", "irt-sample.csv"):
            (folder / name).write_bytes((ROOT / "data" / name).read_bytes())
        valid = (ROOT / "data/iris.csv").read_bytes().replace(b"\r\n", b"\n")
        with patch.object(prepare_data, "DATA_ROOT", folder):
            for response in (valid.replace(b"\n", b"\r\n"), valid + b"x", b"x" * 2_000_001, valid):
                with patch.object(prepare_data.urllib.request, "urlopen", return_value=io.BytesIO(response)):
                    if response == valid:
                        prepare_data.prepare(download=True)
                        assert (folder / "iris.csv").read_bytes() == valid
                    else:
                        try:
                            prepare_data.prepare(download=True)
                        except ValueError:
                            pass
                        else:
                            raise AssertionError("Invalid download was accepted")
                        assert not (folder / "iris.csv").exists()


def test_no_implicit_download_or_overwrite() -> None:
    """不足時は通信せず、新しい手順を案内し、取得許可でも既存改変を保全する。"""
    with tempfile.TemporaryDirectory(dir=ROOT / "log/_work") as temporary:
        folder = Path(temporary)
        (folder / "sources.json").write_bytes((ROOT / "data/sources.json").read_bytes())
        with patch.object(prepare_data, "DATA_ROOT", folder), patch.object(mod_load_data, "DATA_ROOT", folder), \
                patch.object(prepare_data.urllib.request, "urlopen") as network:
            for operation in (prepare_data.prepare, lambda: mod_load_data.load_data("iris.csv")):
                try:
                    operation()
                except FileNotFoundError as error:
                    assert "call env.cmd prepare-data --download" in str(error)
                else:
                    raise AssertionError("Missing data was accepted")
            target = folder / "iris.csv"
            target.write_bytes(b"USER CONTENT")
            try:
                prepare_data.prepare(download=True)
            except ValueError:
                pass
            else:
                raise AssertionError("Modified data was overwritten")
            assert target.read_bytes() == b"USER CONTENT"
            network.assert_not_called()


if __name__ == "__main__":
    (ROOT / "log/_work").mkdir(exist_ok=True)
    test_line_endings_and_tampering()
    test_download_bytes()
    test_no_implicit_download_or_overwrite()
    print("Passed: LF/CRLF, rejected modifications, strict/limited downloads, missing data guidance, no network")
