# - 作成日: 2026-09-27。教材CSVの改行互換と内容改変・不正取得の拒否を検証する。
# - 更新日: 2026-09-28。独立したデータ準備コードの読込先とCLIを検証する。
# - 実行: uv run --locked --group notebook python log/test_data_integrity.py
# - 制約: Notebook専用。試験ファイルはlog/_work内、通信はモックで遮断する。
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
# - 通常の入口を実行せず、data内の独立したモジュールを読み込む。
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "notebooks"))
import prepare_data
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
                    assert "python data/prepare_data.py --download" in str(error)
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


def test_command_line() -> None:
    """別の作業ディレクトリから起動し、引数と終了コード・既存CSVの保全を確認する。"""
    with tempfile.TemporaryDirectory(dir=ROOT / "log/_work") as temporary:
        folder = Path(temporary)
        script = folder / "prepare_data.py"
        script.write_bytes(Path(prepare_data.__file__).read_bytes())
        sources = (ROOT / "data/sources.json").read_bytes()
        (folder / "sources.json").write_bytes(sources)
        for name in json.loads(sources):
            (folder / name).write_bytes((ROOT / "data" / name).read_bytes())
        before = {path.name: path.read_bytes() for path in folder.iterdir()}
        cases = (([], 0, "Passed:"), (["--help"], 0, "--download"),
                 (["--unknown"], 2, "unrecognized arguments"),
                 (["--down"], 2, "unrecognized arguments"),
                 (["--download", "extra"], 2, "unrecognized arguments"))
        for arguments, expected_code, message in cases:
            # - 標準ライブラリだけで起動し、全CSVを用意して実通信を発生させない。
            result = subprocess.run([sys.executable, "-I", "-B", "-S", str(script), *arguments],
                                    cwd=ROOT.parent, capture_output=True, text=True,
                                    encoding="utf-8", timeout=30)
            assert result.returncode == expected_code, result.stderr
            assert message in result.stdout + result.stderr
            assert {path.name: path.read_bytes() for path in folder.iterdir()} == before
        # - 不足時も自動取得せず、独立したCLIの終了コード1と復旧案内を返す。
        (folder / "iris.csv").unlink()
        result = subprocess.run([sys.executable, "-I", "-B", "-S", str(script)],
                                cwd=ROOT.parent, capture_output=True, text=True,
                                encoding="utf-8", timeout=30)
        assert result.returncode == 1 and "--download" in result.stderr
        assert not (folder / "iris.csv").exists()
    # - 明示フラグが取得処理へ渡ることは、実通信を行わず確認する。
    with patch.object(sys, "argv", ["prepare_data.py", "--download"]), \
            patch.object(prepare_data, "prepare") as operation:
        assert prepare_data.main() == 0
        operation.assert_called_once_with(download=True)


if __name__ == "__main__":
    (ROOT / "log/_work").mkdir(exist_ok=True)
    test_line_endings_and_tampering()
    test_download_bytes()
    test_no_implicit_download_or_overwrite()
    test_command_line()
    print("Passed: LF/CRLF, rejected modifications, strict/limited downloads, missing data guidance, CLI, no network")
