# - 作成日: 2026-09-27
# - 目的: Notebook専用の登録器の所有判定・復旧・起動設定を検証する。
# - 役割: 他者ファイルの保全と設定の再現性を、使い捨てデータで確認する。
# - 使用: call env.cmd後、uv run --locked --group notebook python log/test_configure_runtime.py。
# - 制約: 実環境は登録済み。変更する試験データは.cache内だけ。
# - 非対応: VS Code画面操作、ログイン状態、EXE配布、他のPython環境。
import contextlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import configure_runtime as config


def test_lifecycle() -> None:
    """所有判定と解除を確認する。引数・戻り値なし。前提はenv.cmd適用済み。
    .cacheの一時ディレクトリだけを更新・削除する。不一致はAssertionError。
    例: test_lifecycle()。実.venvの起動フックは変更しない。
    """
    print("Checking registration ownership and recovery", flush=True)
    # - 実環境に触れず、生成物と登録記録を一時ディレクトリへ差し替える。
    with tempfile.TemporaryDirectory(dir=config.ROOT / ".cache") as temporary:
        root = Path(temporary)
        site = root / ".venv/Lib/site-packages"
        site.mkdir(parents=True)
        files = (site / "_bayes_runtime.py", site / "000_bayes_runtime.pth")
        state = root / ".cache/runtime-registration.json"
        # - 登録器のファイル操作だけを試験し、実際の起動検証は別ケースで行う。
        with contextlib.ExitStack() as stack:
            # - 実装と同じ固定2ファイルを、閉じた一時領域に限定する。
            for name, value in {"ROOT": root, "VENV": root / ".venv", "SITE": site,
                                "STATE": state, "FILES": files}.items():
                stack.enter_context(patch.object(config, name, value))
            stack.enter_context(patch.object(config, "validate_runtime"))
            stack.enter_context(patch.object(config, "read_values", return_value={"OMP_NUM_THREADS": "2"}))
            stack.enter_context(patch.object(config, "verify"))
            stack.enter_context(patch.object(sys, "argv", ["configure_runtime.py", "configure"]))
            files[0].write_text("USER CONTENT", encoding="utf-8")
            # - 未所有の同名ファイルがあれば、内容を保全して登録を拒否する。
            try:
                config.main()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Unowned file was accepted")
            assert files[0].read_text(encoding="utf-8") == "USER CONTENT"
            files[0].unlink()
            config.main()
            before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}
            config.main()
            assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}
            files[0].write_bytes(before[files[0]][0] + b"# user edit\n")
            # - 所有記録があっても利用者の変更を上書きしない。
            try:
                config.main()
            except RuntimeError:
                pass
            else:
                raise AssertionError("User edit was overwritten")
            assert files[0].read_bytes().endswith(b"# user edit\n")
            files[0].write_bytes(before[files[0]][0])
            lock = state.with_suffix(".lock")
            lock.write_text("OTHER OWNER", encoding="utf-8")
            # - 他者が保持するロックはエラーになっても残る。
            try:
                config.main()
            except FileExistsError:
                pass
            else:
                raise AssertionError("Concurrent configuration was accepted")
            assert lock.read_text(encoding="utf-8") == "OTHER OWNER"
            lock.unlink()
            sys.argv[1] = "unconfigure"
            config.main()
            assert not any(path.exists() for path in (*files, state))
            backup = state.parent / "runtime-before-unconfigure"
            assert (backup / files[0].name).read_bytes() == before[files[0]][0]
            sys.argv[1] = "configure"
            config.main()
            assert all(path.is_file() for path in (*files, state))
            # - 仮想環境再作成で生成物だけが消え、.cacheの記録が残る条件を再現する。
            for path in files:
                path.unlink()
            config.main()
            assert all(path.is_file() for path in files)
            record = state.read_bytes()
            state.write_text(json.dumps([path.name for path in files]), encoding="utf-8")
            # - 有効なJSONでも記録形式が違えば、生成物を保全して拒否する。
            try:
                config.main()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Malformed ownership record was accepted")
            assert all(path.is_file() for path in files)
            state.write_bytes(record)
            files[0].unlink()
            # - 片方だけの欠落は再作成と決めつけず、残ったファイルを保全する。
            try:
                config.main()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Partial damage was silently overwritten")
            assert files[1].is_file()
    print("Passed: conflicts, repeat, lock, removal, recreation and partial damage\n", flush=True)


def test_fresh_process() -> None:
    """初期化済み環境の新規プロセスを確認する。引数・戻り値なし。
    前提はconfigure実施済み。子を逐次起動、各30秒。外部データ取得なし。
    不一致はAssertionError。例: test_fresh_process()。GUI検証を代替しない。
    """
    print("Checking a fresh bridge and kernel without terminal settings", flush=True)
    env = dict(os.environ)
    # - VS Codeからの起動と同様、ターミナルの設定を渡さずフックを実行する。
    for key in (*config.RUNTIME_KEYS, "PYTHONHOME", "PYTHONPATH"):
        env.pop(key, None)
    env["TEMP"] = env["TMP"] = str(config.ROOT / ".cache")
    env["OMP_NUM_THREADS"] = "19"
    child = ("import os,sys,tempfile,site,json; import numpy,matplotlib; "
             "print(json.dumps([sys.executable,sys.flags.utf8_mode,os.environ['OMP_NUM_THREADS'],"
             "tempfile.gettempdir(),sys.dont_write_bytecode,site.ENABLE_USER_SITE]))")
    bridge = ("import subprocess,sys; subprocess.run([sys.executable,'-c',"
              + repr(child) + "],check=True,timeout=30)")
    result = subprocess.run([sys.executable, "-c", bridge], env=env, cwd=config.ROOT,
                            capture_output=True, text=True, encoding="utf-8", timeout=40)
    assert result.returncode == 0 and not result.stderr, "Fresh kernel failed"
    values = json.loads(result.stdout)
    assert Path(values[0]) == config.VENV / "Scripts/python.exe"
    assert values[1:] == [1, "2", str(config.ROOT / ".cache/tmp"), True, False]
    assert env["OMP_NUM_THREADS"] == "19"
    print("Passed: selected Python, UTF-8 child, thread limit, local temp, unchanged parent\n", flush=True)


def test_cmd_errors() -> None:
    """未準備時のバッチを検証する。引数・戻り値なし。前提はWindows。
    .cache内の複製だけを起動し、各15秒で終了する。不一致はAssertionError。
    例: test_cmd_errors()。実際の.venvは移動・削除しない。
    """
    print("Checking unprepared-environment and command errors", flush=True)
    # - 固定配置の判定を試験用コピー内だけで置換し、未作成の環境を再現する。
    with tempfile.TemporaryDirectory(dir=config.ROOT / ".cache") as temporary:
        root = Path(temporary)
        source = (config.ROOT / "env.cmd").read_text(encoding="utf-8")
        source = source.replace(str(config.ROOT), str(root))
        script = root / "env.cmd"
        script.write_bytes(source.replace("\n", "\r\n").encode("utf-8"))
        cases = (("", 0, ""), ("configure", 1, "Prepare the environment"),
                 ("unexpected", 2, "Usage:"))
        # - 誤操作と環境未準備を区別し、ダウンロードを起動しないことを確認する。
        for arguments, expected_code, message in cases:
            result = subprocess.run(f'cmd.exe /d /c call "{script}" {arguments}',
                                    capture_output=True, timeout=15)
            assert result.returncode == expected_code, arguments
            assert message.encode() in result.stdout, arguments
            assert not result.stderr, "Unexpected batch error"
            assert not (root / ".venv").exists()
    print("Passed: initial setup, missing environment, invalid argument, no installation\n", flush=True)


# - 全ケースを順番に実行し、失敗は英語表示と非ゼロ終了で返す。
if __name__ == "__main__":
    # - 他ケースに進む前に失敗を通知して終了する。
    try:
        test_lifecycle()
        test_fresh_process()
        test_cmd_errors()
    # - 例外の詳細は失わず、失敗した検証であることを明示する。
    except Exception:
        print("Failed: runtime registration regression check\n", flush=True)
        raise
