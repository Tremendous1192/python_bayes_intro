@echo off
chcp 65001 >nul

rem - 作成日: 2026-09-26

rem - 更新日: 2026-09-27。既存ウィンドウのターミナルとNotebookに設定する。

rem - 目的: uvとNotebookの保存先・実行系・並列数を、この学習環境に揃える。

rem - 役割: uv操作と専用.venvが共有する、環境変数設定値の原本。

rem - 使用方法: call env.cmd。同期後は call env.cmd configure でNotebookへ登録する。

rem - 検証: call env.cmd test。登録後に明示実行し、試験用ファイルはlog/_work内に限定。

rem - データ準備: call env.cmd prepare-data。不足分の取得時だけ --download を付ける。

rem - 確認・解除: call env.cmd check / call env.cmd unconfigure。既存カーネルは先に停止する。

rem - 制約: この配置のWindows 11 AMD64専用。設定は呼出元と子プロセスに有効。

rem - 非対応: VS Code起動、認証変更、環境の自動導入、EXE・GPU・condaとの混用。

rem - 引数の誤りでは設定を変更せず、利用できる操作だけを案内する。
if not "%~3"=="" goto :usage
if /i "%~1"=="prepare-data" goto :data_args
if not "%~2"=="" goto :usage
rem - 引数なしは現在のターミナルの設定だけを行う。
if "%~1"=="" goto :settings
rem - 環境の登録・確認・解除・テストの操作を選ぶ。
if /i "%~1"=="configure" goto :settings
rem - checkは登録内容を変更せず確認する。
if /i "%~1"=="check" goto :settings
rem - unconfigureは所有する登録だけを解除する。
if /i "%~1"=="unconfigure" goto :settings
if /i "%~1"=="test" goto :settings
goto :usage

:data_args
rem - 取得許可はデータ準備だけで受け付け、未知の指定は設定前に拒否する。
if "%~2"=="" goto :settings
if "%~2"=="--download" goto :settings
goto :usage

:settings

rem - 誤った配置から別プロジェクトの環境を変更しないよう、対象を限定する。
if /i not "%~dp0"=="C:\dev\python_bayes_intro\marimo\" (
    echo ERROR: Run env.cmd from its documented project location.
    exit /b 1
)
set "BAYES_PROJECT=C:\dev\python_bayes_intro\marimo"
rem - 絶対パスで呼ばれた場合も、uvが別フォルダの依存定義を使わないよう揃える。
cd /d "%BAYES_PROJECT%"
rem - 作業フォルダを確保できなければ、環境変数の変更前に停止する。
if errorlevel 1 exit /b 1

rem - 別環境の実行系・インデックス指定を解除し、プロジェクト定義を使用する。
set "PYTHONHOME="
set "PYTHONPATH="
set "VIRTUAL_ENV="
set "UV_CONFIG_FILE="
set "UV_INDEX="
set "UV_INDEX_URL="
set "UV_EXTRA_INDEX_URL="
set "UV_DEFAULT_INDEX="

rem - Python本体と仮想環境を固定し、通常操作での自動取得・共通登録を禁止する。
set "UV_CACHE_DIR=%BAYES_PROJECT%\.cache\uv"
set "UV_PYTHON_INSTALL_DIR=%BAYES_PROJECT%\.cache\python"
set "UV_PYTHON_BIN_DIR=%BAYES_PROJECT%\.cache\python-bin"
set "UV_PYTHON_INSTALL_BIN=0"
set "UV_PYTHON_INSTALL_REGISTRY=0"
set "UV_PROJECT_ENVIRONMENT=%BAYES_PROJECT%\.venv"
set "UV_PYTHON=%BAYES_PROJECT%\.cache\python\cpython-3.14.7-windows-x86_64-none\python.exe"
set "UV_PYTHON_DOWNLOADS=never"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUSERBASE=%BAYES_PROJECT%\.cache\python-user"

rem - ターミナルの個人領域は維持し、Notebookの個人領域だけを別途指定する。
set "TEMP=%BAYES_PROJECT%\.cache\tmp"
set "TMP=%TEMP%"
set "BAYES_RUNTIME_PROFILE=%BAYES_PROJECT%\.cache\profile"

rem - 描画、JIT、データ、marimo設定の書込先をプロジェクト内に限定する。
set "MPLCONFIGDIR=%BAYES_PROJECT%\.cache\matplotlib"
set "NUMBA_CACHE_DIR=%BAYES_PROJECT%\.cache\numba"
rem - PyTensorの設定文字列では、エスケープを避けるためスラッシュを使う。
set "PYTENSOR_FLAGS=base_compiledir=C:/dev/python_bayes_intro/marimo/.cache/pytensor"
set "TORCH_HOME=%BAYES_PROJECT%\.cache\torch"
set "TORCH_EXTENSIONS_DIR=%BAYES_PROJECT%\.cache\torch-extensions"
set "SEABORN_DATA=%BAYES_PROJECT%\.cache\seaborn"
set "XDG_CONFIG_HOME=%BAYES_PROJECT%\.cache\config"
set "XDG_CACHE_HOME=%BAYES_PROJECT%\.cache"
set "XDG_DATA_HOME=%BAYES_PROJECT%\.cache\data"
set "MARIMO_SKIP_UPDATE_CHECK=1"

rem - 導入時と計算時の並列数を制限し、メモリとCPUの過剰利用を避ける。
set "UV_CONCURRENT_DOWNLOADS=4"
set "UV_CONCURRENT_INSTALLS=4"
set "UV_CONCURRENT_BUILDS=1"
set "OMP_NUM_THREADS=2"
set "OPENBLAS_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "NUMBA_NUM_THREADS=2"
set "POLARS_MAX_THREADS=2"

rem - 初回だけ必要な保存先を作り、作成できなければ後続操作を停止する。
for %%D in ("%TEMP%" "%BAYES_RUNTIME_PROFILE%\AppData\Roaming" "%BAYES_RUNTIME_PROFILE%\AppData\Local") do (
    rem - 既存ディレクトリを維持し、不足する親ディレクトリもまとめて作る。
    if not exist "%%~D" mkdir "%%~D"
    rem - 作成失敗を見逃して既定の個人領域へ書き込むことを防ぐ。
    if not exist "%%~D" (
        echo ERROR: Cannot prepare a project-local directory.
        exit /b 1
    )
)
rem - 環境未作成の初回でも、uvによる準備に使える設定を正常に返す。
if "%~1"=="" exit /b 0
rem - 自動インストールを避け、利用者による同期を案内する。
if not exist "%BAYES_PROJECT%\.venv\Scripts\python.exe" (
    echo ERROR: Prepare the environment with HowToUse_uv.md, then run call env.cmd configure.
    exit /b 1
)
rem - 埋込Pythonをメモリ上で実行する。-Sで既存フックを読まず、修復時も入口を保つ。
"%BAYES_PROJECT%\.venv\Scripts\python.exe" -I -B -S -c "import pathlib,sys; p=pathlib.Path(sys.argv[1]); s=p.read_text(encoding='utf-8'); head,sep,body=s.partition(chr(10)+':__BAYES_PYTHON__'+chr(10)); assert sep and body, 'Embedded Python is missing'; globals()['__file__']=str(p); sys.argv=[str(p)]+[arg for arg in sys.argv[2:] if arg]; exec(compile(chr(10)*(head.count(chr(10))+2)+body,str(p),'exec'),globals())" "%~f0" "%~1" "%~2"
exit /b %errorlevel%

:usage
echo ERROR: Usage: call env.cmd [configure^|check^|unconfigure^|test^|prepare-data [--download]]
exit /b 2

:__BAYES_PYTHON__
import contextlib
import urllib.request
import tempfile
from unittest.mock import patch
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

# - 値はenv.cmdを原本とし、この一覧は出力を許可する非秘密のキーだけを定める。
RUNTIME_KEYS = (
    "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_PYTHON_BIN_DIR",
    "UV_PYTHON_INSTALL_BIN", "UV_PYTHON_INSTALL_REGISTRY", "UV_PROJECT_ENVIRONMENT",
    "UV_PYTHON", "UV_PYTHON_DOWNLOADS", "PYTHONUTF8", "PYTHONNOUSERSITE",
    "PYTHONDONTWRITEBYTECODE", "PYTHONUSERBASE", "TEMP", "TMP", "MPLCONFIGDIR",
    "NUMBA_CACHE_DIR", "PYTENSOR_FLAGS", "TORCH_HOME", "TORCH_EXTENSIONS_DIR",
    "SEABORN_DATA", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME",
    "MARIMO_SKIP_UPDATE_CHECK", "UV_CONCURRENT_DOWNLOADS", "UV_CONCURRENT_INSTALLS",
    "UV_CONCURRENT_BUILDS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS",
)
# - uvのWindowsランチャーが設定する正当なPYTHONHOMEは検証後に子へ残さない。
CLEAR_KEYS = ("PYTHONHOME", "VIRTUAL_ENV", "UV_CONFIG_FILE", "UV_INDEX", "UV_INDEX_URL",
              "UV_EXTRA_INDEX_URL", "UV_DEFAULT_INDEX")
# - env.cmd自身の場所を基準にし、呼出元ディレクトリに依存しない。
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
VENV = ROOT / ".venv"
SITE = VENV / "Lib" / "site-packages"
STATE = ROOT / ".cache" / "runtime-registration.json"
FILES = (SITE / "_bayes_runtime.py", SITE / "000_bayes_runtime.pth")
MARKER = "# Generated by env.cmd; bayes-runtime-v1\n"


def prepare(download: bool = False) -> None:
    """固定ファイルを検査する。引数は不足時の取得許可、戻り値なし。
    前提: sources.jsonが存在する。副作用: 許可時のみ不足CSVを保存する。
    不一致・取得失敗は例外。例: prepare(True)。既存ファイルは置換しない。
    """
    sources = json.loads((DATA_ROOT / "sources.json").read_text(encoding="utf-8"))
    # - 取得対象を3本に限定し、JSONから任意パスを指定させない。
    for name in ("iris.csv", "test_scores.csv", "irt-sample.csv"):
        target = DATA_ROOT / name
        source = sources[name]
        # - 既存データは読み取りのみで検査する。
        if target.exists():
            data = target.read_bytes()
        # - 明示指定した初回だけネットワークへ接続する。
        elif download:
            request = urllib.request.Request(source["url"], headers={"User-Agent": "bayes-notebook"})
            # - 接続・読取時間と最大サイズを制限する。
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(2_000_001)
            # - 想定外の応答や内容は保存しない。
            if len(data) > 2_000_000 or hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise ValueError(f"Invalid download: {name}")
            # - 排他作成により、同時処理や利用者のファイルを上書きしない。
            with target.open("xb") as handle:
                handle.write(data)
        # - 学習中の不足は勝手に取得せず、準備手順を示す。
        else:
            raise FileNotFoundError(f"Missing {name}; run call env.cmd prepare-data --download.")
        # - GitのCRLF変換だけを読込時に戻す。取得時は上記で元バイト列を厳密に検証する。
        if hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() != source["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}; preserve the file and inspect it.")
        print(f"Passed: {name}, {len(data)} bytes")


def validate_runtime() -> None:
    """実行系を確認する。"""
    # - コピー先・別Python・起動済みフックからの実行を拒否し、所有範囲を限定する。
    if (ROOT != Path("C:/dev/python_bayes_intro/marimo")
            or Path(sys.prefix).resolve() != VENV
            or Path(sys.executable).resolve() != VENV / "Scripts" / "python.exe"
            or sys.version_info[:3] != (3, 14, 7)
            or platform.python_implementation() != "CPython"
            or platform.machine() != "AMD64" or sys.platform != "win32"
            or not sys._is_gil_enabled() or not sys.flags.no_site):
        raise RuntimeError("Use call env.cmd with the project's GIL-enabled CPython 3.14.7.")


def read_values() -> dict[str, str]:
    """許可した設定値を取得する。"""
    values = {}
    # - 明示したキーだけを取得し、任意の親環境を保存しない。
    for key in RUNTIME_KEYS:
        value = os.environ.get(key, "")
        # - 空値・改行は壊れた設定として登録前に止める。
        if not value or "\n" in value or "\r" in value:
            raise RuntimeError(f"Missing or invalid {key}; run call env.cmd configure.")
        values[key] = value
    profile = Path(os.environ.get("BAYES_RUNTIME_PROFILE", ""))
    # - Notebook側の個人領域だけを固定し、VS Codeやターミナルの認証状態を保つ。
    if profile != ROOT / ".cache" / "profile":
        raise RuntimeError("Invalid Notebook profile location.")
    values.update(USERPROFILE=str(profile), APPDATA=str(profile / "AppData/Roaming"),
                  LOCALAPPDATA=str(profile / "AppData/Local"))
    # - 保存先はROOT以下だけを認め、数値などの設定と区別する。
    for key, value in values.items():
        candidate = value.removeprefix("base_compiledir=")
        # - Windowsの絶対パスを使う項目を検査し、範囲外への誘導を防ぐ。
        if ":" in candidate:
            path = Path(candidate).resolve()
            # - Python実行系も含め、このプロジェクト外は使用しない。
            if not path.is_relative_to(ROOT):
                raise RuntimeError(f"Path outside the project: {key}")
    return values


def render(values: dict[str, str]) -> dict[Path, bytes]:
    """起動コードを生成する。"""
    source = MARKER + f'''import os
import sys
if os.path.normcase(os.path.abspath(sys.prefix)) != {os.path.normcase(str(VENV))!r}:
    raise SystemExit("ERROR: This runtime hook belongs to another environment.")
home = os.environ.get("PYTHONHOME")
if (home and os.path.normcase(os.path.abspath(home)) != os.path.normcase(sys.base_prefix)) or os.environ.get("PYTHONPATH"):
    raise SystemExit("ERROR: Clear inherited PYTHONHOME/PYTHONPATH before using this kernel.")
os.environ.update({values!r})
for key in {CLEAR_KEYS!r}:
    os.environ.pop(key, None)
sys.dont_write_bytecode = True
'''
    # - フック欠落時の黙った継続を防ぎ、-Sの登録器から修復できる状態で停止する。
    loader = ('try:\n import _bayes_runtime\nexcept Exception:\n'
              ' raise SystemExit("ERROR: Invalid runtime registration. Run call env.cmd configure.") from None\n')
    pth = MARKER + f"import sys; sys.dont_write_bytecode = True; exec({loader!r})\n"
    return {FILES[0]: source.encode("utf-8"), FILES[1]: pth.encode("utf-8")}


def owned_files() -> dict[str, str]:
    """既存生成物の所有を検査する。"""
    # - 初回は他者の同名ファイルを採用せず、専用名の衝突として扱う。
    if not STATE.exists():
        # - 登録記録がない生成物は所有を証明できない。
        if any(path.exists() for path in FILES):
            raise RuntimeError("Unowned runtime files exist; preserve them and inspect manually.")
        return {}
    hashes = json.loads(STATE.read_text(encoding="utf-8"))
    # - 記録から任意の削除先を受け取らず、固定した2ファイルだけを検査する。
    if (not isinstance(hashes, dict) or set(hashes) != {path.name for path in FILES}
            or any(not isinstance(value, str) or len(value) != 64 for value in hashes.values())):
        raise RuntimeError("Invalid registration record; inspect the backup before recovery.")
    # - .venv再作成で2本とも消えた場合は、残った記録から安全に再登録できる。
    if not any(path.exists() for path in FILES):
        return hashes
    # - 不在・利用者による変更は上書きも削除もしない。
    for path in FILES:
        # - 手編集や片方だけの削除は、自動修復で隠さず停止する。
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != hashes[path.name]:
            raise RuntimeError("Runtime files changed; preserve them and inspect the backup.")
    return hashes


def verify(values: dict[str, str], expected: dict[Path, bytes]) -> None:
    """登録と新規Pythonを確認する。"""
    # - env.cmd変更後に登録を更新し忘れた場合を検出する。
    for path, data in expected.items():
        # - 欠落・古い内容ではカーネル再起動を促す前に再登録を要求する。
        if not path.is_file() or path.read_bytes() != data:
            raise RuntimeError("Runtime configuration is missing or stale; run call env.cmd configure.")
    probe = ("import os,sys,tempfile,site; expected=" + repr(values)
             + "; assert all(os.environ.get(k)==v for k,v in expected.items());"
             + " assert tempfile.gettempdir()==expected['TEMP'];"
             + " assert sys.dont_write_bytecode and not site.ENABLE_USER_SITE;"
             + " print('Passed: runtime values, local temp, bytecode and user-site isolation')")
    clean = dict(os.environ)
    # - ターミナルからの継承に依存せず、フックが値を設定できることを確認する。
    for key in RUNTIME_KEYS:
        clean.pop(key, None)
    probe_temp = ROOT / ".cache" / "wrong-temp-probe"
    probe_temp.mkdir(exist_ok=True)
    clean["TEMP"] = clean["TMP"] = str(probe_temp)
    result = subprocess.run([sys.executable, "-I", "-B", "-c", probe], env=clean,
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30)
    # - 子の環境全体や個人情報を出力せず、終了コードで異常を伝える。
    if result.returncode or result.stderr:
        raise RuntimeError("Fresh-process runtime verification failed; inspect project configuration.")
    print(result.stdout.strip())


def main() -> None:
    """指定操作を実行する。"""
    validate_runtime()
    action = sys.argv[1].lower() if len(sys.argv) == 2 else ""
    # - 不正な操作ではファイルを変更しない。
    if action not in {"configure", "check", "unconfigure"}:
        raise RuntimeError("Expected configure, check, or unconfigure.")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    lock = STATE.with_suffix(".lock")
    # - 同時設定を拒否する。既存ロックは別所有者の可能性があるため削除しない。
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("Runtime configuration in progress\n")
    # - 所有するロックを成功・失敗にかかわらず最後に解除する。
    try:
        hashes = owned_files()
        # - 解除は所有が確認できたファイルだけを退避して行う。
        if action == "unconfigure":
            # - 未登録なら何も削除せず正常終了する。
            if hashes:
                backup = STATE.parent / "runtime-before-unconfigure"
                backup.mkdir(exist_ok=True)
                # - 既存の退避と異なる場合は保全して停止する。
                for path in (*FILES, STATE):
                    # - .venv再作成直後に既に消えた生成物は、退避・削除の対象にしない。
                    if not path.exists():
                        continue
                    target = backup / path.name
                    # - 手元の退避を無断で置き換えない。
                    if target.exists() and target.read_bytes() != path.read_bytes():
                        raise RuntimeError("Previous removal backup differs; preserve it before proceeding.")
                    target.write_bytes(path.read_bytes())
                # - 全退避成功後に、固定された所有ファイルだけを解除する。
                for path in (*FILES, STATE):
                    # - 不在の生成物を新たに作成してから消すことはしない。
                    if path.exists():
                        path.unlink()
            print("Passed: runtime registration removed; stop using kernels until configured again")
            return
        values = read_values()
        expected = render(values)
        # - configureだけが生成物を更新する。checkは内容を変更しない。
        if action == "configure":
            # - 更新前の生成物を保存し、別ファイルとして追跡できるようにする。
            for path, data in expected.items():
                # - 内容が同じなら再生成せず、カーネルの読取と衝突させない。
                if path.exists() and path.read_bytes() == data:
                    continue
                # - 既存版は内容ハッシュ名で保存して上書きしない。
                if path.exists():
                    saved = STATE.parent / (path.name + "." + hashes[path.name] + ".bak")
                    saved.write_bytes(path.read_bytes())
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_bytes(data)
                temporary.replace(path)
            record = {path.name: hashlib.sha256(data).hexdigest() for path, data in expected.items()}
            record_temp = STATE.with_suffix(".json.tmp")
            record_temp.write_text(json.dumps(record, indent=2), encoding="utf-8")
            record_temp.replace(STATE)
        verify(values, expected)
        print("Passed: restart the Notebook kernel to apply registered settings")
    # - 処理全体の原子性は主張せず、自分の操作ロックだけ確実に解除する。
    finally:
        lock.unlink()


# - 同じモジュールの所有先を試験領域へ差し替え、実.venvを変更しない。
config = sys.modules[__name__]
TEST_ROOT = ROOT / "log/_work/runtime-tests"


def test_lifecycle() -> None:
    """所有判定と解除を確認する。"""
    print("Checking registration ownership and recovery", flush=True)
    # - 実環境に触れず、生成物と登録記録を一時ディレクトリへ差し替える。
    with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
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
            # - 旧登録器の生成物も所有記録が一致すれば安全に更新できる。
            files[0].write_bytes(files[0].read_bytes().replace(b"Generated by env.cmd", b"Generated by configure_runtime.py"))
            state.write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}), encoding="utf-8")
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
    """初期化済み環境の新規プロセスを確認する。"""
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
    """未準備時のバッチを検証する。"""
    print("Checking unprepared-environment and command errors", flush=True)
    # - 固定配置の判定を試験用コピー内だけで置換し、未作成の環境を再現する。
    with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
        root = Path(temporary)
        source = (config.ROOT / "env.cmd").read_text(encoding="utf-8")
        source = source.replace(str(config.ROOT), str(root)).replace(config.ROOT.as_posix(), root.as_posix())
        script = root / "env.cmd"
        script.write_bytes(source.replace("\n", "\r\n").encode("utf-8"))
        cases = (("", 0, ""), ("configure", 1, "Prepare the environment"),
                 ("unexpected", 2, "Usage:"), ("test extra", 2, "Usage:"),
                 ("check", 1, "Prepare the environment"), ("test", 1, "Prepare the environment"),
                 ("prepare-data", 1, "Prepare the environment"),
                 ("prepare-data --download", 1, "Prepare the environment"),
                 ("PREPARE-DATA", 1, "Prepare the environment"),
                 ("prepare-data unexpected", 2, "Usage:"),
                 ("prepare-data --download extra", 2, "Usage:"),
                 ("check --download", 2, "Usage:"), ("--download", 2, "Usage:"))
        # - 誤操作と環境未準備を区別し、ダウンロードを起動しないことを確認する。
        for arguments, expected_code, message in cases:
            result = subprocess.run(f'cmd.exe /d /c call "{script}" {arguments}',
                                    capture_output=True, timeout=15)
            assert result.returncode == expected_code, arguments
            assert message.encode() in result.stdout, arguments
            assert not result.stderr, "Unexpected batch error"
            assert not (root / ".venv").exists()
    print("Passed: initial setup, missing environment, invalid argument, no installation\n", flush=True)


# - 明示したtestだけが使い捨て検証を実行する。通常の登録操作では呼ばない。
if __name__ == "__main__":
    try:
        if sys.argv[1].lower() == "test":
            validate_runtime()
            TEST_ROOT.mkdir(parents=True, exist_ok=True)
            test_lifecycle()
            test_fresh_process()
            test_cmd_errors()
        elif sys.argv[1].lower() == "prepare-data":
            validate_runtime()
            prepare(download=sys.argv[2:] == ["--download"])
        else:
            main()
    except (OSError, RuntimeError, ValueError, AssertionError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
