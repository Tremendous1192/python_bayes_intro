@echo off
chcp 65001 >nul
rem - 作成日: 2026-09-26
rem - 更新日: 2026-09-27。既存ウィンドウのターミナルとNotebookに設定する。

rem - 目的: uvとNotebookの保存先・実行系・並列数を、この学習環境に揃える。

rem - 役割: uv操作と専用.venvが共有する、環境変数設定値の原本。

rem - 使用方法: call env.cmd。同期後は call env.cmd configure でNotebookへ登録する。

rem - 確認・解除: call env.cmd check / call env.cmd unconfigure。既存カーネルは先に停止する。

rem - 制約: この配置のWindows 11 AMD64専用。設定は呼出元と子プロセスに有効。

rem - 非対応: VS Code起動、認証変更、環境の自動導入、EXE・GPU・condaとの混用。

rem - 引数の誤りでは設定を変更せず、利用できる操作だけを案内する。
if not "%~2"=="" goto :usage
rem - 引数なしは現在のターミナルの設定だけを行う。
if "%~1"=="" goto :settings
rem - 登録・確認・解除以外の操作は実行しない。
if /i "%~1"=="configure" goto :settings
rem - checkは登録内容を変更せず確認する。
if /i "%~1"=="check" goto :settings
rem - unconfigureは所有する登録だけを解除する。
if /i "%~1"=="unconfigure" goto :settings
goto :usage

:settings

rem - 誤った配置から別プロジェクトの環境を変更しないよう、対象を限定する。
if /i not "%~dp0"=="C:\dev\python_bayes_intro\marimo\2026\" (
    echo ERROR: Run env.cmd from its documented project location.
    exit /b 1
)
set "BAYES_PROJECT=C:\dev\python_bayes_intro\marimo\2026"
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
set "PYTENSOR_FLAGS=base_compiledir=C:/dev/python_bayes_intro/marimo/2026/.cache/pytensor"
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
rem - 登録器が欠けた配置では、不完全な初期化を行わない。
if not exist "%BAYES_PROJECT%\log\configure_runtime.py" (
    echo ERROR: log/configure_runtime.py is missing. Restore the project files.
    exit /b 1
)
rem - -Sで古い登録を読まずに修復・解除し、-Iと-Bで外部設定とpycを抑止する。
"%BAYES_PROJECT%\.venv\Scripts\python.exe" -I -B -S "%BAYES_PROJECT%\log\configure_runtime.py" %1
exit /b %errorlevel%

:usage
echo ERROR: Usage: call env.cmd [configure^|check^|unconfigure]
exit /b 2
