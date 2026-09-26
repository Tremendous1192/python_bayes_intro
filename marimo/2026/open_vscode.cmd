@echo off
chcp 65001 >nul
rem 作成日: 2026-09-26

rem 目的: 保存先設定を継承した、この学習環境専用のVS Codeを開く。

rem 役割: 既存のVS Codeプロセスから環境変数を再利用する動作を避ける。

rem 使用方法: Command Promptで open_vscode.cmd を実行する。

rem 制約: Windows版VS CodeがPATH上に必要。専用ウィンドウで学習する。

rem 非対応: VS Code本体の導入、拡張機能の自動導入、他ウィンドウの終了。
setlocal

rem 共通設定に失敗した場合は、VS Codeを不完全な環境で起動しない。
call "%~dp0env.cmd"
rem 呼出元のエラーコードを引き継ぎ、保存先未設定のまま進めない。
if errorlevel 1 exit /b 1

rem 仮想環境の準備を先に済ませ、拡張機能からの暗黙の環境導入を避ける。
if not exist "%BAYES_PROJECT%\.venv\Scripts\python.exe" (
    echo ERROR: Prepare the Python environment using uv_HowToUse.md first.
    exit /b 1
)
rem ファイルの存在だけで判断せず、通常版CPythonとmarimoの導入を確かめる。
"%BAYES_PROJECT%\.venv\Scripts\python.exe" -I -B -c "import sys, platform, importlib.util; assert platform.python_implementation() == 'CPython'; assert sys.version_info[:3] == (3, 14, 7); assert sys._is_gil_enabled(); assert importlib.util.find_spec('marimo') is not None"
rem 版・GIL・依存が違う環境では起動せず、明示的な復元を要求する。
if errorlevel 1 (
    echo ERROR: Expected CPython 3.14.7 with GIL and marimo in the project environment.
    exit /b 1
)
where code.cmd >nul 2>&1
rem Windows版の公式CLIが見つからなければ、導入先確認を促して停止する。
if errorlevel 1 (
    echo ERROR: The Windows VS Code command code.cmd was not found on PATH.
    exit /b 1
)

rem VS Codeの状態・拡張機能・ログも、このプロジェクトの除外領域へ保存する。
set "BAYES_CODE_DATA=%BAYES_PROJECT%\.cache\vscode-user"
set "BAYES_CODE_EXTENSIONS=%BAYES_PROJECT%\.cache\vscode-extensions"
rem 独立したユーザー設定と拡張機能の保存先を一つずつ準備する。
for %%D in ("%BAYES_CODE_DATA%\User" "%BAYES_CODE_EXTENSIONS%") do (
    rem 初回だけディレクトリを作成し、既存の設定はそのまま利用する。
    if not exist "%%~D" mkdir "%%~D"
    rem 保存先を確保できない場合は、別の場所へフォールバックしない。
    if not exist "%%~D" (
        echo ERROR: Cannot prepare the isolated VS Code directories.
        exit /b 1
    )
)

rem 初回の専用ユーザー設定だけ生成し、以後の利用者の設定を上書きしない。
if not exist "%BAYES_CODE_DATA%\User\settings.json" (
    >"%BAYES_CODE_DATA%\User\settings.json" (
        echo {
        echo     "update.mode": "none",
        echo     "extensions.autoUpdate": false,
        echo     "extensions.autoCheckUpdates": false,
        echo     "telemetry.telemetryLevel": "off"
        echo }
    )
    rem 設定保存に失敗した場合も、既定の自動更新状態で起動しない。
    if errorlevel 1 (
        echo ERROR: Cannot write the isolated VS Code settings.
        exit /b 1
    )
)

rem 同じ専用ウィンドウを再開するときも、状態の保存先を必ず明示する。
call code.cmd --new-window --user-data-dir "%BAYES_CODE_DATA%" --extensions-dir "%BAYES_CODE_EXTENSIONS%" "%BAYES_PROJECT%"
exit /b %errorlevel%
