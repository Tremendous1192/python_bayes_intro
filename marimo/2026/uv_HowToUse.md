# uv利用手順

作成日・検証日: 2026-09-26 JST。
Windows 11 AMD64、uv 0.11.7、通常のGIL付きCPython 3.14.7で確認した手順です。
構成・採用バージョン・検証結果は[README](README.md)を参照してください。
この環境はNotebook専用です。conda環境の作成、`conda init`、PyInstallerは不要です。

## 1. VS CodeのCommand Promptを準備する

VS Codeで`Terminal: Select Default Profile`から`Command Prompt`を選び、
新しい統合ターミナルを開きます。以下のコマンドはすべて`cmd.exe`用です。
環境変数はこのターミナルと子プロセスだけに設定し、`setx`やシステム設定は使いません。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
rem UTF-8のツール出力を、このターミナルで正しく表示する。
chcp 65001>nul
set "BAYES_PROJECT=C:\dev\python_bayes_intro\marimo\2026"

rem 別環境の標準ライブラリや仮想環境指定を継承しないようにする。
set "PYTHONHOME="
set "PYTHONPATH="
set "VIRTUAL_ENV="

rem 別プロジェクトのインデックス指定を解除し、このpyproject.tomlを原本にする。
set "UV_CONFIG_FILE="
set "UV_INDEX="
set "UV_INDEX_URL="
set "UV_EXTRA_INDEX_URL="
set "UV_DEFAULT_INDEX="

rem 実行系・依存・一時ファイルの書込先を、このプロジェクト内に限定する。
set "UV_CACHE_DIR=%BAYES_PROJECT%\.cache\uv"
set "UV_PYTHON_INSTALL_DIR=%BAYES_PROJECT%\.cache\python"
set "UV_PYTHON_BIN_DIR=%BAYES_PROJECT%\.cache\python-bin"
set "UV_PYTHON_INSTALL_BIN=0"
set "UV_PYTHON_INSTALL_REGISTRY=0"
set "UV_PROJECT_ENVIRONMENT=%BAYES_PROJECT%\.venv"
set "UV_PYTHON_DOWNLOADS=never"
set "TEMP=%BAYES_PROJECT%\.cache\tmp"
set "TMP=%BAYES_PROJECT%\.cache\tmp"
rem TEMPとTMPが参照するディレクトリを、未作成の場合だけ用意する。
if not exist "%TEMP%" mkdir "%TEMP%"

rem 描画・JIT・Notebookの設定とログも、個人領域へ保存しないようにする。
set "MPLCONFIGDIR=%BAYES_PROJECT%\.cache\matplotlib"
set "NUMBA_CACHE_DIR=%BAYES_PROJECT%\.cache\numba"
rem PyTensorの設定文字列ではバックスラッシュが解釈されるため、スラッシュを使用する。
set "PYTENSOR_FLAGS=base_compiledir=C:/dev/python_bayes_intro/marimo/2026/.cache/pytensor"
set "TORCH_HOME=%BAYES_PROJECT%\.cache\torch"
set "TORCH_EXTENSIONS_DIR=%BAYES_PROJECT%\.cache\torch-extensions"
set "SEABORN_DATA=%BAYES_PROJECT%\.cache\seaborn"
set "XDG_CONFIG_HOME=%BAYES_PROJECT%\.cache\config"
set "XDG_CACHE_HOME=%BAYES_PROJECT%\.cache"
set "XDG_DATA_HOME=%BAYES_PROJECT%\.cache\data"
set "MARIMO_SKIP_UPDATE_CHECK=1"
set "PYTHONUTF8=1"

rem 初回導入と数値検証で使用した並列数の上限を設定する。
set "UV_CONCURRENT_DOWNLOADS=4"
set "UV_CONCURRENT_INSTALLS=4"
set "UV_CONCURRENT_BUILDS=1"
set "OMP_NUM_THREADS=2"
set "OPENBLAS_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "NUMBA_NUM_THREADS=2"
set "POLARS_MAX_THREADS=2"

uv --version
dot -V
g++ --version
```

`uv`は導入済みのものを使用します。今回の作業では更新していません。
`dot`はPythonパッケージ`graphviz`とは別の実行ファイルです。
この端末ではGraphviz 13.1.2とMinGW g++ 16.1.0を確認しました。
別端末で不足している場合は導入先・権限を確認し、Python依存の変更で代用しないでください。

## 2. Python 3.14.7を明示的に導入する

この端末のuv 0.11.7には3.14.7の配布情報が内蔵されていません。
uv本体を更新せず、2026-09-24のAstral公式コミットに固定した配布カタログを使用します。
取得するPythonはCPython 3.14.7、Windows x86_64、通常のGIL付き、ビルド20260924です。

```cmd
curl.exe --fail --location --connect-timeout 15 --max-time 120 --output .cache\python-downloads.json https://raw.githubusercontent.com/astral-sh/uv/299a93de4b94e754f260c673d2de456afbdd4fb7/crates/uv-python/download-metadata.json
certutil -hashfile .cache\python-downloads.json SHA256
```

カタログのSHA-256が以下と一致することを確認してから次へ進みます。
取得失敗・不一致の場合は導入を止め、URLと通信状態を確認してください。

```text
016746da52b4558782e2e71e621c682db786a0d07025a5b67bdac08c668f1aa2
```

```cmd
rem 明示的なPython導入だけ取得を許可し、自動取得は禁止したままにする。
set "UV_PYTHON_DOWNLOADS=manual"
uv python install 3.14.7 --no-bin --no-registry --python-downloads-json-url .cache\python-downloads.json
rem 通常の環境操作ではPythonを勝手に取得しない設定へ戻す。
set "UV_PYTHON_DOWNLOADS=never"
.cache\python\cpython-3.14.7-windows-x86_64-none\python.exe -I -B -c "import sys; print(sys.version); print(sys.executable)"
```

`--no-bin`と`--no-registry`により、共通のPythonコマンドやWindowsレジストリへ登録しません。
Python配布物のSHA-256は公式カタログに含まれ、uvが照合します。
今回の配布物は`1493fc4185edf84bbd4305c15c5fbac2d4fcd4ddd7eb6273903669a5d3106178`です。

## 3. ロック済み環境を復元する

```cmd
uv lock --check
uv sync --locked --group notebook --python .cache\python\cpython-3.14.7-windows-x86_64-none\python.exe
uv run --locked --group notebook python -c "import sys, platform; print(sys.executable); print(sys.version); print(platform.machine()); assert sys.version_info[:3] == (3, 14, 7); assert sys._is_gil_enabled()"
```

実行先は`C:\dev\python_bayes_intro\marimo\2026\.venv\Scripts\python.exe`になります。
`notebook`グループは明示指定が必要です。marimoを使うときも同じ指定を続けます。
`uv sync`は対象環境を依存定義へ合わせるため、別用途のパッケージをこの`.venv`へ混在させません。
`pip install`や`uv pip install`による未記録の追加、`--frozen`だけの検証は行いません。

`.cache/python`には`.venv`が参照するPython本体があります。
環境の使用中に`.cache`全体を削除・移動すると起動できなくなります。
別の配置へ移す場合は、元のNotebookを保全して実行系と仮想環境を作り直します。

## 4. VS Codeとmarimoを起動する

VS Codeの`Python: Select Interpreter`で、次の実行ファイルを選択します。

```text
C:\dev\python_bayes_intro\marimo\2026\.venv\Scripts\python.exe
```

marimo拡張機能を使用する場合も、このプロジェクトの実行系を選びます。
拡張機能が別プロセスを起動するとターミナルの環境変数を継承しない場合があるため、
保存先を確実に揃える起動方法は、以下の統合ターミナルからのコマンドです。

Windows版marimo 0.25.0では履歴などの一部がユーザープロファイル配下に保存されます。
子`cmd.exe`の`USERPROFILE`だけを作業用ディレクトリに設定します。
Windowsのアカウント設定、親ターミナルのプロファイル、既存の個人設定は変更しません。

```cmd
rem marimo専用の作業用プロファイルを、未作成の場合だけ用意する。
if not exist "%BAYES_PROJECT%\.cache\marimo-profile" mkdir "%BAYES_PROJECT%\.cache\marimo-profile"
cmd.exe /d /c "set USERPROFILE=%BAYES_PROJECT%\.cache\marimo-profile&& uv run --locked --group notebook marimo edit --headless --no-sandbox --host 127.0.0.1"
```

ターミナルに表示されたURLをブラウザーで開きます。認証トークン付きURLは共有しません。
`--headless`はブラウザーの自動起動を止めます。`--no-sandbox`はプロジェクト環境を使用する指定です。
停止するときは、そのターミナルで`Ctrl+C`を押します。

新しいNotebookを作る場合の例です。`first_notebook.py`は未作成の例示名です。

```cmd
cmd.exe /d /c "set USERPROFILE=%BAYES_PROJECT%\.cache\marimo-profile&& uv run --locked --group notebook marimo edit first_notebook.py --headless --no-sandbox --host 127.0.0.1"
```

セル内でも実行系を確認できます。

```python
# サーバーの起動元だけでなく、実際にセルを実行するPythonを確認する。
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
```

Jupyter、`uvx`、`--with`、Notebook内のインストール処理は、この環境の依存管理に使用しません。
書籍Notebookの移植時は、重いサンプリングセルの再実行条件も明示してください。

## 5. 確認・依存変更に使うコマンド

```cmd
uv lock --check
uv run --locked --group notebook marimo --version
uv run --locked --group notebook python -c "import numpy, polars, pandas, scipy, torch, pymc, nutpie, arviz, numba, graphviz, openpyxl, matplotlib_fontja; print('Imports passed')"
```

依存変更は次の順で行います。

1. 2026-09-26 JST以前の安定版とWindows/Python互換性を公式情報で確認します。
2. `pyproject.toml`の対象依存を変更し、採用理由をREADMEへ記録します。
3. 次のコマンドでロックを再生成・確認・同期します。
4. 変更されたライブラリの描画・数値処理・Notebookを、新しいセッションから確認します。

```cmd
uv lock
uv lock --check
uv sync --locked --group notebook
```

`uv.lock`は手編集しません。初回生成はuv 0.11.7で実行し、1,025行・83パッケージになりました。
`exclude-newer`は2026-09-26 JSTの終了境界を固定し、個々の配布物の公開日時を制限します。
直接依存の版も固定しています。基準日を変える更新は、別の要件変更として扱います。

## 6. 問題の切り分けと検証範囲

| 症状 | 確認する内容 |
| --- | --- |
| marimo起動時のDLLエラー、別Pythonの標準ライブラリを参照 | 手順1で`PYTHONHOME`と`PYTHONPATH`を解除し、`sys.executable`を再確認する。今回実際に検出・解消した問題。 |
| Python 3.14.7が見つからない | 固定カタログと`UV_PYTHON_INSTALL_DIR`を確認する。3.13へ置き換えない。 |
| Python downloads are not allowed | 明示的な導入時だけ`UV_PYTHON_DOWNLOADS=manual`とし、終了後に`never`へ戻す。 |
| lockとmanifestの不一致 | 意図した変更か確認してから`uv lock`で更新する。`--frozen`で回避しない。 |
| Graphviz実行失敗 | 同じターミナルで`dot -V`を確認する。Python版`graphviz`だけでは本体は入らない。 |
| PyTensorのコンパイラ警告 | 同じターミナルの`g++ --version`と実際の数値実行を確認する。コンパイラを自動追加しない。 |
| 旧書籍の描画・推論APIで失敗 | PyMC 6／ArviZ 1のAPI移行を行う。今回の環境準備は16本の移植完了を意味しない。 |
| 外部CSV・Excelの取得失敗 | 元URL、通信、保存先を確認する。完全オフライン対応は今回の対象外。 |

今回、合成データによるExcel読込、型を明示したPolars処理、PyTorchのCPU自動微分、
Numbaと非JIT結果の比較、Matplotlib/Seaborn/Graphviz描画を確認しました。
PyMCとnutpieは同じBeta-Bernoulliモデルを各2 chains・400 tune・600 draws、`cores=1`、
`random_seed=42`で実行し、事後平均が解析解`2/7`との差`0.06`未満であることを確認しました。
marimoは新規プロセスのNotebook実行と、認証付きHTTP起動・終了APIによる正常終了を確認しました。
検証用コードと画像はGit対象外の`.cache/validation`にあり、キャッシュ削除後の存在は保証しません。
VS Code GUIの操作確認、ブラウザーでのセル編集、移植元16本の全実行は未実施です。

## 参考資料

- [Python 3.14.7公式リリース](https://www.python.org/downloads/release/python-3147/)
- [uvのPython導入](https://docs.astral.sh/uv/guides/install-python/)
- [uvのロックと同期](https://docs.astral.sh/uv/concepts/projects/sync/)
- [uvの公開日時制限](https://docs.astral.sh/uv/concepts/resolution/#reproducible-resolutions)
- [marimoのプロジェクト環境](https://docs.marimo.io/guides/package_management/projects/)
- [marimoの設定](https://docs.marimo.io/guides/configuration/)
