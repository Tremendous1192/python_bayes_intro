# uv利用手順

uvで、このプロジェクトのPythonとライブラリを準備します。
初回は第1～4節、準備済みなら新しいターミナルで第1節を実行してから[marimo利用手順](marimo_HowToUse.md)へ進んでください。

確認済みの構成（2026-09-26）：Windows 11 AMD64 / uv 0.11.7 / CPython 3.14.7（GIL有効）。
採用バージョンと検証記録は[README](README.md)にあります。

## 1. ターミナルを準備する（毎回）

VS Codeの`Terminal: Select Default Profile`で`Command Prompt`を選び、新しいターミナルを開きます。
**以下をまとめて実行してください。** 別のPython環境の設定を外し、保存先とスレッド数を揃えます。
設定はこのターミナルと子プロセスにだけ有効です。

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

`uv`、Graphviz本体（`dot`）、MinGW（`g++`）は導入済みのものを使います。
この端末の確認版は、それぞれ0.11.7、13.1.2、16.1.0です。
見つからない場合は導入先を確認してください。Pythonパッケージの`graphviz`だけでは`dot`は入りません。

## 2. Pythonを導入する（初回のみ）

Python 3.14.7を導入済みなら、第3節へ進みます。
uv 0.11.7にこの版の配布情報がないため、固定したAstral公式カタログを使います。

```cmd
curl.exe --fail --location --connect-timeout 15 --max-time 120 --output .cache\python-downloads.json https://raw.githubusercontent.com/astral-sh/uv/299a93de4b94e754f260c673d2de456afbdd4fb7/crates/uv-python/download-metadata.json
certutil -hashfile .cache\python-downloads.json SHA256
```

SHA-256が次の値と一致したら、導入へ進みます。取得失敗や不一致の場合は止めてください。

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

共通のPythonコマンドやWindowsレジストリには登録しません。
取得先はプロジェクト内の`.cache/python/`です。

<details>
<summary>Python配布物の詳細</summary>

CPython 3.14.7 / Windows x86_64 / GIL有効 / Astralビルド20260924。
配布物のSHA-256は`1493fc4185edf84bbd4305c15c5fbac2d4fcd4ddd7eb6273903669a5d3106178`で、
uvがカタログの値と照合します。

</details>

## 3. ライブラリを揃える（初回・環境の復元時）

`uv.lock`に記録された版を、このプロジェクトの`.venv`へ導入します。

```cmd
uv lock --check
uv sync --locked --group notebook --python .cache\python\cpython-3.14.7-windows-x86_64-none\python.exe
uv run --locked --group notebook python -c "import sys, platform; print(sys.executable); print(sys.version); print(platform.machine()); assert sys.version_info[:3] == (3, 14, 7); assert sys._is_gil_enabled()"
```

出力されたPythonの場所が次のパスで、版が3.14.7なら準備完了です。

```text
C:\dev\python_bayes_intro\marimo\2026\.venv\Scripts\python.exe
```

marimoを使うときは`--locked --group notebook`を付けます。
依存追加は第5節の手順で行い、`pip install`やセルからの自動インストールは使いません。

**`.cache/python/`にはPython本体があります。** `.cache`全体を削除・移動しないでください。
配置を変える場合はNotebookを保全し、Pythonと仮想環境を作り直します。

## 4. marimoを開く

VS Codeの`Python: Select Interpreter`で、第3節のPythonを選びます。
続いて、[marimo利用手順](marimo_HowToUse.md)の第2節でNotebookを開いてください。

起動には、第1節を実行した統合ターミナルを使います。
marimo拡張機能からの起動では、ターミナルの保存先設定を引き継がない場合があります。

## 5. 確認・ライブラリの変更

いずれも第1節を実行したターミナルで使います。環境の確認コマンドは次のとおりです。

```cmd
uv lock --check
uv run --locked --group notebook marimo --version
uv run --locked --group notebook python -c "import numpy, polars, pandas, scipy, torch, pymc, nutpie, arviz, numba, graphviz, openpyxl, matplotlib_fontja; print('Imports passed')"
```

ライブラリを変更するときは、次の順に進めます。

1. 2026-09-26 JST以前の安定版から、WindowsとPythonの対応を確認します。
2. `pyproject.toml`を変更し、採用版と理由をREADMEに記録します。
3. 次のコマンドでロックを更新し、環境へ反映します。
4. 影響する計算・描画・Notebookを新しいセッションで確認します。

```cmd
uv lock
uv lock --check
uv sync --locked --group notebook
```

`uv.lock`はuvが生成するため、手編集しません。
`exclude-newer`は2026-09-26 JSTの終了時点までの配布物に制限しています。
この基準日やPythonの版を変える場合は、環境の要件も見直してください。

## 6. 困ったとき

| 症状 | 確認すること |
| --- | --- |
| DLLエラー・別のPythonが動く | 第1節で`PYTHONHOME`・`PYTHONPATH`を解除し、`sys.executable`を確認 |
| Python 3.14.7が見つからない | 第2節の固定カタログと`UV_PYTHON_INSTALL_DIR`を確認 |
| `Python downloads are not allowed` | 明示的な導入時だけ`UV_PYTHON_DOWNLOADS=manual`にし、終了後は`never`へ戻す |
| lockと依存定義が合わない | 意図した変更か確認してから`uv lock`。`--frozen`では回避しない |
| Graphvizが動かない | 同じターミナルで`dot -V`を確認 |
| PyTensorのコンパイラ警告 | `g++ --version`と数値計算の結果を確認 |
| 旧API・CSV・Excelで失敗する | 元NotebookのAPIや取得先を確認。書籍16本の移植は未実施 |

## 参考資料

- [Python 3.14.7公式リリース](https://www.python.org/downloads/release/python-3147/)
- [uvのPython導入](https://docs.astral.sh/uv/guides/install-python/)
- [uvのロックと同期](https://docs.astral.sh/uv/concepts/projects/sync/)
- [uvの公開日時制限](https://docs.astral.sh/uv/concepts/resolution/#reproducible-resolutions)
