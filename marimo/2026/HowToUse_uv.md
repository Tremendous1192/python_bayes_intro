# uv利用手順

uvでPythonとライブラリを準備し、marimoのVS Code拡張機能で学習します。
初回は第1～4節、準備済みなら第5節から始めてください。
作成日：2026-09-26。対象：Windows 11 AMD64、通常のGIL付きCPython 3.14.7、uv 0.11.7。

## 1. 初回のターミナルを準備する

VS Codeの`Terminal: Select Default Profile`で`Command Prompt`を選び、新しい統合ターミナルを開きます。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
call env.cmd
uv --version
dot -V
g++ --version
```

`env.cmd`がエラーになった場合は、後続の操作を止めてください。
uv、Graphviz本体の`dot`、MinGWの`g++`は、既存の導入済みツールを使います。
見つからない場合は導入先を確認します。Pythonパッケージの`graphviz`だけでは`dot`は入りません。

`env.cmd`は、以前の長い環境変数設定をまとめたファイルです。設定の意味は日本語コメントで説明しています。
仮想環境やライブラリの導入は行わず、作業フォルダ・保存先・実行系・並列数をこのプロジェクトに揃えます。
通常のuv操作は、引き続き利用者が実行します。

| 保存先・設定 | 役割 |
| --- | --- |
| `.venv/` | このプロジェクトのPython環境 |
| `.cache/python/` | 固定したPython本体 |
| `.cache/uv/` | uvの取得キャッシュ |
| `.cache/tmp/`、`.cache/profile/` | 一時ファイル、プロセス内の個人領域参照 |
| `.cache/`内の各専用ディレクトリ | Matplotlib、Numba、PyTensor、marimoなどの設定・キャッシュ |
| `UV_PYTHON_DOWNLOADS=never` | 通常操作でのPython自動取得を禁止 |
| 数値計算の各スレッド上限：2 | CPU・メモリの過剰利用を抑える |

Windows全体のPATH、レジストリ、既存VS Codeの設定は変更しません。
このターミナルでは`USERPROFILE`・`APPDATA`などもプロジェクト内へ切り替わるため、学習用として使います。
設定を終えるときはターミナルを閉じてください。

## 2. Pythonを導入する（初回のみ）

`.cache\python\cpython-3.14.7-windows-x86_64-none\python.exe`がある場合は、第3節へ進みます。
第1節を実行したターミナルで、固定したAstral公式カタログを取得します。
uv 0.11.7の内蔵カタログには対象版がないため、このカタログを使用します。

```cmd
curl.exe --fail --location --connect-timeout 15 --max-time 120 --output .cache\python-downloads.json https://raw.githubusercontent.com/astral-sh/uv/299a93de4b94e754f260c673d2de456afbdd4fb7/crates/uv-python/download-metadata.json
certutil -hashfile .cache\python-downloads.json SHA256
```

取得に成功し、SHA-256が次の値と一致した場合だけ、次の導入コマンドへ進みます。
失敗や不一致の場合は止めてください。

```text
016746da52b4558782e2e71e621c682db786a0d07025a5b67bdac08c668f1aa2
```

```cmd
rem 明示的な導入時だけ取得を許可し、共通コマンドとレジストリへの登録は行わない。
set "UV_PYTHON_DOWNLOADS=manual"
uv python install 3.14.7 --no-bin --no-registry --python-downloads-json-url .cache\python-downloads.json
rem 導入の成功・失敗にかかわらず、通常の取得禁止設定へ戻す。
set "UV_PYTHON_DOWNLOADS=never"
.cache\python\cpython-3.14.7-windows-x86_64-none\python.exe --version
```

採用した配布物はAstral python-build-standaloneのビルド20260924、Windows x86_64、GIL有効版です。
配布物のSHA-256は`1493fc4185edf84bbd4305c15c5fbac2d4fcd4ddd7eb6273903669a5d3106178`です。
uvがカタログの値と照合します。

## 3. ライブラリを揃える（初回・復元時）

同じターミナルで、ロックを確認してから同期します。
`env.cmd`が使用するPythonと`.venv`の保存先を固定しているため、長いパスを毎回入力する必要はありません。

```cmd
uv lock --check
uv sync --locked --group notebook
uv run --locked --group notebook python -c "import sys; print(sys.executable); print(sys.version); assert sys.version_info[:3] == (3, 14, 7); assert sys._is_gil_enabled()"
```

各コマンドが成功し、次の実行ファイルとPython 3.14.7が表示されることを確認します。

```text
C:\dev\python_bayes_intro\marimo\2026\.venv\Scripts\python.exe
```

`--locked`は依存定義とロックの不一致をエラーにします。`--frozen`で回避しないでください。
`--group notebook`はmarimoなどのNotebook用依存を含める指定です。

## 4. 学習用のVS Codeを開く（初回設定）

```cmd
open_vscode.cmd
```

専用のVS Codeウィンドウが開きます。以後の編集・統合ターミナル操作は、このウィンドウで行います。
初回は、開いたフォルダが`C:\dev\python_bayes_intro\marimo\2026`であることを確認し、
自分の学習用フォルダとして信頼するかをVS Codeの確認画面で選択してください。
制限モードでは拡張機能によるNotebook実行を完了できません。

拡張機能一覧で、公式の`marimo-team.vscode-marimo`を確認します。
この手順の確認対象は**0.18.1**です。未導入なら専用ウィンドウの拡張機能画面で導入し、
必要に応じて`Install Specific Version...`から版を選択します。依存するMicrosoft Python拡張機能も必要です。
既存の通常ウィンドウに入っていても、専用の拡張機能保存先には存在しない場合があります。

続いて[marimo利用手順](marimo_HowToUse.md)で、NotebookのPythonを選択します。
画面操作の確認状況と採用した拡張機能の版は[README](README.md)に記録します。

<details>
<summary>専用ウィンドウにする理由と保存先</summary>

VS Codeは、起動済みプロセスの環境変数を再利用する場合があります。
通常ウィンドウの統合ターミナルで`set`しても、拡張機能側へは遡って反映されません。

`open_vscode.cmd`は`env.cmd`を呼び、`--user-data-dir`と`--extensions-dir`を指定して起動します。
これにより、拡張機能の子プロセスにも同じ保存先設定を継承させます。

| 保存先 | 内容 |
| --- | --- |
| `.cache/vscode-user/` | 専用ウィンドウの設定、履歴、ログ |
| `.cache/vscode-extensions/` | 専用ウィンドウの拡張機能 |
| `.vscode/settings.json` | 共有するプロジェクト設定 |

専用ユーザー設定は初回だけ生成し、以後は上書きしません。
初期設定ではVS Code本体と拡張機能の自動更新を止め、確認済みの組合せを維持します。
更新するときは版を記録し、Notebookの作成・実行・再開を確認してください。
信頼設定や認証情報はコピーしません。

同じ専用ウィンドウが開いている間は、その起動時の環境変数が使われます。
`env.cmd`を変更した場合は、実行と保存を終えて専用ウィンドウをすべて閉じ、再度起動します。

</details>

## 5. 毎日の操作

VS CodeのCommand Promptで、次の2行を実行します。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
open_vscode.cmd
```

開いた専用ウィンドウでNotebookを編集します。毎回の同期・再インストールは不要です。
その統合ターミナルは保存先設定を継承するので、例えば次の確認をそのまま実行できます。

```cmd
uv lock --check
uv run --locked --group notebook marimo --version
```

通常ウィンドウでuvだけを操作する場合は、第1節と同じ場所へ移動して`call env.cmd`を実行します。
この操作だけでは通常ウィンドウの拡張機能側の設定は変わりません。

## 6. 依存を変更する

1. 2026-09-26 JST以前の安定版から、Windows AMD64とPython 3.14への対応を確認します。
2. `pyproject.toml`を変更し、採用版と調整理由をREADMEに記録します。
3. 専用ウィンドウの統合ターミナルで、次のコマンドを順に実行します。
4. Notebookを新しいセッションで開き、影響する計算・描画を確認します。

```cmd
uv lock
uv lock --check
uv sync --locked --group notebook
```

`uv.lock`はuvが生成するため手編集しません。`pip install`やセルからの導入も使いません。
`exclude-newer`は2026-09-26 JSTの終了境界までの配布物に制限しています。
Pythonの版や基準日を変える場合は、要件・固定値・ガイド・実行確認を一緒に見直します。

## 7. 困ったとき・復元する

| 症状 | 対処 |
| --- | --- |
| `env.cmd`の配置エラー | 指定のフォルダに配置する。コピー先での使用は対象外 |
| Python本体が見つからない | 第1・2節を確認。別のPythonへ自動的に切り替えない |
| `.venv`未作成、版やmarimoが違う | 第3節の同期と実行ファイル確認を行う |
| Pythonの取得が禁止される | 初回の明示的な導入時だけ`manual`、終了後は`never` |
| `code.cmd`が見つからない | Windows版VS CodeのCLI導入先を確認する |
| 拡張機能がない・制限モードになる | 第4節の専用ウィンドウ、拡張機能、フォルダの信頼を確認する |
| ロック不一致 | 意図した変更を確認してから第6節へ進む |
| Graphviz・PyTensorのエラー | 同じターミナルで`dot -V`、`g++ --version`を確認する |

**`.cache/python/`には実行に必要なPython本体があります。`.cache`全体を削除しないでください。**
Notebookを保全し、環境だけの復元は第1～3節に従います。
既存の`.venv`やキャッシュを一括削除する手順は含めていません。

## 参考資料

- [Python 3.14.7公式リリース](https://www.python.org/downloads/release/python-3147/)
- [uvのPython導入](https://docs.astral.sh/uv/guides/install-python/)
- [uvのロックと同期](https://docs.astral.sh/uv/concepts/projects/sync/)
- [uvの公開日時制限](https://docs.astral.sh/uv/concepts/resolution/#reproducible-resolutions)
- [VS Codeの環境変数継承](https://code.visualstudio.com/docs/terminal/advanced#_environment-variables-between-vscode-instances)
- [公式marimo拡張機能](https://marketplace.visualstudio.com/items?itemName=marimo-team.vscode-marimo)
