# uv利用手順

- VS Code・uv・marimoで学習するため、Pythonとライブラリを準備します。
- 作成日：2026-09-26。文章・参照先の更新日：2026-09-27。
- 対象：Windows 11 AMD64、GIL付きCPython 3.14.7、uv 0.11.7。
- 初回は第1～4節、準備済みなら第5節から始めます。

## 1. 初回のターミナルを準備する

1. VS Codeの`Terminal: Select Default Profile`で`Command Prompt`を選びます。
2. 新しい統合ターミナルを開き、次を実行します。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
call env.cmd
uv --version
dot -V
g++ --version
```

- `env.cmd`がエラーになった場合は、後続の操作を止めます。
- uv・Graphviz本体の`dot`・MinGWの`g++`は、導入済みのものを使います。
- コマンドが見つからない場合は導入先を確認します。Pythonの`graphviz`パッケージには`dot`本体が含まれません。
- `env.cmd`は作業フォルダ・保存先・実行系・並列数を設定します。環境やライブラリの導入は、利用者がuvで行います。

| 保存先・設定 | 役割 |
| --- | --- |
| `.venv/` | このプロジェクトのPython環境 |
| `.cache/python/` | 固定したPython本体 |
| `.cache/uv/` | uvの取得キャッシュ |
| `.cache/tmp/`、`.cache/profile/` | 一時ファイル、プロセス内の個人領域参照 |
| `.cache/`内の各専用ディレクトリ | 描画・JIT・marimoなどの設定とキャッシュ |
| `UV_PYTHON_DOWNLOADS=never` | 通常操作でのPython自動取得を禁止 |
| 数値計算の各スレッド上限：2 | CPU・メモリの過剰利用を抑制 |

- 設定は呼出元ターミナルと子プロセスに有効です。Windows全体のPATH・レジストリ・通常のVS Code設定は変更しません。
- `USERPROFILE`・`APPDATA`などもプロジェクト内へ切り替わるため、このターミナルは学習用として使います。
- 設定を終えるときはターミナルを閉じます。

## 2. Pythonを導入する（初回のみ）

1. `.cache\python\cpython-3.14.7-windows-x86_64-none\python.exe`がある場合は、第3節へ進みます。
2. 第1節のターミナルで、固定したAstral公式カタログを取得します。
   - uv 0.11.7の内蔵カタログには対象版がないため、このカタログを使います。

```cmd
curl.exe --fail --location --connect-timeout 15 --max-time 120 --output .cache\python-downloads.json https://raw.githubusercontent.com/astral-sh/uv/299a93de4b94e754f260c673d2de456afbdd4fb7/crates/uv-python/download-metadata.json
certutil -hashfile .cache\python-downloads.json SHA256
```

3. 取得が成功し、SHA-256が次の値と一致したことを確認します。不一致・取得失敗の場合は止めます。

```text
016746da52b4558782e2e71e621c682db786a0d07025a5b67bdac08c668f1aa2
```

4. Pythonを導入し、取得禁止設定へ戻してから版を確認します。

```cmd
rem - 明示的な導入時だけ取得を許可し、共通コマンド・レジストリへの登録を避ける。
set "UV_PYTHON_DOWNLOADS=manual"
uv python install 3.14.7 --no-bin --no-registry --python-downloads-json-url .cache\python-downloads.json
rem - 導入の成功・失敗にかかわらず、通常の取得禁止設定へ戻す。
set "UV_PYTHON_DOWNLOADS=never"
.cache\python\cpython-3.14.7-windows-x86_64-none\python.exe --version
```

- 採用配布物：Astral python-build-standalone、ビルド20260924、Windows x86_64、GIL有効版。
- 配布物のSHA-256：`1493fc4185edf84bbd4305c15c5fbac2d4fcd4ddd7eb6273903669a5d3106178`。
- uvが配布物をカタログのハッシュと照合します。Python 3.14.7が確認できたら第3節へ進みます。

## 3. ライブラリを揃える（初回・復元時）

1. 同じターミナルで、ロック確認・同期・Python確認を順に実行します。
2. 各コマンドの成功を確認してから次へ進みます。

```cmd
uv lock --check
uv sync --locked --group notebook
uv run --locked --group notebook python -c "import sys; print(sys.executable); print(sys.version); assert sys.version_info[:3] == (3, 14, 7); assert sys._is_gil_enabled()"
```

- 実行ファイルが次の場所で、Python 3.14.7・GIL有効であることを確認します。

```text
C:\dev\python_bayes_intro\marimo\2026\.venv\Scripts\python.exe
```

- `env.cmd`がPythonと仮想環境の保存先を固定するため、毎回の長いパス指定は不要です。
- `--locked`は依存定義とロックの不一致をエラーにします。`--frozen`で回避しません。
- `--group notebook`はmarimoなどのNotebook用依存を含める指定です。

## 4. 学習用のVS Codeを開く（初回設定）

1. 次のコマンドで専用ウィンドウを開きます。

```cmd
open_vscode.cmd
```

2. 開いたフォルダが`C:\dev\python_bayes_intro\marimo\2026`であることを確認します。
3. VS Codeの確認画面で、自分の学習用フォルダとして信頼するかを選びます。
   - 制限モードでは拡張機能によるNotebook実行を完了できません。
4. 専用ウィンドウの拡張機能一覧で、公式の`marimo-team.vscode-marimo`を確認します。
   - 確認対象は0.18.1です。未導入なら導入し、必要に応じて`Install Specific Version...`で版を選びます。
   - 依存するMicrosoft Python拡張機能も必要です。
   - 通常ウィンドウに導入済みでも、専用の保存先には存在しない場合があります。
5. [marimo利用手順](HowToUse_marimo.md)でNotebookのPythonを選びます。
   - 採用した拡張機能の版・画面操作の検証結果は[README](README.md)に記録します。

<details>
<summary>専用ウィンドウの理由と保存先</summary>

- VS Codeは起動済みプロセスの環境変数を再利用する場合があります。
- 通常ウィンドウの統合ターミナルで`set`しても、拡張機能側には遡って反映されません。
- `open_vscode.cmd`は`env.cmd`を呼び、`--user-data-dir`・`--extensions-dir`を指定します。
- 拡張機能の子プロセスにも、同じ保存先・実行系の設定を継承させます。

| 保存先 | 内容 |
| --- | --- |
| `.cache/vscode-user/` | 専用ウィンドウの設定・履歴・ログ |
| `.cache/vscode-extensions/` | 専用ウィンドウの拡張機能 |
| `.vscode/settings.json` | 共有するプロジェクト設定 |

- 専用ユーザー設定は初回だけ生成し、以後は上書きしません。信頼設定・認証情報はコピーしません。
- 初期設定ではVS Code本体・拡張機能の自動更新を止めます。更新時は版を記録し、作成・実行・再開を確認します。
- `env.cmd`を変更した場合は処理と保存を終え、専用ウィンドウをすべて閉じてから起動し直します。

</details>

## 5. 毎日の操作

1. VS CodeのCommand Promptで次を実行します。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
open_vscode.cmd
```

2. 開いた専用ウィンドウでNotebookを編集します。
   - 毎回の同期・再インストールは不要です。
   - 統合ターミナルにも保存先設定が継承され、次の確認を実行できます。

```cmd
uv lock --check
uv run --locked --group notebook marimo --version
```

- 通常ウィンドウでuvだけを操作する場合は、第1節と同じ場所で`call env.cmd`を実行します。
- この操作だけでは、通常ウィンドウの拡張機能側の設定は変わりません。

## 6. 依存を変更する

1. 2026-09-26 JST以前の安定版から、Windows AMD64・Python 3.14への対応を確認します。
2. `pyproject.toml`を変更し、採用版・調整理由をREADMEに記録します。
3. 専用ウィンドウの統合ターミナルで次を順に実行します。

```cmd
uv lock
uv lock --check
uv sync --locked --group notebook
```

4. 新しいNotebookセッションで、影響する計算・描画を確認します。

- `uv.lock`はuvが生成します。手編集・`pip install`・セルからの導入は使いません。
- `exclude-newer`は2026-09-26 JSTの終了境界までの配布物に制限します。
- Pythonの版や基準日を変える場合は、要件・固定値・ガイド・実行確認を一緒に見直します。

## 7. 困ったとき・復元する

| 症状 | 対処 |
| --- | --- |
| `env.cmd`の配置エラー | 指定フォルダに配置する。コピー先での使用は対象外 |
| Python本体が見つからない | 第1・2節を確認。別のPythonへ自動的に切り替えない |
| `.venv`未作成、版やmarimoが違う | 第3節の同期と実行ファイル確認を行う |
| Pythonの取得が禁止される | 明示的な導入時だけ`manual`、終了後は`never` |
| `code.cmd`が見つからない | Windows版VS CodeのCLI導入先を確認する |
| 拡張機能がない・制限モードになる | 第4節の専用ウィンドウ、拡張機能、フォルダの信頼を確認する |
| ロック不一致 | 意図した変更を確認してから第6節へ進む |
| Graphviz・PyTensorのエラー | 同じターミナルで`dot -V`、`g++ --version`を確認する |

- **`.cache/python/`には実行に必要なPython本体があります。`.cache`全体を削除しないでください。**
- Notebookを保全し、環境だけの復元は第1～3節に従います。

## 参考資料

- [Python 3.14.7公式リリース](https://www.python.org/downloads/release/python-3147/)
- [uvのPython導入](https://docs.astral.sh/uv/guides/install-python/)
- [uvのロックと同期](https://docs.astral.sh/uv/concepts/projects/sync/)
- [uvの公開日時制限](https://docs.astral.sh/uv/concepts/resolution/#reproducible-resolutions)
- [VS Codeの環境変数継承](https://code.visualstudio.com/docs/terminal/advanced#_environment-variables-between-vscode-instances)
- [公式marimo拡張機能](https://marketplace.visualstudio.com/items?itemName=marimo-team.vscode-marimo)
