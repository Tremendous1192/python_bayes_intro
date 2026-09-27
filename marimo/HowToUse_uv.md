# uv利用手順

- VS Codeの現在のウィンドウで、Pythonとライブラリを準備します。
- 作成日：2026-09-26。更新日：2026-09-27。
- 対象：Windows 11 AMD64、GIL付きCPython 3.14.7、uv 0.11.7。
- 初回は第1～4節、準備済みなら第5節から始めます。

## 0. 仮想環境(venv)を作成するフォルダをルートフォルダとする
1. `File` タブの `Open Folder` で仮想環境(venv)を作成するフォルダを開く。
   * `C:\dev\python_bayes_intro\marimo>`

## 1. 現在のターミナルを準備する

1. 現在のVS Codeで`Terminal: Select Default Profile`から`Command Prompt`を選びます。
2. 統合ターミナルで次を実行します。

```cmd
cd /d C:\dev\python_bayes_intro\marimo
call env.cmd
uv --version
dot -V
g++ --version
```

- 各コマンドの成功を確認してから次へ進みます。
- uv・Graphviz本体の`dot`・MinGWの`g++`は導入済みのものを使います。
- `env.cmd`は保存先などを設定します。Python・ライブラリを自動導入しません。
- ターミナルの`USERPROFILE`・`APPDATA`・`LOCALAPPDATA`を変更しません。
- 旧手順を実行済みのターミナルは閉じ、同じウィンドウで新しいCommand Promptを開いて移行します。
- VS Codeの既存ログイン・ユーザー設定・拡張機能を使います。
- このターミナルを閉じると、ターミナルへ適用した環境変数は終了します。

| 保存先・設定 | 役割 |
| --- | --- |
| `.venv/` | プロジェクト専用のPython環境 |
| `.cache/python/` | 固定したPython本体 |
| `.cache/uv/` | uvの取得キャッシュ |
| `.cache/tmp/` | 一時ファイル |
| `.cache/profile/` | 登録済みPythonプロセス内だけで使う個人領域参照 |
| `.cache/`内の各専用ディレクトリ | 描画・JIT・marimoなどのキャッシュ |
| `UV_PYTHON_DOWNLOADS=never` | 通常操作でのPython自動取得を禁止 |
| 数値計算の各スレッド上限：2 | CPU・メモリの過剰利用を抑制 |

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

## 3. ライブラリとNotebookの設定を揃える

1. 同じターミナルで次を順に実行します。
2. 失敗した場合は次へ進まず、表示された案内を確認します。

```cmd
uv lock --check
uv sync --locked --group notebook
call env.cmd configure
call env.cmd check
```

- `configure`は専用`.venv`に起動時設定を登録し、新規Pythonで反映を確認します。
- 登録処理と回帰テストは[env.cmd](env.cmd)に内蔵しています。登録処理は標準ライブラリだけを使います。
- 登録後の検証は`call env.cmd test`です。試験用ファイルは`log/_work/`に限定します。
- `check`は現在の`env.cmd`と登録内容の一致を確認します。
- `--locked`は依存定義とロックの不一致をエラーにします。`--frozen`で回避しません。
- `--group notebook`はmarimoなどのNotebook用依存を含めます。
- 初回・`.venv`再作成後・`env.cmd`変更後は`configure`を実行します。
- 設定変更・解除の前に、この`.venv`を使うNotebookカーネルを停止します。
- 設定値の原本は`env.cmd`です。生成物は手編集しません。

## 3.5. `.venv` のあるフォルダをルートフォルダにしていれば marimo 拡張機能でノートブックを編集する
1. marimoのインポートと起動ができるようになっていれば、`.py`ファイル右上のボタンからノートブックに切り替えることができる。
1. Jupyter Notebookと同様にKernelを選択して各セルを編集して、Runすればプログラムを実行できる。


## 4. 同じウィンドウでNotebookを準備する

1. 対象の`marimo`フォルダがワークスペースのルートとして表示されていることを確認します。
   - 親の`C:\dev`などを開いている場合は、`File: Add Folder to Workspace`で`C:\dev\python_bayes_intro\marimo`を現在のウィンドウへ追加します。
   - 子フォルダの`.vscode/settings.json`は、親だけを開いた状態では自動適用されません。
   - ワークスペースを保存する場合は`C:\dev\python_bayes_intro\marimo\log\`以下へ保存し、既存ファイルを上書きしません。
2. 現在のウィンドウで公式拡張機能`marimo-team.vscode-marimo`とMicrosoft Pythonの有効化を確認します。
   - 確認対象のmarimo拡張機能は0.18.1です。更新・追加導入は自動実行しません。
   - フォルダの信頼確認が出た場合は、自分の学習用フォルダとして確認します。
3. marimoの言語サーバーは既定の`wasm`を使います。
   - 以前に別設定へ変更していた場合は、ウィンドウの設定で`marimo.lsp.server`を確認します。
4. [marimo利用手順](HowToUse_marimo.md)に従い、Notebookのカーネルに次を選びます。

```text
C:\dev\python_bayes_intro\marimo\.venv\Scripts\python.exe
```

- 別ウィンドウ・専用ユーザーデータ領域の作成は不要です。
- ターミナルの`set`を拡張機能へ遡って反映する方式ではありません。
- 選択したPythonが起動フックを読み、Notebook用の設定を適用します。
- `PYTHONUTF8`は起動済みPythonのモードを変更しません。拡張機能の中継プロセスから起動される子カーネルへの継承を確認しています。
- `-S`は登録器の修復用です。Notebook実行には使いません。
- `PYTHONHOME`・`PYTHONPATH`で別Pythonを指定したウィンドウは対象外です。uv自身の正当な`PYTHONHOME`は検証して許可します。
- `env.cmd`変更後は再登録し、Notebookカーネルを起動し直します。
- 設定・環境の候補が更新されない場合だけ、保存後に同じウィンドウで`Developer: Reload Window`を使います。

## 5. 毎日の操作

1. uvや検査コマンドを使うターミナルで次を実行します。

```cmd
cd /d C:\dev\python_bayes_intro\marimo
call env.cmd
call env.cmd check
```

2. 同じウィンドウでNotebookを開き、登録済み`.venv`のカーネルを使います。

- 毎回の同期・再インストール・再登録は不要です。
- 登録後のNotebook実行は、ターミナルで`env.cmd`を呼んだかどうかに依存しません。
- Pythonの実体・版はNotebookの確認セルでも確認します。
- VS Code本体と拡張機能の更新はPythonのロックとは別管理です。

## 6. 依存を変更する

1. 2026-09-26 JST以前の安定版から、Windows AMD64・Python 3.14対応を確認します。
2. `pyproject.toml`を変更し、採用版・調整理由をREADMEに記録します。
3. カーネルを停止してから、設定済みターミナルで次を順に実行します。

```cmd
uv lock
uv lock --check
uv sync --locked --group notebook
call env.cmd configure
```

4. 新しいNotebookセッションで、影響する計算・描画を確認します。

- `uv.lock`はuvが生成します。手編集・`pip install`・セルからの導入は使いません。
- Pythonの版や基準日を変える場合は、要件・固定値・ガイド・実行確認を一緒に見直します。

## 7. 困ったとき・復元する

| 症状 | 対処 |
| --- | --- |
| 環境未作成・Python本体がない | 第1～3節を実行。別のPythonへ切り替えない |
| 設定が未登録・古い | カーネル停止後に`call env.cmd configure` |
| 登録ファイルが変更されている | 手編集を保全し、`.cache/runtime-registration.json`と退避内容を確認する |
| 登録操作のロックが残る | 他の設定操作が実行中でないか確認。存在だけを理由に削除しない |
| カーネルが起動時に失敗する | `call env.cmd check`で確認。登録器は`-S`で起動するためフック破損時も調査可能 |
| 保存先が期待と異なる | 選択したPython・設定の再登録・カーネル再起動を確認 |
| ロック不一致 | 意図した依存変更を確認して第6節へ進む |
| Graphviz・PyTensorのエラー | `dot -V`、`g++ --version`を確認 |

- 登録解除が必要な場合だけ、カーネルを停止して次を実行します。

```cmd
cd /d C:\dev\python_bayes_intro\marimo
call env.cmd unconfigure
```

- 自分の生成物だけを検査・退避して解除します。他者のファイルや変更済みファイルは削除しません。
- 解除時の退避先：`.cache/runtime-before-unconfigure/`。既存の異なる退避は上書きしません。
- 再登録は`call env.cmd configure`です。解除中はNotebookを実行しません。
- **`.cache/python/`にはPython本体があります。`.cache`全体を削除しないでください。**
- 環境だけを復元する場合は第1～3節に従います。Notebookと退避内容を保全します。

## 参考資料

- [Python 3.14の起動時設定](https://docs.python.org/3.14/library/site.html)
- [uvのPython導入](https://docs.astral.sh/uv/guides/install-python/)
- [uvのロックと同期](https://docs.astral.sh/uv/concepts/projects/sync/)
- [VS Codeの複数フォルダと設定](https://code.visualstudio.com/docs/editing/workspaces/multi-root-workspaces)
- [公式marimo拡張機能](https://marketplace.visualstudio.com/items?itemName=marimo-team.vscode-marimo)
