# ベイズ推論「超」入門 — marimo環境（2026）

- 書籍『Pythonでスラスラわかる ベイズ推論「超」入門』を、VS Code・uv・marimoで学ぶための環境です。
- 環境定義・起動コマンド・操作ガイド・基本操作のNotebookを用意しています。
- 書籍の本編11本・参考5本の移植は後続作業です。
- 作成日：2026-09-26。文章・参照先の更新日：2026-09-27。

## まずはここから

| 状況 | 操作 |
| --- | --- |
| 初めて準備する | [uv利用手順](HowToUse_uv.md)の第1～4節 |
| 準備済みで学習を始める | 下のコマンドで専用VS Codeを開く |
| Notebookの編集・実行を試す | [marimo利用手順](HowToUse_marimo.md) |
| 計算結果を再確認する | このREADMEの「検証」 |

- VS Codeの統合ターミナルで`Command Prompt`を選び、次を実行します。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
open_vscode.cmd
```

- 初回は専用ウィンドウの拡張機能・フォルダの信頼を確認します。
- Notebookのカーネルには`.venv\Scripts\python.exe`を選びます。
- 日常の起動時に、依存の同期や長い環境変数設定をやり直す必要はありません。

## フォルダの中身

- 対象：`/python_bayes_intro/marimo/2026/`。

| ファイル・フォルダ | 役割 |
| --- | --- |
| [pyproject.toml](pyproject.toml) | Pythonの対応範囲と直接依存の原本 |
| [.python-version](.python-version) | 使用するPythonのパッチ版 |
| [uv.lock](uv.lock) | uvが生成した依存・配布物の固定情報 |
| [env.cmd](env.cmd) | 保存先・実行系・並列数の共通設定 |
| [open_vscode.cmd](open_vscode.cmd) | 設定を継承した専用VS Codeの起動 |
| [.vscode/settings.json](.vscode/settings.json) | Command Prompt・Python候補・marimoの共有設定 |
| [examples/basic_usage.py](examples/basic_usage.py) | 平均・表・図・ボタンの8セルの例 |
| [examples/test_basic_usage.py](examples/test_basic_usage.py) | 例を新規プロセスで検証するNotebook専用コード |
| [HowToUse_uv.md](HowToUse_uv.md) | 初回導入・日常操作・更新・復元 |
| [HowToUse_marimo.md](HowToUse_marimo.md) | 拡張機能での作成・編集・実行・保存 |
| [uv_要件.md](要件/uv_要件.md) / [marimo_要件.md](要件/marimo_要件.md) | 環境とガイドの要件 |
| `.venv/` | プロジェクト専用のPython環境 |
| `.cache/` | Python本体・各種キャッシュ・専用VS Codeの状態・検証用ファイル |

- `.venv/`・`.cache/`は[既存の.gitignore](../.gitignore)でGit管理から除外しています。
- **`.cache/python/`は実行に必要なPython本体です。`.cache`全体を削除・移動しないでください。**
- `env.cmd`の設定はターミナルと子プロセスに有効です。
- `open_vscode.cmd`は専用のユーザー設定・拡張機能保存先を使い、初回生成した設定は上書きしません。
- 保存先・設定変更後の再起動方法は、uv利用手順の第4節にあります。

## 採用環境

- 以下は2026-09-26の既存記録にある組合せです。現在の導入・実行状況は「検証」を確認します。

| 対象 | 固定値・記録版 |
| --- | --- |
| OS・Python | Windows 11 AMD64、CPython 3.14.7、64 bit・GIL有効 |
| Python対応範囲 | `>=3.14,<3.15` |
| uv・marimo | uv 0.11.7、marimo 0.25.0 |
| 環境・依存グループ | プロジェクト内の`.venv`、runtime + `notebook` |
| VS Code・marimo拡張機能 | VS Code 1.139.1、marimo拡張機能0.18.1 |
| Python拡張機能 | Microsoft Python 2026.4.0、Python Environments 1.38.0 |
| 外部ツール | Graphviz 13.1.2、MinGW g++ 16.1.0 |

- CPUで学ぶNotebook専用環境です。上位READMEのconda手順とは混用せず、EXE作成・GPU実行には使いません。
- 指定14ライブラリに、参考Notebook「書籍評価」のExcel読込用`openpyxl`を加えています。
- 既存記録では、15件とも基準日以前の最新安定版と一致し、互換性による版の引下げはありません。
- 公開日時・wheel・importの過去の確認記録と、今回の再検証結果は区別します。

| ライブラリ | 採用版 | 用途 |
| --- | --- | --- |
| [marimo](https://pypi.org/project/marimo/0.25.0/) | 0.25.0 | Notebookフロントエンド |
| [matplotlib](https://pypi.org/project/matplotlib/3.11.2/) | 3.11.2 | 描画 |
| [matplotlib-fontja](https://pypi.org/project/matplotlib-fontja/1.1.0/) | 1.1.0 | 日本語フォント設定 |
| [seaborn](https://pypi.org/project/seaborn/0.13.2/) | 0.13.2 | 統計描画 |
| [pymc](https://pypi.org/project/pymc/6.3.2/) | 6.3.2 | 確率モデル・推論 |
| [nutpie](https://pypi.org/project/nutpie/0.16.11/) | 0.16.11 | CPU用NUTS |
| [arviz](https://pypi.org/project/arviz/1.3.0/) | 1.3.0 | 推論結果の集計・描画 |
| [numpy](https://pypi.org/project/numpy/2.5.3/) | 2.5.3 | 数値配列 |
| [polars](https://pypi.org/project/polars/1.44.2/) | 1.44.2 | 表処理 |
| [pandas](https://pypi.org/project/pandas/3.0.6/) | 3.0.6 | 表処理・入出力 |
| [scipy](https://pypi.org/project/scipy/1.18.1/) | 1.18.1 | 統計・科学計算 |
| [torch](https://pypi.org/project/torch/2.14.0/) | 2.14.0 | 第4章の自動微分 |
| [graphviz](https://pypi.org/project/graphviz/0.21/) | 0.21 | dot本体へのインターフェース |
| [numba](https://pypi.org/project/numba/0.67.0/) | 0.67.0 | Notebook専用の数値JIT |
| [openpyxl](https://pypi.org/project/openpyxl/3.1.5/) | 3.1.5 | Excel読込 |

- `marimo`・`numba`の直接指定は`notebook`グループです。
- PyTensorもNumbaに依存するため、`notebook`グループの省略ではNumbaを除外できません。
- 禁止ライブラリの`japanize_matplotlib`・`bambi`・`numpyro`はロックに含まれていません。
- 既存pandas処理を一律にPolarsへ置き換える方針ではありません。

## 再現性と依存の管理

- Python本体：Astral python-build-standalone、ビルド20260924。
- uv 0.11.7の内蔵カタログには対象版がないため、[固定した公式カタログ](https://raw.githubusercontent.com/astral-sh/uv/299a93de4b94e754f260c673d2de456afbdd4fb7/crates/uv-python/download-metadata.json)を使います。
- 取得コマンド・ハッシュ・復元方法：uv利用手順の第2節。
- ロックの原本：`pyproject.toml`。生成先：`uv.lock`。生成時のuv：0.11.7。
- 初回の生成コマンド：`uv lock --python C:/dev/python_bayes_intro/marimo/2026/.cache/python/cpython-3.14.7-windows-x86_64-none/python.exe`。
- ロック：83パッケージ・1,025行。既存記録の導入数：80パッケージ。
- 取得元：公式PyPI。`exclude-newer = "2026-09-27T00:00:00+09:00"`で基準日の終了境界までに制限します。
- Windows AMD64用の安定版を使い、依存パッケージのソースビルドは許可しません。
- Pythonの通常の自動取得は禁止し、初回の明示的な導入時だけ許可します。
- `marimo.disableUvIntegration`で拡張機能の自動パッケージ導入案内を無効にします。
- NotebookのPythonは利用者が選び、依存追加はuv利用手順に従います。
- VS Code本体・拡張機能の更新は、Pythonのロックとは別に確認・記録します。

## 書籍Notebookの移植

- 本編11本・参考5本、計16本が対象です。このフォルダへの移植は未実施です。
- 既存記録の移植元コードセルは計3,212行です。説明・セル構造・検証を含む移植後の行数ではありません。

| 移植元 | 対象 |
| --- | --- |
| [notebooks/](../../notebooks/) | 第1～4章、5.1～5.4、6.1～6.3の11本 |
| [sample-notebooks/](../../sample-notebooks/) | 第4章の図、潜在変数モデル簡略版、3クラス版、FAQ、書籍評価の5本 |

- `6_3_IRTによるテスト結果評価_GPU版.ipynb`は対象外です。
- 移植時にColab処理・IPythonマジック・旧フォントライブラリ・PyMC 6／ArviZ 1のAPIと結果形式に対応します。
- Iris・CSV・Excelなどの取得元・保存先・取得失敗時の対処は、移植時に決めます。
- 説明・コメントは日本語、画面出力・図の文字は英語にします。

## 検証

- 再実行は、環境を準備した専用VS Codeの統合ターミナルで行います。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
uv lock --check
uv run --locked --group notebook marimo check examples/basic_usage.py
uv run --locked --group notebook python examples/test_basic_usage.py
```

- 通常ウィンドウでは、先に`call env.cmd`を実行します。
- 検証コードはGit管理対象の`examples/`を原本とし、以前のキャッシュ内の検証器には依存しません。
- 3ケースは初期値10・変更後20・変更後のボタン押下相当です。`app.run(defs=...)`による模擬で、画面操作ではありません。
- 画面ではmarimo利用手順の第2・3・8節に従い、入力・保存・再読込・カーネル停止と再起動を確認します。

### 文章要件反映の検証（2026-09-27）

- Task 1～4は順番に実施し、統合していません。
- 対象はガイド3本・設定とコード5本・親の.gitignoreです。要件書のユーザー変更は保持しています。

| 判定 | Task | 今回の結果 |
| --- | --- | --- |
| Passed | 1 | 旧参照を修正。ガイド内のローカルリンク23件の参照先を確認 |
| Passed | 2 | 説明を箇条書き化。掲載したcmdコマンドの実行部分を保持 |
| Passed | 3 | Pythonの説明を除く構文木、TOML設定値、バッチ命令を照合。案内先以外の実行内容を保持 |
| Passed | 4 | カタログのSHA-256照合、CPython 3.14.7の復元、uv 0.11.7でのロック確認・80パッケージ同期 |
| Passed | 4 | .venvの実行ファイル、AMD64・GIL有効、直接依存15件の導入版と保存先設定を確認 |
| Passed | 4 | marimo静的検査、別々の新規プロセスによる3ケースの平均・表・図・停止条件 |
| Passed | 4 | 環境未準備時の正しい案内と終了コード1、準備後のVS Code起動コマンドの終了コード0 |
| Passed | 4 | UTF-8・CRLF、コード行数、最終差分、Python固定値・ロック・共有VS Code設定の保持 |
| Not run | 4 | VS Code画面での入力・保存・再読込・カーネル停止と再起動 |
| Not run | 4 | 全依存の基準日時点の最新性と、全15件のimportの再検証 |
| Not applicable | — | 書籍16本の移植・EXE・GPU検証 |

- Pythonと依存はuv利用手順の固定カタログ・ロックから復元しました。新しい版への更新は行っていません。
- 作成した環境・取得物・専用VS Codeの状態は、Git管理外の`.venv/`・`.cache/`に保存しています。
- 編集中のLF改行で`env.cmd`が一度失敗したため、元のCRLFへ戻し、正常終了と保存先設定を再確認しました。
- VS Code起動の終了コード0は、拡張機能や画面操作の成功を示すものではありません。
- GUI操作用ツールがないため、画面検証は未実施です。専用ウィンドウで拡張機能を準備し、marimo利用手順の第2・3・8節で確認します。
- 現在のコード行数：`pyproject.toml` 63、`env.cmd` 91、`open_vscode.cmd` 74、`basic_usage.py` 148、`test_basic_usage.py` 79。
- いずれも300行以下です。文書・コメントの検査用Python 3.13.13と、Notebook実行用の3.14.7は区別しています。

<details>
<summary>環境準備時の検証記録（既存文書の記録日：2026-09-26）</summary>

- 以下は変更前READMEに記載された結果です。2026-09-27の再実行結果ではありません。
- 旧Task番号は当時の作業単位を表します。

| 判定 | 旧Task | 内容 |
| --- | --- | --- |
| Passed | 1 | 基準日と15直接依存の照合、ロック確認、80パッケージの同期、全直接依存のimport |
| Passed | 1 | .venvの実行ファイル、CPython 3.14.7、AMD64、GIL有効を確認 |
| Passed | 2-A | 起動引数、保存先・スレッド設定の子プロセスへの継承、再起動時の既存設定保持 |
| Passed | 2-A | VS Code CLI未検出時の停止、子コマンドの失敗コード伝搬 |
| Not run | 2-B | 拡張機能のカーネル実行。専用プロファイルでフォルダの信頼確認が必要 |
| Passed | 3 | 掲載した準備・同期・版確認コマンド、文書内のローカルリンク |
| Passed | 4 | marimo静的検査、新規プロセス3ケースの平均・表・図・ボタン停止条件 |
| Passed | 4 | 保存した図の目視確認。軸・凡例は英語、平均線は5.5 |
| Not run | 4 | VS Code画面での入力操作、保存・再読込、カーネル停止・再起動 |
| Passed | 5 | ガイド間の整合、ローカルリンク、コード行数、構文、差分・要件書の保持 |
| Not applicable | — | 書籍16本の移植・全実行、EXE・GPU検証 |

- さらに以前の記録には、Excel・Polars・SciPy・PyTorch・Graphviz・Numba・PyMC／nutpie・ArviZの小規模検証があります。
- PyMC／nutpieの条件：合成データ、各2 chains・400 tune・600 draws、`cores=1`、`random_seed=42`。
- 記録された事後平均：0.287488／0.278773。解析解`2/7`との差は0.06未満。
- Numbaの二乗和：338350。非JIT計算と絶対誤差1e-10以内で一致した記録があります。
- これらの数値検証・ブラウザーHTTP起動の記録は、現在の再実行結果には含めません。
- 小さな例の成功だけで、書籍全体のAPI互換性・収束・実行時間は保証できません。

</details>
