# ベイズ推論「超」入門 — marimo環境（2026）

書籍『Pythonでスラスラわかる ベイズ推論「超」入門』を、VS Code・uv・marimoで学ぶための環境です。
環境定義、起動用コマンド、操作ガイド、基本操作のNotebookを用意しています。
書籍の本編11本・参考5本の移植は、次の作業です。

## まずはここから

| 状況 | 操作 |
| --- | --- |
| 初めて準備する | [uv利用手順](uv_HowToUse.md)の第1～4節 |
| 準備済みで学習を始める | 下のコマンドで専用VS Codeを開く |
| Notebookの編集・実行を試す | [marimo利用手順](marimo_HowToUse.md) |
| 計算結果を再確認する | このREADMEの「検証」 |

VS Codeの統合ターミナルで`Command Prompt`を選び、実行します。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
open_vscode.cmd
```

初回は専用ウィンドウの拡張機能とフォルダの信頼を確認します。
Notebookのカーネルには`.venv\Scripts\python.exe`を選びます。
日常の起動時に、依存の同期や長い環境変数設定をやり直す必要はありません。

## フォルダの中身

対象：`/python_bayes_intro/marimo/2026/`

| ファイル・フォルダ | 役割 |
| --- | --- |
| [pyproject.toml](pyproject.toml) | Pythonの対応範囲と直接依存の原本 |
| [.python-version](.python-version) | 使用するPythonのパッチ版 |
| [uv.lock](uv.lock) | uvが生成した依存・配布物の固定情報 |
| [env.cmd](env.cmd) | 保存先、実行系、並列数の共通設定 |
| [open_vscode.cmd](open_vscode.cmd) | 設定を継承した専用VS Codeの起動 |
| [.vscode/settings.json](.vscode/settings.json) | Command Prompt、Python候補、marimoの共有設定 |
| [examples/basic_usage.py](examples/basic_usage.py) | 平均・表・図・ボタンの8セルの例 |
| [examples/test_basic_usage.py](examples/test_basic_usage.py) | 例を新規プロセスで検証するNotebook専用コード |
| [uv_HowToUse.md](uv_HowToUse.md) | 初回導入、日常操作、更新、復元 |
| [marimo_HowToUse.md](marimo_HowToUse.md) | 拡張機能での作成・編集・実行・保存 |
| [uv_要件.md](uv_要件.md) / [marimo_要件.md](marimo_要件.md) | 環境とガイドの要件 |
| `.venv/` | プロジェクト専用のPython環境 |
| `.cache/` | Python本体、各種キャッシュ、専用VS Codeの状態、検証用ファイル |

`.venv/`と`.cache/`は[既存の.gitignore](../.gitignore)でGit管理から除外しています。
**`.cache/python/`は実行に必要なPython本体です。`.cache`全体を削除・移動しないでください。**

`env.cmd`はターミナルと子プロセスの設定だけを変更します。
`open_vscode.cmd`は専用のユーザー設定・拡張機能保存先を使うため、通常のVS Codeウィンドウとは別の環境です。
初回生成した専用ユーザー設定は、その後の起動では上書きしません。
保存先の詳細と設定変更後の再起動方法は、uv利用手順の第4節にあります。

## 採用環境

2026-09-26に確認した組合せ：

- Windows 11 / AMD64、CPython **3.14.7**（64 bit・GIL有効）
- uv **0.11.7**、marimo **0.25.0**
- Python対応範囲：`>=3.14,<3.15`
- 実行環境：プロジェクト内の`.venv`、依存グループ：runtime + `notebook`
- VS Code **1.139.1**、marimo拡張機能 **0.18.1**
- Microsoft Python拡張機能 **2026.4.0**、Python Environments **1.38.0**
- Graphviz本体 **13.1.2**、MinGW g++ **16.1.0**

この環境はCPUで学ぶNotebook専用です。上位READMEのconda手順とは混用せず、EXE作成・GPU実行には使いません。
NumbaはPyTensorからも必要とされるため、`notebook`グループを省略しても除外できません。

指定14ライブラリとExcel読込用の`openpyxl`を採用しています。
今回、PyPIの公開日時と安定版を照合し、15件とも2026-09-26 JSTの終了境界以前の最新安定版と一致しました。
Windows AMD64用またはOS共通のwheelを確認し、固定版での同期と15件のimportも成功しています。
互換性のために版を引き下げたものはありません。

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

`marimo`と`numba`の直接指定は`notebook`グループです。
禁止ライブラリの`japanize_matplotlib`、`bambi`、`numpyro`はロックに含まれません。
既存pandas処理を、一律にPolarsへ置き換える方針ではありません。

## 再現性と依存の管理

Python本体はAstral python-build-standaloneのビルド20260924を使用しています。
uv 0.11.7の内蔵カタログに対象版がなかったため、
[固定した公式カタログ](https://raw.githubusercontent.com/astral-sh/uv/299a93de4b94e754f260c673d2de456afbdd4fb7/crates/uv-python/download-metadata.json)から導入しました。
取得コマンド・ハッシュ・復元方法はuv利用手順の第2節にあります。

ロックの原本は`pyproject.toml`、生成先は`uv.lock`です。
初回の生成コマンドはuv 0.11.7による
`uv lock --python C:/dev/python_bayes_intro/marimo/2026/.cache/python/cpython-3.14.7-windows-x86_64-none/python.exe`です。
ロックは83パッケージ・1,025行、今回の導入は80パッケージでした。
今回、依存定義・Python固定値・ロックは変更せず、`uv lock --check`で整合を確認しました。

公式PyPIを使用し、`exclude-newer = "2026-09-27T00:00:00+09:00"`で公開日時を制限しています。
対象はWindows AMD64用の安定版で、依存パッケージのソースビルドは許可していません。
通常のPython自動取得は禁止し、初回の明示的な導入時だけ許可します。

拡張機能側の自動パッケージ導入案内は、`marimo.disableUvIntegration`で無効にしています。
NotebookのPythonは利用者が選択し、依存追加はuv利用手順に従います。
VS Code本体や拡張機能の更新は、Pythonのロックとは別に確認・記録します。

## 書籍Notebookの移植

対象は本編11本と参考5本の計16本です。このフォルダへの移植は未実施です。
移植元のコードセル部分は計3,212行でした。説明・セル構造・検証を含む移植後の行数ではありません。

| 移植元 | 対象 |
| --- | --- |
| [notebooks/](../../notebooks/) | 第1～4章、5.1～5.4、6.1～6.3の11本 |
| [sample-notebooks/](../../sample-notebooks/) | 第4章の図、潜在変数モデル簡略版、3クラス版、FAQ、書籍評価の5本 |

`6_3_IRTによるテスト結果評価_GPU版.ipynb`は対象外です。
移植ではColab向け処理、IPythonマジック、旧フォントライブラリ、PyMC 6／ArviZ 1のAPI・結果形式に対応します。
Iris・CSV・Excelなどのデータは、取得元・保存先・失敗時の対処を移植時に決めます。
説明・コメントは日本語、画面出力や図の文字は英語にします。

## 検証

以下は今回の作業で実施した確認です。Taskは統合せず、個別に実施しています。

| 判定 | Task | 内容 |
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

3ケースは、初期値10・変更後20・変更後のボタン押下相当です。
変更と押下は`app.run(defs=...)`による模擬であり、画面操作の確認ではありません。
画面での確認手順はmarimo利用手順の第2・3・8節にあります。

専用VS Codeの統合ターミナルで再実行できます。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
uv lock --check
uv run --locked --group notebook marimo check examples/basic_usage.py
uv run --locked --group notebook python examples/test_basic_usage.py
```

通常のウィンドウで実行する場合は、先に`call env.cmd`を実行します。
検証用コードとNotebookはGit管理対象なので、キャッシュを共有せずに再現できます。

<details>
<summary>以前の環境準備時の検証記録</summary>

以前の記録にはExcel、Polars、SciPy、PyTorch、Graphviz、Numba、PyMC・nutpie、ArviZの小規模検証があります。
PyMC・nutpieは合成データで各2 chains・400 tune・600 draws、`cores=1`、`random_seed=42`を使用し、
事後平均は0.287488・0.278773、解析解`2/7`との差は0.06未満でした。
Numbaの二乗和338350は非JIT計算と絶対誤差1e-10以内で一致した記録があります。
これらの数値検証とブラウザーHTTP起動は、今回の再実行結果には含めません。

以前の検証器はGit管理外のキャッシュに置かれていたため、現在の再実行手順には使いません。
今回の基本操作の検証は`examples/`を原本にします。
小さな例の成功だけで、書籍全体のAPI互換性・収束・実行時間を保証することはできません。

</details>
