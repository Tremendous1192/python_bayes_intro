# ベイズ推論「超」入門 — marimo環境（2026）

- 書籍『Pythonでスラスラわかる ベイズ推論「超」入門』を、VS Code・uv・marimoで学ぶための環境です。
- 現在のVS Codeウィンドウとログイン状態を使い、プロジェクト専用のPythonで実行します。
- 書籍の本編11本・参考5本の移植は後続作業です。
- 作成日：2026-09-26。更新日：2026-09-27。

## まずはここから

| 状況 | 操作 |
| --- | --- |
| 初めて準備する | [uv利用手順](HowToUse_uv.md)の第1～4節 |
| 準備済みで学習を始める | 同じウィンドウでNotebookを開き、登録済み.venvを選ぶ |
| Notebookの編集・実行を試す | [marimo利用手順](HowToUse_marimo.md) |
| 設定を変更した・.venvを再作成した | カーネル停止後に`call env.cmd configure` |
| 登録内容を確認する | 設定済みターミナルで`call env.cmd check` |

- uvや検査を使う場合は、現在のウィンドウのCommand Promptで次を実行します。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
call env.cmd
call env.cmd check
```

- 初回の依存同期後は、先に`call env.cmd configure`で登録します。
- Notebookのカーネルには`.venv\Scripts\python.exe`を選びます。
- 親フォルダを開いている場合は、対象の`2026`を現在のワークスペースに追加します。
- 詳しい設定範囲・カーネル選択はuv利用手順の第4節を参照します。
- 毎回の依存同期・再登録・別ウィンドウ起動は不要です。

## フォルダの中身

- 対象：`/python_bayes_intro/marimo/2026/`。

| ファイル・フォルダ | 役割 |
| --- | --- |
| [pyproject.toml](pyproject.toml) | Pythonの対応範囲と直接依存の原本 |
| [.python-version](.python-version) | 使用するPythonのパッチ版 |
| [uv.lock](uv.lock) | uvが生成した依存・配布物の固定情報 |
| [env.cmd](env.cmd) | 設定値の原本、ターミナル設定、登録・確認・解除の入口 |
| [configure_runtime.py](configure_runtime.py) | 専用.venvへの起動設定の生成・確認・解除 |
| [test_configure_runtime.py](test_configure_runtime.py) | 所有判定・繰返し・解除・新規プロセスの検証 |
| [.vscode/settings.json](.vscode/settings.json) | Python候補・marimoの共有設定 |
| [examples/basic_usage.py](examples/basic_usage.py) | 平均・表・図・ボタンの8セルの例 |
| [examples/test_basic_usage.py](examples/test_basic_usage.py) | 例を新規プロセスで検証するNotebook専用コード |
| [HowToUse_uv.md](HowToUse_uv.md) | 初回導入・日常操作・更新・復元 |
| [HowToUse_marimo.md](HowToUse_marimo.md) | 拡張機能での作成・編集・実行・保存・画面確認 |
| [uv_要件.md](要件/uv_要件.md) / [marimo_要件.md](要件/marimo_要件.md) | 環境とガイドの要件 |
| `.venv/` | プロジェクト専用のPython環境と生成された起動フック |
| `.cache/` | Python本体・キャッシュ・登録記録・復元用の退避 |

- `.venv/`・`.cache/`は[既存の.gitignore](../.gitignore)でGit管理から除外しています。
- **`.cache/python/`はPython本体です。`.cache`全体を削除・移動しないでください。**
- VS Codeのユーザー設定・認証情報・拡張機能のコピーは行いません。

## 設定が届く仕組み

- ターミナル：`call env.cmd`が、呼出元と子プロセスにuv用の設定を適用します。
- Notebook：選択した専用`.venv`が、Python起動時に登録済み設定を読みます。
- `configure_runtime.py`は、親のVS Codeへ環境変数を遡って書き込むための処理ではありません。
- 補助コードの追加理由は、拡張機能から起動されるPythonに影響範囲を限定するためです。
- Notebook専用Pythonでは、個人領域参照もプロジェクト内へ切り替えます。
- ターミナルとVS Code本体の個人領域・ログイン状態は切り替えません。
- この.venvの全Python起動に設定が適用されます。他用途の環境と共用しません。
- 起動後の設定だけではUTF-8モードを変更できません。中継プロセスから子カーネルへの継承を検証します。
- `-S`は起動フックを無効にするため、修復用の登録器にだけ使用します。

| 生成物 | 原本・生成方法 |
| --- | --- |
| `.venv/Lib/site-packages/_bayes_runtime.py` | `configure_runtime.py`と`env.cmd`から生成する設定適用コード |
| `.venv/Lib/site-packages/000_bayes_runtime.pth` | 同じ登録器から生成する読込入口 |
| `.cache/runtime-registration.json` | 生成物2本の所有確認用SHA-256 |

- 生成コマンド：`call env.cmd configure`。実行系：CPython 3.14.7。
- 生成物は手編集しません。同名の他者ファイル・変更済み生成物は上書きしません。
- 更新前の生成物は`.cache/`へハッシュ付きの名前で退避します。
- 設定変更・解除前にカーネルを停止します。複数ファイル全体の原子性は保証しません。
- 解除・再登録・破損時の扱いはuv利用手順の第7節を参照します。

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

- 同じウィンドウのCommand Promptで、各コマンドの成功を確認してから次へ進みます。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
call env.cmd
call env.cmd check
uv lock --check
uv run --locked --group notebook python test_configure_runtime.py
uv run --locked --group notebook marimo check examples/basic_usage.py
uv run --locked --group notebook python examples/test_basic_usage.py
```

- 登録器の試験は`.cache/`の使い捨てデータで、他者ファイル・利用者の編集・操作ロックを保全することを確認します。
- 新規プロセス試験は、ターミナルの管理対象設定を外してから、起動フックによる適用を確認します。
- Notebookの3ケースは初期値10・変更後20・変更後のボタン押下相当です。
- `app.run(defs=...)`による模擬であり、画面操作ではありません。
- 画面確認はmarimo利用手順の第10節で行い、自動検証と区別します。

### 既存ウィンドウ対応（2026-09-27）

- Task 1～7は順番に扱い、統合していません。
- Task 1：使い捨て仮想環境で起動フック、子へのUTF-8継承、一時保存先、親環境の保持を確認しました。
- Task 2：ターミナル設定、引数エラー、登録器未配置時の案内を確認しました。
- Task 3：登録、所有競合、繰返し、操作ロック、解除・復元、新規プロセス、uv同期後の登録保持を確認しました。
- Task 4：プロジェクト設定の構文と導入済みmarimo拡張機能0.18.1の設定名を照合しました。
- Task 5：操作ガイドを更新し、ローカルリンク25件・旧起動コマンドへの参照解消・差分の空白検査を確認しました。
- Task 6：以下の自動検証はPassedです。画面での受入確認はNot runで、Task全体は未完了です。
- Task 7：Task 6完了前のため、旧起動ファイルの削除は保留しています。
- 今回の固定環境復元：カタログSHA-256一致、CPython 3.14.7、uv 0.11.7、83パッケージ解決・80パッケージ同期。
- GUI操作用ツールは利用できません。画面・ログイン状態の確認は利用者による結果を待ちます。
- 元ファイルの退避：`.cache/same-window-backup-20260927/`。ファイル別SHA-256は同フォルダの`manifest.json`にあります。
- 既存の採用版・依存ロック・要件書は維持しています。全ライブラリの最新性は再調査していません。

| 判定 | Task 6の確認項目 |
| --- | --- |
| Passed | 固定CPython 3.14.7、AMD64・64 bit・GIL、実際の.venv実行ファイル |
| Passed | 登録内容とenv.cmdの一致、新規Pythonへの設定適用 |
| Passed | 他者ファイル・利用者編集・操作ロックの保全、登録の繰返し |
| Passed | 解除・復元、.venv再作成相当、部分破損・不正な所有記録の拒否 |
| Passed | 未作成環境・引数違い・別Pythonの拒否、暗黙の導入がないこと |
| Passed | marimo静的検査、初期値・入力変更・ボタン押下相当の3ケース |
| Not run | VS Code画面での入力・保存・再読込・停止・再起動と、追加ログイン要求がないこと |

- .venv再作成後に旧登録記録だけが残るケースは、Task 6で修正し回帰確認しました。
- 外部環境の継承試験では、実際のターミナルやVS Codeの環境変数は変更していません。
- 試験の保存先確認は設定値と既知の一時出力を対象とし、OS全体の書込監視を実施したものではありません。

<details>
<summary>以前の検証記録の扱い</summary>

- 2026-09-26の環境準備と、2026-09-27の文章整理時の記録は、退避したREADMEに保全しています。
- 以前の成功記録を、今回の変更後に再実行した結果として扱いません。
- 以前の記録でも、VS Code画面の入力・保存・再読込・カーネル操作は未実施でした。
- 書籍16本の移植・全実行、EXE・GPU検証は今回も対象外です。

</details>
