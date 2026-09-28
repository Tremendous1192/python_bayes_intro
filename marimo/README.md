# ベイズ推論「超」入門 — marimo環境（2026）

- 書籍『Pythonでスラスラわかる ベイズ推論「超」入門』を、VS Code・uv・marimoで学ぶための環境です。
- 現在のVS Codeウィンドウとログイン状態を使い、プロジェクト専用のPythonで実行します。
- 書籍の本編11本・参考4本、計15本の学習内容を`notebooks/`へ移植しています。
- 作成日：2026-09-26。更新日：2026-09-28。
- 登録処理統合の8 Taskの状態・実行結果は[統合検証](log/統合検証.md)を参照します。過去の移植記録と区別しています。
- データ準備の統合・分離とテスト移動の履歴は[追加の検証記録](log/データ統合とテスト移動.md)を参照します。

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
cd /d C:\dev\python_bayes_intro\marimo
call env.cmd
call env.cmd check
```

- 初回の依存同期後は、先に`call env.cmd configure`で登録します。
- Notebookのカーネルには`.venv\Scripts\python.exe`を選びます。
- 親フォルダを開いている場合は、対象の`marimo`を現在のワークスペースに追加します。
- 詳しい設定範囲・カーネル選択はuv利用手順の第4節を参照します。
- 毎回の依存同期・再登録・別ウィンドウ起動は不要です。

## フォルダの中身

- 対象：`/python_bayes_intro/marimo/`。データ準備は`data/`、Notebook用テストは`tests/`、その他の追加補助ファイル・検証記録は`log/`へ配置し、直下のファイル数を増やしません。

| ファイル・フォルダ | 役割 |
| --- | --- |
| [pyproject.toml](pyproject.toml) | Pythonの対応範囲と直接依存の原本 |
| [.python-version](.python-version) | 使用するPythonのパッチ版 |
| [uv.lock](uv.lock) | uvが生成した依存・配布物の固定情報 |
| [env.cmd](env.cmd) | 設定値の原本、ターミナル設定、登録・確認・解除・回帰テスト |
| [data/prepare_data.py](data/prepare_data.py) | 固定CSVの検査と、明示指定時だけの不足分取得 |
| [log/](log/) | データ検証コード・検証記録・画像 |
| [.vscode/settings.json](.vscode/settings.json) | Python候補・marimoの共有設定 |
| [examples/basic_usage.py](examples/basic_usage.py) | 平均・表・図・ボタンの8セルの例 |
| [examples/test_basic_usage.py](examples/test_basic_usage.py) | 例を新規プロセスで検証するNotebook専用コード |
| [notebooks/](notebooks/) | 書籍のmarimo Notebook、必要な共通処理 |
| [tests/](tests/) | Notebook用の章別・共通処理テスト14本 |
| [data/README.md](data/README.md) | データの取得元・固定コミット・再取得手順 |
| [HowToUse_uv.md](HowToUse_uv.md) | 初回導入・日常操作・更新・復元 |
| [HowToUse_marimo.md](HowToUse_marimo.md) | 拡張機能での作成・編集・実行・保存・画面確認 |
| [要件.md](要件/要件.md) | 環境・ガイド・移植の要件 |
| `.venv/` | プロジェクト専用のPython環境と生成された起動フック |
| `.cache/` | Python本体・キャッシュ・登録記録・復元用の退避 |

- `.venv/`・`.cache/`は[.gitignore](.gitignore)、一時検証用の`log/_work/`は[log/.gitignore](log/.gitignore)でGit管理から除外しています。
- **`.cache/python/`はPython本体です。`.cache`全体を削除・移動しないでください。**
- VS Codeのユーザー設定・認証情報・拡張機能のコピーは行いません。

## 設定が届く仕組み

- ターミナル：`call env.cmd`が、呼出元と子プロセスにuv用の設定を適用します。
- Notebook：選択した専用`.venv`が、Python起動時に登録済み設定を読みます。
- 登録処理・回帰テストは`env.cmd`の埋込Pythonに統合しています。データ準備は`data/prepare_data.py`を直接実行します。
- 起動フックは拡張機能から起動される専用Pythonへ設定を適用します。
- Notebook専用Pythonでは、個人領域参照もプロジェクト内へ切り替えます。
- ターミナルとVS Code本体の個人領域・ログイン状態は切り替えません。
- この.venvの全Python起動に設定が適用されます。他用途の環境と共用しません。
- 起動後の設定だけではUTF-8モードを変更できません。中継プロセスから子カーネルへの継承を検証します。
- `env.cmd`の埋込Pythonは`-S`で起動フックを読みません。登録の修復に使います。

| 生成物 | 原本・生成方法 |
| --- | --- |
| `.venv/Lib/site-packages/_bayes_runtime.py` | `env.cmd`の埋込Pythonから生成する設定適用コード |
| `.venv/Lib/site-packages/000_bayes_runtime.pth` | 同じ登録器から生成する読込入口 |
| `.cache/runtime-registration.json` | 生成物2本の所有確認用SHA-256 |

- 生成コマンド：`call env.cmd configure`。実行系：CPython 3.14.7。
- 生成物は手編集しません。同名の他者ファイル・変更済み生成物は上書きしません。
- 更新前の生成物は`.cache/`へハッシュ付きの名前で退避します。
- 設定変更・解除前にカーネルを停止します。複数ファイル全体の原子性は保証しません。
- 解除・再登録・破損時の扱いはuv利用手順の第7節を参照します。

## 採用環境

- Python・依存は2026-09-27に再確認しました。VS Codeと拡張機能の版は以前の記録値で、実画面での受入とは区別します。

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
- 指定14ライブラリに、5.3節のExcel保存・検証用`openpyxl`を加えています。
- 15件とも基準日以前の最新安定版と一致し、互換性による版の引下げはありません。公式PyPIの確認結果は[依存確認](log/依存確認.json)を参照します。
- 実環境のimport・版・実行系は[環境確認](log/environment-check.txt)に記録しています。

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
| [openpyxl](https://pypi.org/project/openpyxl/3.1.5/) | 3.1.5 | 5.3節のExcel保存・検証 |

- `marimo`・`numba`の直接指定は`notebook`グループです。
- PyTensorもNumbaに依存するため、`notebook`グループの省略ではNumbaを除外できません。
- 禁止ライブラリの`japanize_matplotlib`・`bambi`・`numpyro`はロックに含まれていません。
- 既存pandas処理を一律にPolarsへ置き換える方針ではありません。

## 再現性と依存の管理

- Python本体：Astral python-build-standalone、ビルド20260924。
- uv 0.11.7の内蔵カタログには対象版がないため、[固定した公式カタログ](https://raw.githubusercontent.com/astral-sh/uv/299a93de4b94e754f260c673d2de456afbdd4fb7/crates/uv-python/download-metadata.json)を使います。
- 取得コマンド・ハッシュ・復元方法：uv利用手順の第2節。
- ロックの原本：`pyproject.toml`。生成先：`uv.lock`。生成時のuv：0.11.7。
- 再生成する場合のコマンド：`uv lock --python C:/dev/python_bayes_intro/marimo/.cache/python/cpython-3.14.7-windows-x86_64-none/python.exe`。
- ロック：83パッケージ・1,025行。今回の導入数：80パッケージ。固定値とロックは変更していません。
- 取得元：公式PyPI。`exclude-newer = "2026-09-27T00:00:00+09:00"`で基準日の終了境界までに制限します。
- Windows AMD64用の安定版を使い、依存パッケージのソースビルドは許可しません。
- Pythonの通常の自動取得は禁止し、初回の明示的な導入時だけ許可します。
- `marimo.disableUvIntegration`で拡張機能の自動パッケージ導入案内を無効にします。
- NotebookのPythonは利用者が選び、依存追加はuv利用手順に従います。
- VS Code本体・拡張機能の更新は、Pythonのロックとは別に確認・記録します。

## 書籍Notebookの移植

- 対象は本編11本・参考4本、計15本です。原本のコードセルは計3,024行です。
- GPU版と`書籍評価.ipynb`は対象外です。元の`.ipynb`は変更しません。
- 長い章は学習単位で分割しています。15本は原本の件数で、移植先は20本のmarimo Notebookです。
- 各ファイルの先頭に対応する原本を記載しています。

| 原本の学習内容 | 移植先 |
| --- | --- |
| 第1章 確率分布 | [ch01_distributions.py](notebooks/ch01_distributions.py) |
| 第2章 各種分布 | [離散分布](notebooks/ch02_discrete.py)、[正規分布](notebooks/ch02_normal.py)、[一様・ベータ分布](notebooks/ch02_uniform_beta.py)、[区間](notebooks/ch02_intervals.py) |
| 第3章 ベイズ推論 | [ch03_bayes.py](notebooks/ch03_bayes.py) |
| 第4章 実習 | [尤度](notebooks/ch04_likelihood.py)、[事後分布](notebooks/ch04_posterior.py)、[事前分布の比較](notebooks/ch04_prior_comparison.py) |
| 5.1 データ分布 | [ch05_01_distribution.py](notebooks/ch05_01_distribution.py) |
| 5.2 線形回帰 | [ch05_02_regression.py](notebooks/ch05_02_regression.py) |
| 5.3 階層ベイズ | [ch05_03_hierarchical.py](notebooks/ch05_03_hierarchical.py) |
| 5.4 潜在変数 | [ch05_04_latent.py](notebooks/ch05_04_latent.py) |
| 6.1 ABテスト | [ch06_01_ab_test.py](notebooks/ch06_01_ab_test.py) |
| 6.2 回帰の効果検証 | [ch06_02_effects.py](notebooks/ch06_02_effects.py) |
| 6.3 IRT・ADVI | [ch06_03_irt.py](notebooks/ch06_03_irt.py) |
| 参考 第4章の図 | [sample_ch04_figure.py](notebooks/sample_ch04_figure.py) |
| 参考 潜在変数の簡略版 | [sample_ch05_04_simplified.py](notebooks/sample_ch05_04_simplified.py) |
| 参考 3クラス潜在変数 | [sample_three_class.py](notebooks/sample_three_class.py) |
| 参考 FAQ | [sample_latent_faq.py](notebooks/sample_latent_faq.py) |

- 共通処理：[mod_load_data.py](notebooks/mod_load_data.py)は固定データの検証・読込、[mod_sampling.py](notebooks/mod_sampling.py)はサンプラーと資源制限、[mod_plots.py](notebooks/mod_plots.py)は診断・モデル図の表示を担当します。
- モデル定義：[model_two_class.py](notebooks/model_two_class.py)、[model_three_class.py](notebooks/model_three_class.py)、[model_irt.py](notebooks/model_irt.py)を対応する教材から読み込みます。
- 文書・コメントは日本語、Notebook内の説明・画面出力・図の文字は英語です。元教材の日本語の集計キーは内部で保持します。
- `pm.ConstantData`を`pm.Data`へ、事前予測の`samples`を`draws`へ移行しています。
- PyMC 6のDataTreeとArviZ 1のPlotCollectionを使い、信用区間は元教材に合わせて94% HDIを明示しています。
- ベータ密度の正規化式、潜在成分を種名と断定する表示、IRTの集計順序と固定IDの混在を修正しています。
- 5.1節の5件推論は発散を避けるため`target_accept=0.95`に調整しています。モデル・観測は保持しています。
- 6.2節の縮小モデルは`target_accept=0.99`に調整し、本実行で発散1件が0件になることを確認しています。
- 収束しにくい比較例も教材として残しています。実行完了と収束成功は別に判断します。

### データ準備と実行

- 同梱のCSVを使い、Notebookの読込時にハッシュを検証します。既存CSVのCRLFはメモリ上でLFへ戻して照合します。
- 設定済みターミナルで`uv run --locked --offline --group notebook python data/prepare_data.py`を実行すると3本を検査し、`--download`を付けた場合だけ不足分を取得します。
- `--offline`はuvによる依存取得を禁止します。スクリプトの`--download`を付けた場合のCSV取得は別に許可されます。
- CSVの再取得が必要な場合だけ、[データ手順](data/README.md)の保守用コマンドを使います。
- ハッシュ不一致は自動上書きしません。Notebookも不一致を検出して停止します。
- 対応表の`.py`をVS Codeのmarimo拡張機能で開き、登録済み`.venv`を選びます。
- MCMC教材では`Sampling mode`を選び、`Run inference`を押します。変更だけで重い計算を始めない構成です。
- `Book run`が標準です。チェーン数・反復数・乱数種を明示し、CPUのPyMCサンプラーでモデルごとに順番に実行します。
- 通常は同時チェーン1・BLAS1、IRTだけ同時チェーン2とし、各プロセスのBLAS1・Numba上限2に制限します。
- IRTの本実行では4チェーン分のプロセスを作ります。標本を同時生成するのは2チェーンで、Numbaは合計最大4スレッドです。
- 通常はCバックエンド、IRTだけは全観測でCとの密度・勾配一致を検証したNumbaを使います。依存追加はありません。
- `Quick check`はdraw 100・tune 150以下の配線確認です。収束・精度の評価には使いません。
- IRTの`Book run`は全50,000回答、MCMCの後にADVI 20,000回です。短縮時も観測は間引きません。
- IRTのADVIは短縮時500回・100標本です。ADVI標本をMCMCのチェーンとして診断しません。
- IRTの能力値は原本に合わせ、MCMCは標本標準偏差、ADVIは母標準偏差で変換します。
- PDF・Excelは専用の保存ボタンから`.cache/exports/`へ新規保存します。既存ファイルは上書きしません。
- PyTensorのBLAS未リンク警告が出る環境です。依存やコンパイラを自動変更しません。

### 章別の再検証

- 次のコマンドは、新規プロセスで`Book run`の実行・数値・出力を検証します。長い推論を同時に起動しません。
- 登録処理統合時の結果・警告・未実施項目は[統合検証](log/統合検証.md)、以前の結果は[移植検証](log/移植検証.md)に記録しています。

```cmd
cd /d C:\dev\python_bayes_intro\marimo
call env.cmd
uv run --locked --offline --group notebook python tests/test_notebook_data.py
uv run --locked --offline --group notebook python log/test_data_integrity.py
uv run --locked --offline --group notebook python tests/test_ch01_ch03.py
uv run --locked --offline --group notebook python tests/test_ch02.py
uv run --locked --offline --group notebook python tests/test_ch04.py
uv run --locked --offline --group notebook python tests/test_ch05_01.py
uv run --locked --offline --group notebook python tests/test_ch05_02_03.py
uv run --locked --offline --group notebook python tests/test_ch05_04.py
uv run --locked --offline --group notebook python tests/test_latent_references.py
uv run --locked --offline --group notebook python tests/test_ch06_01.py
uv run --locked --offline --group notebook python tests/test_ch06_02.py
uv run --locked --offline --group notebook python tests/test_irt_backend.py
uv run --locked --offline --group notebook python tests/test_ch06_03.py book
uv run --locked --offline --group notebook python tests/test_notebook_plots.py
uv run --locked --offline --group notebook python tests/test_latent_views.py
```

- 各テストは自身の配置から`notebooks/`を読み込みます。手動の`PYTHONPATH`設定は不要です。

- `app.run(defs=...)`は新規セッションでの計算・表示生成の検査です。VS Code画面のクリック・保存・再起動の検査とは区別します。
- [PyMCのsample](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html)・[fit](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.fit.html)、[ArviZのforest plot](https://python.arviz.org/projects/plots/en/stable/api/generated/arviz_plots.plot_forest.html)を固定環境の実装と照合しています。

## 検証

- 同じウィンドウのCommand Promptで、各コマンドの成功を確認してから次へ進みます。

```cmd
cd /d C:\dev\python_bayes_intro\marimo
call env.cmd
call env.cmd check
uv lock --check
call env.cmd test
uv run --locked --group notebook marimo check examples/basic_usage.py
uv run --locked --group notebook python examples/test_basic_usage.py
```

- 登録器の試験は`log/_work/`の使い捨てデータで、他者ファイル・利用者の編集・操作ロックを保全することを確認します。
- 凡例の検証画像とIRTの測定記録は`log/validation/`に保存します。JIT等の実行キャッシュは`.cache/`です。
- 新規プロセス試験は、ターミナルの管理対象設定を外してから、起動フックによる適用を確認します。
- Notebookの3ケースは初期値10・変更後20・変更後のボタン押下相当です。
- `app.run(defs=...)`による模擬であり、画面操作ではありません。
- 画面確認はmarimo利用手順の第10節で行い、自動検証と区別します。

### 既存ウィンドウ対応の過去記録（2026-09-27）

- 以下のTask 1～7は以前の環境対応時の記録です。以前の移植Task 1～13と整理Taskは[移植検証](log/移植検証.md)で確認します。
- Task 1～7は順番に扱い、統合していません。
- Task 1：使い捨て仮想環境で起動フック、子へのUTF-8継承、一時保存先、親環境の保持を確認しました。
- Task 2：ターミナル設定、引数エラー、登録器未配置時の案内を確認しました。
- Task 3：登録、所有競合、繰返し、操作ロック、解除・復元、新規プロセス、uv同期後の登録保持を確認しました。
- Task 4：プロジェクト設定の構文と導入済みmarimo拡張機能0.18.1の設定名を照合しました。
- Task 5：操作ガイドを更新し、ローカルリンク25件・旧起動コマンドへの参照解消・差分の空白検査を確認しました。
- Task 6：以下の自動検証はPassedです。画面での受入確認はNot runで、Task全体は未完了です。
- Task 7：当時は削除を保留。その後のファイル整理Task 6で旧起動ファイルを削除しました。
- 当時の固定環境復元：カタログSHA-256一致、CPython 3.14.7、uv 0.11.7、83パッケージ解決・80パッケージ同期。
- GUI操作用ツールは利用できません。画面・ログイン状態の確認は利用者による結果を待ちます。
- 当時の退避先の記録：`.cache/same-window-backup-20260927/`。今回の旧2ファイルの退避先は[統合検証](log/統合検証.md)を参照します。
- 当時は採用版・依存ロック・要件書を維持し、全ライブラリの最新性を再調査していませんでした。

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
- この以前の作業では、書籍Notebookの移植・全実行とEXE・GPU検証は対象外でした。

</details>
