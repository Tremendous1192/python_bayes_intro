# ベイズ推論「超」入門 — marimo環境（2026）

作成日・検証日: 2026-09-26 JST。
書籍『Pythonでスラスラわかる ベイズ推論「超」入門』のコードを、
VS Code・uv・marimoで学習するためのNotebook専用環境です。
[要件書](uv_要件.md)に基づく環境準備を実装しました。
本編11本・参考5本の書換えは後続作業であり、このフォルダに移植済みNotebookはまだありません。

最初に[uv利用手順](uv_HowToUse.md)の環境変数とPython導入手順を実行してください。
環境設定を省略した裸の`uv run`では、保存先の制限や別Python環境の継承防止を保証できません。
上位の`/python_bayes_intro/marimo/README.md`にあるconda手順は、このuv環境では使用しません。

## フォルダの役割

対象ルートは`/python_bayes_intro/marimo/2026/`です。

| ファイル・ディレクトリ | 役割 | Git管理 |
| --- | --- | --- |
| [pyproject.toml](pyproject.toml) | Python対応範囲、直接依存、Notebookグループ、公開日時制限の原本 | 対象 |
| [.python-version](.python-version) | 実行するCPythonを3.14.7に固定 | 対象 |
| [uv.lock](uv.lock) | uvが生成した間接依存・配布物ハッシュを含む解決結果 | 対象 |
| [uv_HowToUse.md](uv_HowToUse.md) | VS Codeのcmd.exeで行う導入・実行・更新・切分け | 対象 |
| [uv_要件.md](uv_要件.md) | ユーザー管理の要件書。今回の実装では変更していない | 対象 |
| `.venv/` | このプロジェクトだけに使う仮想環境 | 除外 |
| `.cache/python/` | uvが導入したPython本体。`.venv`から参照される | 除外 |
| `.cache/`のその他 | uv・描画・JITのキャッシュ、一時ファイル、marimoの設定・ログ | 除外 |
| `.cache/validation/` | 今回の使い捨て検証Notebook、検証コード、描画出力 | 除外 |

除外は既存の`/python_bayes_intro/marimo/.gitignore`を使用しています。
`.cache/python`を使用中に削除すると仮想環境が起動できなくなるため、単なる描画キャッシュと区別します。
ユーザーの未コミット変更、移植元Notebook、上位README、保護対象のAGENTS.mdは変更していません。

## 実行環境と再現性

| 項目 | 採用・確認結果 |
| --- | --- |
| OS / アーキテクチャ | Windows 11 / AMD64 |
| Python対応範囲 | `>=3.14,<3.15`。lockの`==3.14.*`は同じ対応範囲 |
| 実測Python | CPython 3.14.7、GIL有効、64 bit |
| Python配布 | Astral python-build-standalone、ビルド20260924 |
| 実行ファイル | `C:\dev\python_bayes_intro\marimo\2026\.venv\Scripts\python.exe` |
| uv | 0.11.7。既存のものを使用し、更新していない |
| ロック | uv 0.11.7による生成、1,025行、83パッケージ（プロジェクト自身・条件付き依存を含む） |
| 実際の導入 | runtime + `notebook`グループ、80パッケージ |
| Graphviz本体 | 既存の13.1.2。Pythonパッケージの版とは別 |
| C++コンパイラ | 既存のMinGW g++ 16.1.0。追加導入していない |

ロックの生成原本は`pyproject.toml`、出力先は同じフォルダの`uv.lock`です。
初回は`uv lock --python C:/dev/python_bayes_intro/marimo/2026/.cache/python/cpython-3.14.7-windows-x86_64-none/python.exe`
を実行し、生成後の`uv lock --check`も成功しました。

Python 3.14.7は2026-08-05公開の安定版です。
[Python.orgのリリース情報](https://www.python.org/downloads/release/python-3147/)で確認しました。
既存uvの内蔵カタログには対象版がなかったため、
[2026-09-24のAstral公式カタログ](https://raw.githubusercontent.com/astral-sh/uv/299a93de4b94e754f260c673d2de456afbdd4fb7/crates/uv-python/download-metadata.json)を固定して使用しました。
カタログ・Python配布物のハッシュと導入コマンドは利用手順に記載しています。

パッケージは公式PyPIから取得し、`exclude-newer = "2026-09-27T00:00:00+09:00"`で
2026-09-26 JSTの終了境界を設定しています。各配布ファイルのアップロード日時が対象です。
プレリリースとソース配布物のビルドは許可せず、Windows AMD64向けに解決しました。

通常操作ではPythonの自動取得を禁止し、明示的な初回導入時だけ許可します。
Pythonの共通実行ファイル・Windowsレジストリへの登録と、システムPATHの変更はしていません。
marimoのWindows用履歴保存先は、子プロセスだけの作業用プロファイルに隔離します。

## 直接依存の採用バージョン

公開日はPyPIメタデータの初回アップロード日（UTC）です。確認日は2026-09-26 JSTです。
指定14ライブラリと、Excel読込に必要な`openpyxl`を採用しました。
直接依存は調査時点の最新安定版からの引下げなしで解決・検証できました。

| ライブラリ | 採用版 | 公開日（UTC） | 役割 |
| --- | --- | --- | --- |
| [marimo](https://pypi.org/project/marimo/0.25.0/) | 0.25.0 | 2026-09-23 | Notebookフロントエンド |
| [matplotlib](https://pypi.org/project/matplotlib/3.11.2/) | 3.11.2 | 2026-09-11 | 描画 |
| [matplotlib-fontja](https://pypi.org/project/matplotlib-fontja/1.1.0/) | 1.1.0 | 2025-04-24 | 日本語フォント設定 |
| [seaborn](https://pypi.org/project/seaborn/0.13.2/) | 0.13.2 | 2024-01-25 | 統計描画 |
| [pymc](https://pypi.org/project/pymc/6.3.2/) | 6.3.2 | 2026-09-08 | 確率モデルと推論 |
| [nutpie](https://pypi.org/project/nutpie/0.16.11/) | 0.16.11 | 2026-06-30 | CPUでのNUTS |
| [arviz](https://pypi.org/project/arviz/1.3.0/) | 1.3.0 | 2026-08-11 | 推論結果の集計・描画 |
| [numpy](https://pypi.org/project/numpy/2.5.3/) | 2.5.3 | 2026-09-06 | 数値配列 |
| [polars](https://pypi.org/project/polars/1.44.2/) | 1.44.2 | 2026-09-09 | 型を明示した表処理 |
| [pandas](https://pypi.org/project/pandas/3.0.6/) | 3.0.6 | 2026-09-17 | 既存表処理とデータ入出力 |
| [scipy](https://pypi.org/project/scipy/1.18.1/) | 1.18.1 | 2026-08-21 | 統計・科学計算 |
| [torch](https://pypi.org/project/torch/2.14.0/) | 2.14.0 | 2026-09-02 | 第4章の自動微分、CPUで確認 |
| [graphviz](https://pypi.org/project/graphviz/0.21/) | 0.21 | 2025-06-15 | dot本体へのPythonインターフェース |
| [numba](https://pypi.org/project/numba/0.67.0/) | 0.67.0 | 2026-08-11 | Notebook専用の数値JIT |
| [openpyxl](https://pypi.org/project/openpyxl/3.1.5/) | 3.1.5 | 2024-06-28 | 参考Notebook「書籍評価」のxlsx読込に必要な追加依存 |

`marimo`と`numba`の直接指定は、明示的に選択する`notebook`グループに置いています。
ただしPyMCの間接依存であるPyTensor 3.3.2もNumbaを要求します。
グループを省略してもNumbaを排除できる構成ではなく、この環境全体がNotebook専用です。
EXEの実行・テスト・ビルド環境として使用しません。
Numba 0.67.0、llvmlite 0.49.0、NumPy 2.5.3の組合せを実行確認しました。

`japanize_matplotlib`、`bambi`、`numpyro`は直接依存・ロック・導入環境の対象に含めません。
Linux向けCUDA依存やGPU版Notebookの依存も導入していません。

## 将来の移植対象

本編は`/python_bayes_intro/notebooks/`の次の11本です。

| 範囲 | 移植元 |
| --- | --- |
| 第1章 | [1章_確率分布とは.ipynb](../../notebooks/1章_確率分布とは.ipynb) |
| 第2章 | [2章_よく利用される確率分布.ipynb](../../notebooks/2章_よく利用される確率分布.ipynb) |
| 第3章 | [3章_ベイズ推論とは.ipynb](../../notebooks/3章_ベイズ推論とは.ipynb) |
| 第4章 | [4章_はじめてのベイズ推論実習.ipynb](../../notebooks/4章_はじめてのベイズ推論実習.ipynb) |
| 5.1 | [5_1_データ分布のベイズ推論.ipynb](../../notebooks/5_1_データ分布のベイズ推論.ipynb) |
| 5.2 | [5_2_線形回帰のベイズ推論.ipynb](../../notebooks/5_2_線形回帰のベイズ推論.ipynb) |
| 5.3 | [5_3_階層ベイズモデル.ipynb](../../notebooks/5_3_階層ベイズモデル.ipynb) |
| 5.4 | [5_4_潜在変数モデル.ipynb](../../notebooks/5_4_潜在変数モデル.ipynb) |
| 6.1 | [6_1_ABテスト効果検証.ipynb](../../notebooks/6_1_ABテスト効果検証.ipynb) |
| 6.2 | [6_2_ベイズ回帰モデルによる効果検証.ipynb](../../notebooks/6_2_ベイズ回帰モデルによる効果検証.ipynb) |
| 6.3 | [6_3_IRTによるテスト結果評価.ipynb](../../notebooks/6_3_IRTによるテスト結果評価.ipynb) |

参考は`/python_bayes_intro/sample-notebooks/`の次の5本です。

| 範囲 | 移植元 |
| --- | --- |
| 第4章の図 | [4章_図4_4用.ipynb](../../sample-notebooks/4章_図4_4用.ipynb) |
| 潜在変数モデル簡略版 | [5_4_潜在変数モデル_簡略版.ipynb](../../sample-notebooks/5_4_潜在変数モデル_簡略版.ipynb) |
| 3クラス潜在変数モデル | [A_3クラス潜在変数モデル.ipynb](../../sample-notebooks/A_3クラス潜在変数モデル.ipynb) |
| FAQ | [FAQ_潜在変数モデル.ipynb](../../sample-notebooks/FAQ_潜在変数モデル.ipynb) |
| 書籍評価 | [書籍評価.ipynb](../../sample-notebooks/書籍評価.ipynb) |

`/python_bayes_intro/sample-notebooks/6_3_IRTによるテスト結果評価_GPU版.ipynb`は対象外です。
対象16本のコードセルは計3,212行で、Markdown・保存済み出力・JSON構造を含まない調査値です。
この値は、コメントやmarimoのセル定義を追加した後のファイル行数ではありません。

移植時には、旧`japanize_matplotlib`、Colabのファイル取得処理、IPythonのマジックや表示処理を置換します。
PyMC 6／ArviZ 1では推論結果のDataTreeや描画APIに対応し、単純な機械変換だけで完了としません。
[PyMCの現行例](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html)と
[ArviZの破壊的変更の告知](https://github.com/arviz-devs/arviz/issues/2548)を参照してください。
日本語コメント・説明を付け、Notebook上のメッセージや図の文字は英語にします。

Iris、`test_scores.csv`、IRTのCSV、書籍評価のExcelには外部取得処理があります。
後続の移植で取得元・保存先・失敗時の処理を明示し、ダウンロード先をプロジェクト内に限定します。
今回、実データの取得・同梱や完全オフライン対応は行っていません。

## 検証結果

使い捨てのNotebookと合成データを使用し、新しいプロセスから確認しました。
サンプリングは各2 chains、400 tune、600 draws、`cores=1`、`random_seed=42`です。
BLAS・OpenMP・Numba・Polarsは各2スレッドを上限として実行しました。

| 判定 | 確認内容 | 結果 |
| --- | --- | --- |
| Passed | `uv lock --check` | manifestとロックが整合 |
| Passed | `uv sync --locked --group notebook` | 指定`.venv`へ80パッケージ導入 |
| Passed | 実行系・15直接依存 | CPython 3.14.7、GIL有効、実行パスと全固定版が一致 |
| Passed | `marimo check`とNotebookの新規プロセス実行 | セルの依存関係確認と実行成功 |
| Passed | Excel・Polars・SciPy・PyTorch | 合成Excelを読込、整数表の合計6、正規CDF(0)=0.5、自動微分の結果4 |
| Passed | Matplotlib・Seaborn・フォント・Graphviz | PNG/SVG生成、日本語フォントの解決、事後分布画像の目視確認 |
| Passed | Numba | 二乗和338350を解析解・非JIT計算と絶対誤差1e-10以内で照合。初回JITと再実行を別計測 |
| Passed | PyMC | Beta(2,5)の事後平均0.287488。解析解2/7との差0.06未満 |
| Passed | nutpie | 同じ事後平均0.278773。解析解2/7との差0.06未満 |
| Passed | ArviZ | 推論結果のsummaryと事後分布描画が成功 |
| Passed | marimo HTTP | 認証付きHTTP 200、終了API成功、プロセス終了コード0 |
| Passed | 文書中のcmd設定 | 別Pythonの継承解除、正しいPyTensor保存先、子cmdからのmarimoコマンドを確認 |
| Not run | VS Code GUI・ブラウザーでのセル編集 | 手動操作は実施していない。実行系選択の手順を掲載 |
| Not run | 移植元16本の全実行 | 今回は環境準備。API移行・データ取得を含む移植は後続作業 |
| Not applicable | EXE・GPU検証 | 対象外 |

Numbaの計測は性能比較の根拠には使用していません。
小規模な推論の成功は、16本すべての数値結果・実行時間・API互換性の保証ではありません。

## 実装時に解消した点

- 継承された`PYTHONHOME`により、marimo起動時に別Pythonの標準ライブラリを参照していました。
  利用手順でこの指定を解除し、プロジェクトの実行系を確認するようにしました。
- `PYTENSOR_FLAGS`ではバックスラッシュが解釈されるため、保存先はスラッシュ表記にしました。
- Windows版marimoの一部状態保存はXDG設定だけでは移動できないため、
  marimo子プロセスだけ作業用プロファイルを使用する手順にしました。

Taskは統合せず、Task 1、2、4a、3、4b、5、6の順で実行しました。
Task 4は、ロック生成に必要なPython本体の導入（4a）と、環境構築・動作確認（4b）に分割しています。
1Taskあたり10,000行以下とし、生成ロックも行数に含めました。
導入済みパッケージ、Python本体、キャッシュはGit対象の成果物には含めません。
