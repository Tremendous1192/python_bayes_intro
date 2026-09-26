# ベイズ推論「超」入門 — marimo環境（2026）

書籍『Pythonでスラスラわかる ベイズ推論「超」入門』を、VS Code・uv・marimoで学ぶための環境です。
環境と操作ガイドは準備済みです。書籍Notebookの移植はこれから進めます。

## まずはここから

| やりたいこと | 読む文書 |
| --- | --- |
| 初めて環境を準備する | [uv利用手順](uv_HowToUse.md)の第1～4節 |
| Notebookを作って動かす | [marimo利用手順](marimo_HowToUse.md) |
| 構成や採用バージョンを確認する | このREADME |

準備済みでも、新しいターミナルではuv利用手順の第1節を実行してください。
練習用の`first_notebook.py`は、marimo利用手順に沿って自分で作ります。

## フォルダの中身

対象：`/python_bayes_intro/marimo/2026/`

| ファイル・フォルダ | 役割 |
| --- | --- |
| [pyproject.toml](pyproject.toml) | Pythonの対応範囲と、使うライブラリ |
| [.python-version](.python-version) | 実行するPythonの版 |
| [uv.lock](uv.lock) | uvが生成した依存ライブラリの固定情報 |
| [uv_HowToUse.md](uv_HowToUse.md) | 環境の準備・確認・更新 |
| [marimo_HowToUse.md](marimo_HowToUse.md) | Notebookの作成・編集・実行・保存 |
| [uv_要件.md](uv_要件.md) / [marimo_要件.md](marimo_要件.md) | 環境と操作ガイドの要件 |
| `.venv/` | このプロジェクト専用のPython環境 |
| `.cache/` | Python本体、キャッシュ、設定、検証用ファイル |

`.venv/`と`.cache/`は[既存の.gitignore](../.gitignore)でGit管理から除外しています。
**`.cache/python/`は実行に必要なPython本体です。** `.cache`全体を削除・移動しないでください。

## 使っている環境

2026-09-26の検証記録：

- Windows 11 / AMD64、CPython **3.14.7**（64 bit・GIL有効）
- uv **0.11.7**、marimo **0.25.0**
- Python対応範囲：`>=3.14,<3.15`
- 実行環境：プロジェクト内の`.venv`、依存グループ：runtime + `notebook`
- 外部ツール：Graphviz **13.1.2**、MinGW g++ **16.1.0**

この環境はCPUで学ぶNotebook専用です。上位READMEのconda手順とは混用せず、EXE作成やGPU実行には使いません。
NumbaはPyTensorからも必要とされるため、`notebook`グループを省略しても除外できません。

<details>
<summary>ライブラリの採用バージョン</summary>

指定の14ライブラリに、Excel読込用の`openpyxl`を加えています。
環境準備時の記録では、互換性のために版を引き下げたものはありません。

| ライブラリ | 採用版 | 用途 |
| --- | --- | --- |
| [marimo](https://pypi.org/project/marimo/0.25.0/) | 0.25.0 | Notebookフロントエンド |
| [matplotlib](https://pypi.org/project/matplotlib/3.11.2/) | 3.11.2 | 描画 |
| [matplotlib-fontja](https://pypi.org/project/matplotlib-fontja/1.1.0/) | 1.1.0 | 日本語フォント設定 |
| [seaborn](https://pypi.org/project/seaborn/0.13.2/) | 0.13.2 | 統計描画 |
| [pymc](https://pypi.org/project/pymc/6.3.2/) | 6.3.2 | 確率モデルと推論 |
| [nutpie](https://pypi.org/project/nutpie/0.16.11/) | 0.16.11 | CPUでのNUTS |
| [arviz](https://pypi.org/project/arviz/1.3.0/) | 1.3.0 | 推論結果の集計・描画 |
| [numpy](https://pypi.org/project/numpy/2.5.3/) | 2.5.3 | 数値配列 |
| [polars](https://pypi.org/project/polars/1.44.2/) | 1.44.2 | 表処理 |
| [pandas](https://pypi.org/project/pandas/3.0.6/) | 3.0.6 | 表処理・入出力 |
| [scipy](https://pypi.org/project/scipy/1.18.1/) | 1.18.1 | 統計・科学計算 |
| [torch](https://pypi.org/project/torch/2.14.0/) | 2.14.0 | 第4章の自動微分、CPUで確認 |
| [graphviz](https://pypi.org/project/graphviz/0.21/) | 0.21 | dot本体へのPythonインターフェース |
| [numba](https://pypi.org/project/numba/0.67.0/) | 0.67.0 | Notebook専用の数値JIT |
| [openpyxl](https://pypi.org/project/openpyxl/3.1.5/) | 3.1.5 | Excel読込 |

`marimo`と`numba`の直接指定は`notebook`グループです。
`japanize_matplotlib`、`bambi`、`numpyro`は含めていません。

</details>

<details>
<summary>環境を再現するための詳細</summary>

PythonはAstral python-build-standaloneのビルド20260924を使用しています。
uv 0.11.7の内蔵カタログに対象版がなかったため、
[固定した公式カタログ](https://raw.githubusercontent.com/astral-sh/uv/299a93de4b94e754f260c673d2de456afbdd4fb7/crates/uv-python/download-metadata.json)から導入しました。
取得コマンドとハッシュは[uv利用手順](uv_HowToUse.md)の第2節にあります。

依存は公式PyPIから取得し、`exclude-newer = "2026-09-27T00:00:00+09:00"`で公開日時を制限しています。
対象はWindows AMD64用の安定版で、ソース配布物のビルドは許可していません。

ロックの原本は`pyproject.toml`、生成先は`uv.lock`です。
初回はuv 0.11.7で
`uv lock --python C:/dev/python_bayes_intro/marimo/2026/.cache/python/cpython-3.14.7-windows-x86_64-none/python.exe`
を実行しました。ロックは83パッケージ・1,025行、実際の導入は80パッケージでした。

通常はPythonの自動取得を禁止し、初回導入時だけ許可します。
キャッシュ・一時ファイル・marimoの状態はプロジェクト内へ保存します。
WindowsのPATHやレジストリ、共通のPythonコマンドは変更しません。

</details>

## 書籍Notebookの移植について

対象は、本編11本と参考5本の計16本です。このフォルダへの移植はまだ行っていません。

| 移植元 | 対象 |
| --- | --- |
| [notebooks/](../../notebooks/) | 第1～4章、5.1～5.4、6.1～6.3の11本 |
| [sample-notebooks/](../../sample-notebooks/) | 第4章の図、潜在変数モデル簡略版、3クラス版、FAQ、書籍評価の5本 |

`6_3_IRTによるテスト結果評価_GPU版.ipynb`は対象外です。
移植では、Colab向け処理・IPythonマジック・旧`japanize_matplotlib`に加え、
PyMC 6／ArviZ 1のAPI変更に対応します。参考：[PyMCの例](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html)、
[ArviZの変更案内](https://github.com/arviz-devs/arviz/issues/2548)。

Iris・CSV・Excelなどの実データは未取得です。
移植時に取得元と失敗時の対処を決め、保存先をプロジェクト内に揃えます。
説明・コメントは日本語、画面出力や図の文字は英語にします。

## 検証記録

以下は2026-09-26の環境準備・操作ガイド作成時の記録です。今回の文章整理で再実行した結果ではありません。

| 判定 | 内容 |
| --- | --- |
| Passed | ロックと依存定義の整合、Pythonと15直接依存の版 |
| Passed | Excel読込、Polars・SciPy・PyTorchの計算、Matplotlib・Seaborn・Graphvizの描画 |
| Passed | Numbaと非JITの照合、PyMC・nutpieの小規模推論、ArviZの集計・描画 |
| Passed | ガイドの8セルの静的検査、平均5.5／10.5、表・グラフ・ボタン停止条件の確認 |
| Passed | marimoの認証付きHTTP起動、`edit`の終了APIと終了コード0 |
| Not run | VS Code・ブラウザーでの手動操作、ガイドの描画の目視、書籍16本の全実行 |
| Not applicable | EXE・GPU検証 |

小さな操作例の成功だけでは、書籍全体のAPI互換性・収束・実行時間までは確認できません。

<details>
<summary>検証条件と再実行コマンド</summary>

PyMC・nutpieは合成データで各2 chains・400 tune・600 draws、`cores=1`、`random_seed=42`を使用。
事後平均はそれぞれ0.287488・0.278773で、解析解`2/7`との差は0.06未満でした。
BLAS・OpenMP・Numba・Polarsは各2スレッドを上限にしています。
Numbaの二乗和338350は非JIT計算と絶対誤差1e-10以内で一致し、初回JITと再実行を別計測しました。
確認した組合せはNumba 0.67.0 / llvmlite 0.49.0 / NumPy 2.5.3です。

ガイドは基本6セルとボタン用2セルを抽出し、marimo 0.25.0で検査・実行しました。
新規プロセスで基本例2ケース、ボタン未押下2ケース、押下相当1ケースを確認しています。
入力変更とボタン押下は値の差替えによる模擬で、ブラウザー操作の確認ではありません。

生成元は`marimo_HowToUse.md`、生成器は`.cache/validation/marimo-guide-20260926/test_guide.py`です。
生成した`basic_example.py`（66行）・`button_example.py`（84行）で確認しました。
`run_validation.cmd`はuv利用手順の設定と`test_guide.py launcher`から生成したASCII・CRLFのファイルです。
これらはGit管理外のため、残っている場合だけ次のコマンドで再実行できます。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
rem 標準ライブラリだけで、保存先設定を含む検証用cmdを生成する。
.venv\Scripts\python.exe -I -B .cache\validation\marimo-guide-20260926\test_guide.py launcher
rem 外部取得を禁止し、掲載セルの生成・静的検査・5ケースを順に確認する。
cmd.exe /d /c .cache\validation\marimo-guide-20260926\run_validation.cmd
rem 検証用Notebookだけを対象に、ローカルHTTPの起動と回収を確認する。
cmd.exe /d /c .cache\validation\marimo-guide-20260926\run_validation.cmd http
```

HTTP確認の`run`プロセスは、検証が起動したプロセスツリーだけを終了しました。
利用者による`Ctrl+C`の動作確認は未実施です。

</details>
