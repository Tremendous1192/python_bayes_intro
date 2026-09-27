# 目的
`Pythonでスラスラわかる　ベイズ推論「超」入門, 赤石雅典・著　須山敦志・監修` のプログラムを、ローカル環境(VS Code + uv + marimo)で作成したい。
`C:\dev\python_bayes_intro\notebooks` と `C:\dev\python_bayes_intro\sample-notebooks` の内容を書きなおしたい。
ただし、`sample-notebooks/6_3_IRTによるテスト結果評価_GPU版.ipynb` と `sample-notebooks/書籍評価.ipynb` は移植対象外とし、本編11本・参考4本の計15本を対象とする。
uv のコマンド操作は人間が扱うことを前提として、シンプルな操作にする。
marimo は VS Code Extension を使って、既存のVS Codeウィンドウ上で編集する。
## 参考書籍URL
* Pythonでスラスラわかる　ベイズ推論「超」入門, 赤石雅典・著　須山敦志・監修
    * https://www.kspub.co.jp/book/detail/5337639.html


# Codexへの依頼
目的のための準備です。
1. uv の環境設定ファイルを作成してください。
    * `C:\dev\python_bayes_intro\marimo\2026\pyproject.toml`
1. `C:\dev\python_bayes_intro\marimo\2026\HowToUse_uv.md` にuvの使い方・コマンドを書いてください。
1. `C:\dev\python_bayes_intro\marimo\2026\README.md` にこのフォルダの内容を書いてください。
1. 環境設定の補助コード `configure_runtime.py` と検証コード `test_configure_runtime.py` は `C:\dev\python_bayes_intro\marimo\2026\log` に配置してください。
1. 作成したファイルに、人間が理解できるようにコメントを加筆してください

## 文章の要件
1. `.md` ファイルは箇条書きで簡潔に書く
1. `.cmd`, `.toml`, `.py` ファイルには箇条書きのコメントを多くつける


# uv環境の要件
* Python >=3.14, <3.15
* `C:\dev\python_bayes_intro\marimo\2026\.venv\` とする
## ライブラリ
* 2026-09-26時点の最新安定版を原則とする。
* 依存関係に不整合がある場合は、Python・OSの要件を維持したうえで、同日以前に公開された互換性のある安定版へ調整できる。
* 採用バージョンと、調整した場合の理由を記録する。
### ノートブック
* marimo
### プロット
* matplotlib
* matplotlib-fontja
* seaborn
### ベイズ推論
* pymc
* nutpie
* arviz
### その他
* numpy
* polars
* pandas
* scipy
* torch
* graphviz
* numba
### (Must Not)このフォルダでは使用しないライブラリ
* japanize_matplotlib
* bambi
* numpyro

