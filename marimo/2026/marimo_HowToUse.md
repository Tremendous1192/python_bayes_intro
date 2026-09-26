# marimo利用手順

小さなNotebookを作りながら、編集・実行・保存を試します。
練習には平均・表・グラフを使います。外部データの取得は不要です。

対象はWindows 11 AMD64 / CPython 3.14.7 / uv 0.11.7 / marimo 0.25.0。
環境の準備は[uv利用手順](uv_HowToUse.md)、構成と検証記録は[README](README.md)を参照してください。
作成日：2026-09-26。

## 1. 起動前の準備

初回はuv利用手順の第1～3節を済ませます。
準備済みでも、**新しいターミナルではuv利用手順の第1節を毎回実行**してください。

VS Codeで`/python_bayes_intro/marimo/2026/`を開き、
`Python: Select Interpreter`で次のPythonを選びます。

```text
C:\dev\python_bayes_intro\marimo\2026\.venv\Scripts\python.exe
```

設定済みのCommand Promptで、環境を確認します。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
rem 前提の実行系と、ロックと依存定義の整合性を確認する。
uv --version
uv lock --check
uv run --locked --group notebook marimo --version
```

エラーや版の違いがあれば、uv利用手順の「困ったとき」を確認してください。

## 2. Notebookを開く

`first_notebook.py`は練習用の名前です。未作成なら新規作成、既存なら編集で開きます。
同名の大切なファイルがある場合は、別の名前に変えてください。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
rem Windowsの履歴保存先として使う、プロジェクト内の作業用プロファイルを準備する。
if not exist "%BAYES_PROJECT%\.cache\marimo-profile" mkdir "%BAYES_PROJECT%\.cache\marimo-profile"
cmd.exe /d /c "set USERPROFILE=%BAYES_PROJECT%\.cache\marimo-profile&& uv run --locked --group notebook marimo edit first_notebook.py --headless --no-sandbox --host 127.0.0.1"
```

表示されたURLをブラウザーで開きます。作業中はターミナルも開いたままにします。
認証トークン付きのURLは共有しないでください。

<details>
<summary>起動コマンドの意味</summary>

| 指定 | 意味 |
| --- | --- |
| `--locked --group notebook` | ロック済みのNotebook用環境を使う |
| `--headless` | ブラウザーを自動起動しない。セルは実行される |
| `--no-sandbox` | このプロジェクトの環境を使う |
| `--host 127.0.0.1` | このPCから接続する |
| 子cmdの`USERPROFILE` | 履歴などをプロジェクト内へ保存する。親ターミナルの設定は変えない |

</details>

依存の追加はuv利用手順で行います。`uvx`・`--with`・`--sandbox`やセル内のインストールは使いません。

## 3. 編集・実行・保存する

「セル」はコードや説明を置く区切りです。次節のコードブロックを、それぞれ別のPythonセルに貼り付けます。

1. セルにコードを入力し、実行ボタンで結果を確認します。
2. セル追加ボタンで次のセルを作ります。
3. 保存します。標準設定では`Esc`→`s`です。ショートカット一覧は`Ctrl+Shift+H`で開けます。
4. VS Codeで`.py`ファイルが保存されたことを確認します。

セルの移動・削除はメニューから操作できます。削除すると、その変数を使うセルにも影響します。
`@app.cell`や`return`の枠はmarimoが作るので、入力するのは**セル本体だけ**です。
編集はブラウザーかVS Codeのどちらか一方で行い、切り替える前に保存してください。

## 4. 平均・表・グラフを試す

空のNotebookに、次の6セルを順に追加します。

### セル1：ライブラリを読み込む

```python
# 実行系の確認、画面表示、少量の数値計算、描画に使う。
import sys

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
```

同じimportを別セルに重複させる必要はありません。

### セル2：Pythonを確認する

```python
# サーバーではなく、セルを実行するPythonの場所と版を確認する。
print("Python executable:", sys.executable)
print("Python version:", sys.version)
```

第1節の`.venv`とPython 3.14.7が表示されればOKです。
異なる場合は終了して、起動元の環境を確認します。

### セル3：観測数を選ぶ

```python
# 観測数は5～50個、5個刻み。初期値10を下流の平均計算に渡す。
sample_size = mo.ui.slider(start=5, stop=50, step=5, value=10, label="Sample size")
sample_size
```

スライダーの値（`sample_size.value`）は、次のセルで読み取ります。

### セル4：平均を計算する

```python
# 単位のない連続整数1～nをfloat64配列にし、平均を表示する。
observations = np.arange(1, sample_size.value + 1, dtype=np.float64)
sample_mean = float(np.mean(observations))
mo.md(f"Mean: **{sample_mean:.1f}**")
```

1～10の平均は**5.5**、スライダーを20に変えると**10.5**になります。
連続整数の平均`(n + 1) / 2`と比べてみてください。

### セル5：表を表示する

```python
# 平均計算に使った値を表にし、入力と計算結果の対応を確認する。
mo.ui.table({"Value": observations.tolist()})
```

セルの最後の式が画面に表示されます。説明には`mo.md()`、表には`mo.ui.table()`を使います。

### セル6：グラフを表示する

```python
# 横軸は1始まりの観測番号、縦軸は観測値。平均の位置を破線で示す。
figure, axis = plt.subplots(figsize=(6, 3))
axis.plot(np.arange(1, observations.size + 1), observations, marker="o")
axis.axhline(sample_mean, color="tab:red", linestyle="--", label="Mean")
axis.set(xlabel="Observation", ylabel="Value", title="Sample mean")
axis.legend()
figure.tight_layout()
# 再実行で描画管理用のFigureが蓄積しないよう閉じ、オブジェクトを出力する。
plt.close(figure)
figure
```

スライダーを動かし、平均・表・グラフが一緒に変わることを確認してください。

## 5. セル同士のつながり

marimoは、変数のつながりから実行順を決めます。セルの上下を入れ替えるだけでは順番は変わりません。

```text
Imports -> Slider -> Observations / Mean -> Table / Plot
```

| 覚えておくこと | 例・対処 |
| --- | --- |
| 同じ名前の定義は1つのセルに | 変数・import・関数名を別セルで重複させない |
| セル内だけの一時変数は`_`で始める | `_temporary`は他セルへ渡さない |
| 他セルのオブジェクトを後から変更しない | `append()`や列の書換えは依存先へ自動で伝わらない |
| セル同士の相互参照を避ける | 入力→計算→結果の一方向にする |
| `stale`の出力は再実行する | 入力変更後、まだ計算していない状態 |

Jupyterから移すときは、同じ変数の定義・更新を1つのセルにまとめると整理しやすくなります。
複数の結果を残すなら、役割に合わせた別の名前を付けます。

## 6. 重い計算は、実行のタイミングを決める

PyMCなどの時間がかかる処理を追加する前に、Notebookの設定を確認します。

| 設定 | 確認すること |
| --- | --- |
| `On startup` | 起動時に自動実行するか |
| `On cell change` | `autorun`は自動実行、`lazy`は変更の影響を受けるセルを実行待ちにする |

`lazy`でも、結果セルを実行すると必要な上流セルが動きます。
また、この編集用設定は`marimo run`には適用されません。

ボタンを押して計算する例も試せます。**`On cell change`を`autorun`にして**、次の2セルを追加してください。
marimo 0.25.0の`lazy`では、ボタンが未押下に戻らない場合があります。

### 追加セル7：実行ボタン

```python
# 入力の変更と、計算を始める意思を分けて受け取る。
run_calculation = mo.ui.run_button(label="Run calculation")
run_calculation
```

### 追加セル8：押したら計算する

```python
# ボタンが押されていない再実行では、このセルと依存先の計算を止める。
mo.stop(not run_calculation.value, mo.md("Click Run calculation."))
# 練習では軽い平均計算を行う。実際の推論は、この停止判定より後へ置く。
confirmed_mean = float(np.mean(observations))
mo.md(f"Confirmed mean: **{confirmed_mean:.1f}**")
```

最初は停止メッセージが出ます。ボタンを押すと平均が表示されます。
スライダーを変えると再び停止するので、もう一度押して計算してください。

ボタンは計算の開始用です。中断にはNotebookの停止操作を使い、必要なら第8節の方法で再起動します。
本格的な推論では乱数シード・chains・draws・tune・coresを記録し、まず`cores=1`で確認します。
第1節で設定したスレッド上限を維持し、精度や収束はモデルごとに確認してください。

## 7. 検査する・アプリとして開く

基本6セルを保存し、編集サーバーを`Ctrl+C`で終了してから試します。
同じ設定済みターミナルで、重複定義や依存の循環を検査できます。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
rem セルの重複定義・依存の循環などを検査する。自動修正は行わない。
cmd.exe /d /c "set USERPROFILE=%BAYES_PROJECT%\.cache\marimo-profile&& uv run --locked --group notebook marimo check first_notebook.py"
```

エラー・警告を確認してください。この検査では、データ読込や数値計算の正しさまでは確認できません。

コードを編集しないアプリ表示には、`run`を使います。こちらもセルの計算は実行されます。

```cmd
cmd.exe /d /c "set USERPROFILE=%BAYES_PROJECT%\.cache\marimo-profile&& uv run --locked --group notebook marimo run first_notebook.py --headless --no-sandbox --host 127.0.0.1"
```

## 8. 終了・再開する

1. 実行中の処理を止め、編集内容を保存します。
2. 起動したターミナルで`Ctrl+C`を押し、コマンド入力に戻ったことを確認します。
3. 再開は第2節のコマンドです。新しいターミナルなら、先にuv利用手順の第1節を実行します。
4. Pythonの場所と版、平均・表・グラフを初期状態から確認します。

ブラウザーのタブを閉じるだけではサーバーは終了しません。
終了できない場合は起動元と対象プロセスを確認し、他のPythonプロセスをまとめて終了しないでください。

## 9. 困ったとき

| 症状 | 確認すること |
| --- | --- |
| `marimo`が見つからない・DLLエラー | uv利用手順の環境設定、`--group notebook`、`sys.executable` |
| 重複定義のエラー | 同名の変数・importが複数セルにないか |
| 入力を変えても更新されない | `lazy`、無効化セル、ボタン待ち、上流エラー、オブジェクトの直接変更 |
| 入力部品の値を読めない | 部品の定義と`.value`の読取を別セルにしたか |
| 図が出ない | セル末尾が`figure`か、上流の計算が成功しているか |
| URLを開けない | ターミナルのエラーと、実際に表示されたURL |
| ファイルが見つからない | 作業フォルダ、保存状態、ファイル名 |
| 元の`.ipynb`や旧APIで失敗する | 書籍Notebookの移植が必要。元ファイルは上書きしない |

## 10. 参考資料

掲載例の検証記録は[README](README.md)にあります。
VS Code・ブラウザーの手動操作と、書籍16本の移植・全実行は未確認です。

詳しい仕様はmarimo 0.25.0の公式資料を参照してください。

- [セルの実行と依存関係](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/reactivity.md)
- [実行設定](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/configuration/runtime_configuration.md)
- [重いNotebookの扱い](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/expensive_notebooks.md)
- [プロジェクト環境](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/package_management/projects.md)
- [エディターの操作](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/editor_features/overview.md)
- [出力の表示](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/outputs.md)
- [CLIの実装とオプション](https://github.com/marimo-team/marimo/blob/0.25.0/marimo/_cli/cli.py)
