# marimo利用手順

作成日: 2026-09-26 JST。
目的は、書籍Notebookの移植に備えて、ローカルのmarimoで編集・実行・保存する操作を理解することです。
Windows 11 AMD64、CPython 3.14.7、uv 0.11.7、marimo 0.25.0を前提にしています。
環境構築は[uv利用手順](uv_HowToUse.md)、構成と移植状況は[README](README.md)を参照してください。
本書の16本を移植した手順ではありません。以下の例は外部データを取得しない操作練習です。

## 1. 初回と毎回の準備

初回は、uv利用手順の「1. VS CodeのCommand Promptを準備する」から
「3. ロック済み環境を復元する」までを実施します。既に構築済みなら再インストールは不要です。
新しいターミナルを開くたびに、同手順の**第1節の環境変数設定をすべて実行**してください。
この文書の起動コマンドだけでは、Python・一時ファイル・設定の保存先は揃いません。

VS Codeでは`Terminal: Select Default Profile`から`Command Prompt`を選びます。
`Python: Select Interpreter`では、次の実行ファイルを選びます。

```text
C:\dev\python_bayes_intro\marimo\2026\.venv\Scripts\python.exe
```

marimo拡張機能から起動するプロセスは、統合ターミナルの設定を継承しない場合があります。
この手順では、設定済みの統合ターミナルから起動し、表示されたURLをブラウザーで開きます。
VS Codeで開くフォルダは`/python_bayes_intro/marimo/2026/`です。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
rem 前提の実行系と、ロックと依存定義の整合性を確認する。
uv --version
uv lock --check
uv run --locked --group notebook marimo --version
```

版は上記の前提と一致させます。失敗したら先へ進まず、uv利用手順で原因を切り分けます。
`notebook`グループは明示指定が必要です。環境はこのプロジェクトの`.venv`を使用します。

## 2. Notebookを作成・再開する

`first_notebook.py`は学習用の例示名で、配布済みファイルではありません。
次のコマンドは、ファイルが存在すれば編集し、存在しなければ新規作成のために開きます。
同名の大切なファイルがある場合は、例示名を未使用の名前へ変えてください。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
rem Windowsの履歴保存先として使う、プロジェクト内の作業用プロファイルを準備する。
if not exist "%BAYES_PROJECT%\.cache\marimo-profile" mkdir "%BAYES_PROJECT%\.cache\marimo-profile"
cmd.exe /d /c "set USERPROFILE=%BAYES_PROJECT%\.cache\marimo-profile&& uv run --locked --group notebook marimo edit first_notebook.py --headless --no-sandbox --host 127.0.0.1"
```

ターミナルに表示されたURLを開きます。サーバーの実行中は、そのターミナルを開いたままにします。
認証トークン付きURLは共有しません。別のコマンドを使う場合は、停止するか別の設定済みターミナルを使います。

| 指定 | この手順での意味 |
| --- | --- |
| `--locked --group notebook` | 既存ロックとNotebook用依存を使う |
| `--headless` | ブラウザーを自動起動しない。セル実行を止める指定ではない |
| `--no-sandbox` | Notebookごとの別環境を作らず、起動元のプロジェクト環境を使う |
| `--host 127.0.0.1` | このPCからアクセスするローカルサーバーとして起動する |
| 子cmdの`USERPROFILE` | Windows用の状態保存先をプロジェクト内に限定する |

`USERPROFILE`の指定は子プロセスだけに適用します。親ターミナルやWindowsのアカウント設定は変更しません。
依存追加は[uv利用手順](uv_HowToUse.md)に従い、Notebook画面からの自動インストールを使いません。
`uvx`、`--with`、`--sandbox`、セル内の`pip install`も、このプロジェクトの通常操作には使いません。

## 3. セルを編集・実行・保存する

セルは、コードや説明を分けて置く単位です。最初は次の順で操作します。

1. 空のPythonセルにコードを入力します。
2. セルの実行ボタンで実行し、出力とエラーを確認します。
3. セル追加ボタンで次のセルを作ります。次節のコードブロックは、それぞれ別セルに貼り付けます。
4. 保存操作で`.py`ファイルへ保存します。標準のCommand modeでは`Esc`、続けて`s`が保存です。
5. VS Codeのエクスプローラーで、指定した場所にファイルが存在することを確認します。

自動保存の設定もありますが、終了前には保存状態を確認してください。
キー設定を変更している場合は、`Ctrl+Shift+H`で現在のショートカットを確認します。
セルの追加・移動・削除はメニューでも操作できます。
削除したセルの変数は使えなくなるため、それを参照するセルも影響を受けます。

marimoはセル構造を含む通常のPythonファイルとして保存します。
画面のセルには次節の**セル本体だけ**を入力し、`@app.cell`や`return`の枠は手作業で追加しません。
ブラウザーとVS Codeのテキストエディターで同じファイルを同時に編集すると、変更が競合する原因になります。
編集する画面を一つに決め、切り替える前に保存状態を確認してください。

## 4. 平均・表・グラフで基本操作を確認する

以下の6セルを、空の学習用Notebookに一度ずつ追加します。
パッケージ追加、外部通信、CSVの取得、サンプリングは行いません。
説明とコードのコメントは日本語、実行結果と図の文字は英語にしています。

### セル1: 必要なライブラリ

```python
# 実行系の確認、画面表示、少量の数値計算、描画に使う。
import sys

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
```

同じimportを別セルに重複させず、必要なセルから参照します。

### セル2: 実際の実行系

```python
# サーバーではなく、セルを実行するPythonの場所と版を確認する。
print("Python executable:", sys.executable)
print("Python version:", sys.version)
```

`Python executable:`の値が第1節の`.venv`を指し、版が3.14.7であることを確認します。
異なる場合は計算を続けず、終了して起動元と環境設定を確認します。

### セル3: 観測数を選ぶ

```python
# 観測数は5～50個、5個刻み。初期値10を下流の平均計算に渡す。
sample_size = mo.ui.slider(start=5, stop=50, step=5, value=10, label="Sample size")
sample_size
```

入力部品を定義したセルでは、その部品を表示するだけにします。
選択値の`sample_size.value`は別セルで読み、値の変更を計算へ伝えます。

### セル4: 平均を計算する

```python
# 単位のない連続整数1～nをfloat64配列にし、平均を表示する。
observations = np.arange(1, sample_size.value + 1, dtype=np.float64)
sample_mean = float(np.mean(observations))
mo.md(f"Mean: **{sample_mean:.1f}**")
```

初期値10なら平均は5.5です。スライダーを20にすると10.5になります。
比較対象は連続整数の平均`(n + 1) / 2`なので、ライブラリの結果だけに依存せず確認できます。

### セル5: 表を表示する

```python
# 平均計算に使った値を表にし、入力と計算結果の対応を確認する。
mo.ui.table({"Value": observations.tolist()})
```

セルの最後の式が画面上の出力になります。`print()`は実行系確認や診断に使い、
整形した説明は`mo.md()`、表は`mo.ui.table()`、複数の表示は`mo.vstack()`などを使います。
Markdownを表示するときも、上のセルでは`mo.md()`を呼び出すPythonコードです。

### セル6: 図を表示する

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

入力を変えると、平均・表・グラフが更新されます。
Matplotlibの対話ウィンドウを別に開かず、最後の`figure`をNotebookに表示します。
この小さな例で操作を確認してから、書籍の確率モデルへ進んでください。

## 5. セルの依存関係と変数の重複

marimoは、各セルが定義する変数と参照する変数から実行順を決めます。
見た目の上下順だけでは決まりません。上の例では次の関係になります。

```text
Imports -> Slider -> Observations / Mean -> Table / Plot
```

| 注意点 | 理由と対処 |
| --- | --- |
| 同じグローバル名を複数セルで定義しない | `sample_mean`などの定義元を一つにする。import・関数名も対象 |
| セル内だけの一時変数は`_`で始める | `_temporary`は他セルに渡さない局所名として使える |
| 他セルのオブジェクトを後から変更しない | リストの`append()`やDataFrame列の変更は、自動で依存先へ伝わらない |
| 相互参照するセルを作らない | 計算が循環するため、入力から結果へ一方向の関係にする |
| 古い出力を最新結果として読まない | `stale`は入力変更後にまだ計算していない状態。実行して確認する |

Jupyterで使っていた`df`や`summary`を別セルで何度も再定義すると、移植時に重複エラーになります。
まず同じ処理段階の定義・更新を一つのセルにまとめ、複数の結果を同時に保持する場合は
役割を区別できる名前にします。単にセルを上下へ移動しても、重複や循環は解消しません。
詳しくは[0.25.0のセル実行仕様](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/reactivity.md)を参照してください。

## 6. 時間のかかる推論を実行する前に

入力の変更で下流セルが再実行されるため、PyMCのサンプリングを追加する前に実行条件を決めます。
Notebook設定の`On startup`で起動時の自動実行、`On cell change`で変更時の実行方法を確認します。
`On cell change`を`lazy`にすると、影響を受けるセルは直ちに実行されず`stale`になります。
ただし、結果セルを実行すると必要な上流セルも実行されます。`lazy`だけで推論を完全に隔離できるわけではありません。
これらの編集用設定は`marimo run`のアプリ起動には適用されません。

押したときだけ計算する例を試す場合は、基本6セルの後に次の2セルを追加します。
練習では平均をもう一度計算するだけです。ボタン例では`On cell change`を`autorun`にします。
marimo 0.25.0では、`lazy`時にボタンの値が自動で未押下へ戻らない場合があります。
このため、以下の「入力を変えた後はもう一度押す」操作は自動実行モードを前提とします。

### 追加セル7: 実行ボタン

```python
# 入力の変更と、計算を始める意思を分けて受け取る。
run_calculation = mo.ui.run_button(label="Run calculation")
run_calculation
```

### 追加セル8: ボタンを確認してから計算

```python
# ボタンが押されていない再実行では、このセルと依存先の計算を止める。
mo.stop(not run_calculation.value, mo.md("Click Run calculation."))
# 練習では軽い平均計算を行う。実際の推論は、この停止判定より後へ置く。
confirmed_mean = float(np.mean(observations))
mo.md(f"Confirmed mean: **{confirmed_mean:.1f}**")
```

初期表示では停止メッセージを確認し、ボタンを押すと平均が表示されることを確認します。
次にスライダーを変更し、再び停止メッセージになってから、ボタンを押して新しい平均を表示します。
ボタンは計算中断用ではありません。実行中の処理を止める場合は、Notebookの停止操作を使います。
中断後は入力と結果の状態を確認し、必要なら第8節の方法で新しいセッションから実行します。
キャッシュに残る結果だけでは、現在のコードを最初から実行できる証拠にはなりません。

今後サンプリングを追加する際は、乱数シード、chains、draws、tune、coresを記録します。
初期の動作確認はCPUの`cores=1`を基準にし、既存のスレッド上限も維持します。
これは16本すべての推論条件を決めるものではありません。数値精度や収束の確認は各モデルで行います。
[実行設定](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/configuration/runtime_configuration.md)と
[重いNotebookの扱い](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/expensive_notebooks.md)も参照してください。

## 7. 静的検査と編集を伴わない実行

以下は第4節の基本6セルを保存した後に試します。
各コマンドは第1節の環境設定と、第2節の作業用プロファイル作成を済ませたターミナルで実行します。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
rem セルの重複定義・依存の循環などを検査する。自動修正は行わない。
cmd.exe /d /c "set USERPROFILE=%BAYES_PROJECT%\.cache\marimo-profile&& uv run --locked --group notebook marimo check first_notebook.py"
```

`marimo check`が通っても、データ読込や数値計算の成功を保証するものではありません。
警告も確認し、実行時の結果と分けて扱います。自動書換えを行う`--fix`はこの手順では付けません。

アプリ表示で開く場合は、編集サーバーを終了してから次を実行します。

```cmd
cmd.exe /d /c "set USERPROFILE=%BAYES_PROJECT%\.cache\marimo-profile&& uv run --locked --group notebook marimo run first_notebook.py --headless --no-sandbox --host 127.0.0.1"
```

`edit`はコードを編集する画面、`run`はコード編集を伴わないアプリ表示です。
`run`でもセルの計算は実行されます。「ファイルを読むだけ」の操作ではありません。
第4節の軽い例で使い方を確かめてから、重いNotebookへ適用します。
終了方法はどちらも次節のとおりです。

## 8. 終了と新しいセッションでの確認

1. 実行中なら停止操作を行い、編集内容を保存します。
2. 起動したターミナルで`Ctrl+C`を押し、コマンド入力に戻ったことを確認します。
3. 再開するときは、第2節の同じファイルを開くコマンドを実行します。
4. 新しいターミナルを使う場合は、先にuv利用手順の環境変数を設定し直します。
5. 実行系を確認し、基本6セルの平均・表・グラフを初期状態から確認します。

ブラウザーの再読込やタブを閉じる操作だけでは、サーバー終了や状態初期化の確認になりません。
終了できない場合は、起動元のターミナルと対象プロセスを特定してください。
他のPythonプロセスまでまとめて終了するコマンドや、`.cache`全体の削除は使いません。
`.cache/python`には仮想環境が参照するPython本体があります。

## 9. よくある問題

| 症状 | 確認と対処 |
| --- | --- |
| `marimo`が見つからない | 第1節の準備と`--group notebook`を確認する。別のPythonへ追加インストールしない |
| DLLエラーや想定外のPython | uv手順の`PYTHONHOME`・`PYTHONPATH`解除と、セルの`sys.executable`を確認する |
| 重複定義のエラー | 同名変数やimportの定義元を一つにする。第5節を参照 |
| 入力を変えても表示が更新されない | `lazy`、無効化セル、ボタン待ち、上流エラー、別セルでのオブジェクト変更を確認する |
| 入力部品の値を読めない | 部品の定義と`.value`の読取を別セルにする |
| 図が表示されない | セル末尾がFigureなどの表示対象か、上流の計算が成功したかを確認する |
| URLへ接続できない | 起動元ターミナルの終了・エラーを確認し、実際に表示されたURLを使う |
| ファイルが見つからない | 作業ディレクトリ、保存状態、指定したファイル名を確認する |
| `.ipynb`をそのまま実行できない | 移植が必要。元ファイルを上書きせず、マジック・旧API・依存関係を見直す |
| CSV・Excel取得や旧PyMC APIで失敗する | 今回の操作例とは別の移植課題。元Notebookの取得先と使用APIを確認する |

## 10. 検証範囲と参考資料

この手順の検証結果は、READMEの「marimo利用手順の検証」に記録します。
掲載例の静的検査・実行確認と、VS Code・ブラウザーの手動操作確認は区別します。
書籍16本のAPI移行、数値結果、収束、処理時間は、この操作例の成功だけでは検証できません。

参照先は採用したmarimo 0.25.0の公式資料です。

- [プロジェクト環境](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/package_management/projects.md)
- [エディターの操作](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/editor_features/overview.md)
- [出力の表示](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/outputs.md)
- [CLIの実装とオプション](https://github.com/marimo-team/marimo/blob/0.25.0/marimo/_cli/cli.py)
