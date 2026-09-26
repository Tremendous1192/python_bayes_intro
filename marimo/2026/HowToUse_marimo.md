# marimo利用手順

VS Codeの公式marimo拡張機能で、Notebookを作成・編集・実行・保存します。
作成日：2026-09-26。対象：Windows 11 AMD64、CPython 3.14.7、marimo 0.25.0、
拡張機能`marimo-team.vscode-marimo` 0.18.1。

環境の初回準備は[uv利用手順](uv_HowToUse.md)、確認済みの範囲は[README](README.md)を参照してください。

## 1. 専用ウィンドウを開く

VS CodeのCommand Promptで実行します。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
open_vscode.cmd
```

開いた専用ウィンドウを使います。初回の拡張機能とフォルダの信頼確認は、uv利用手順の第4節にあります。
この起動方法で、NotebookのPythonにもキャッシュ保存先とスレッド上限を継承させます。

## 2. 用意されたNotebookを開く

1. エクスプローラーで[examples/basic_usage.py](examples/basic_usage.py)を選択します。
2. コマンドパレットで**`marimo: Open as marimo notebook`**を実行します。
3. Notebookのカーネル選択で、次のPythonを選びます。

```text
C:\dev\python_bayes_intro\marimo\2026\.venv\Scripts\python.exe
```

`Python: Select Interpreter`の設定と、Notebookで実際に選ばれたカーネルは区別して確認します。
`marimo sandbox`は選びません。候補が見つからない場合は環境の同期と、拡張機能の環境選択を確認してください。

この例は8セルです。全セル実行の操作を行い、最初の出力の`Python executable`と`Python version`を確認します。
Python 3.14.7、GIL有効でなければ確認セルがエラーになります。
別の環境で動いた場合はカーネルを停止し、選択を直します。

この版の拡張機能には、旧版の`marimo.pythonPath`設定はありません。
プロジェクト設定は既存の`python.defaultInterpreterPath`と、実際のNotebookのカーネル選択を使います。

## 3. 平均・表・図を試す

| セル | 内容 | 確認すること |
| --- | --- | --- |
| 1 | ライブラリの読込 | importが成功する |
| 2 | Pythonの確認 | 指定の.venv、3.14.7 |
| 3 | `Sample size`スライダー | 初期値10、5～50を5刻み |
| 4 | 平均計算 | 1～10の平均が5.5 |
| 5 | 表 | `Value`列に1～10 |
| 6 | 図 | 観測値と、平均5.5の破線 |
| 7 | `Run calculation`ボタン | 計算開始の入力 |
| 8 | 押下後だけ計算 | 未押下なら停止メッセージ |

スライダーを20へ変えると、平均は**10.5**、表は1～20、図の平均線は10.5になります。
等差数列の平均`(n + 1) / 2`と照合できます。

ボタン操作は、`Cell changes`を**`Auto-run`**にして試します。
アクティブなNotebookで`marimo: Show notebook menu`を開き、実行設定を確認します。
初期状態では`Click Run calculation.`と表示され、ボタンを押すと`Confirmed mean`が表示されます。
スライダーを変えた後は、再びボタンを押して計算します。

`Lazy`では影響するセルが実行待ちになります。`marimo: Run stale cells`で未実行のセルを確認できますが、
この版ではボタンの未押下への復帰が`Auto-run`と異なる場合があるため、ボタン例の確認には使いません。
実行設定は、計算の正しさや収束を保証するものではありません。

## 4. 自分のNotebookを作る

コマンドパレットで**`Create: New marimo notebook`**を実行し、
このフォルダ内に`first_notebook.py`などの名前で保存します。同名ファイルがある場合は別名にします。
第2節と同じカーネルを選び、次のコードを1つのPythonセルへ入力して実行します。

```python
# 画面表示と、外部データを使わない軽い平均計算を準備する。
import marimo as mo
import numpy as np

# 単位のない整数1～10をfloat64で扱い、平均5.5を確認する。
values = np.arange(1, 11, dtype=np.float64)
mean_value = float(np.mean(values))
mo.md(f"Mean: **{mean_value:.1f}**")
```

`@app.cell`・`def`・`return`などのファイル構造は拡張機能が扱うため、セルにはセル本体を入力します。
同じ名前のimportや変数を別セルで重複定義しないでください。

用意された8セルを練習する場合は、`basic_usage.py`を別名でコピーして使えます。
[検証用コード](examples/test_basic_usage.py)は元の例の期待値を確認するため、元ファイルの計算を変えると検証結果も変わります。

## 5. 編集・保存・セルの依存関係

コードを編集し、必要なセルを実行して、**`Ctrl+S`**で保存します。
エクスプローラーで`.py`ファイルに保存されたことを確認してください。

marimoは、変数の依存関係から実行順を決めます。
例の流れは「ライブラリ → スライダー → 観測値・平均 → 表・図」です。

| 規則 | 理由・対処 |
| --- | --- |
| 同じ名前の定義は1つのセルへ | 重複する変数・import・関数定義を避ける |
| セル内だけの一時変数は`_`で始める | 他セルから参照する変数と区別する |
| 他セルのオブジェクトを直接変更しない | `append()`や列の書換えだけでは依存先へ変更が伝わらない場合がある |
| 循環参照を作らない | 入力→計算→表示の一方向にする |
| `stale`を実行済み結果と取り違えない | 入力変更後、まだ計算していないセルを確認する |

通常のテキストエディターとNotebookエディターで、同じファイルを同時に編集しないでください。
セルの説明やコメントは日本語、画面出力と図の文字は英語にします。

## 6. 検査と再現確認

専用ウィンドウの統合ターミナルで、例の静的検査と数値確認を行えます。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
uv run --locked --group notebook marimo check examples/basic_usage.py
uv run --locked --group notebook python examples/test_basic_usage.py
```

静的検査は、セルの重複定義や依存の循環などを確認します。
検証コードは、初期値10・変更後20・変更後のボタン押下相当の3ケースを、別々の新規プロセスで順に実行します。
平均5.5／10.5、表の値、図の値、未押下時の停止を確認し、失敗時は終了コードを失敗にします。

スライダー変更・ボタン押下は定義の差替えによる模擬です。実際の画面操作の代わりにはなりません。
画面では第2・3節に加え、保存後の再読込、カーネルの停止・再起動も確認してください。

## 7. 重い計算を追加するとき

PyMCなどの推論は、実行ボタンの停止判定より後に置きます。

```python
# ボタン未押下の再実行では、このセルと下流の重い計算を止める。
mo.stop(not run_calculation.value, mo.md("Click Run calculation."))
```

この断片だけではボタンは作成されません。例のセル7・8のように、ボタン定義と値を読むセルを分けます。
ボタンは開始用であり、実行中の計算の中断用ではありません。

乱数シード、chains、draws、tune、coresを記録し、まず`cores=1`で確認します。
`env.cmd`のスレッド上限を維持し、数値精度・収束・時間はモデルごとに検証します。
ライブラリ追加はuv利用手順に従い、セルや拡張機能の導入機能から依存を変更しません。

## 8. 終了・再開する

1. 実行中の処理を停止し、`Ctrl+S`で保存します。
2. アクティブなNotebookで`marimo: Shut Down Kernel`を実行します。
3. 作業を終える場合は専用ウィンドウを閉じます。
4. 再開時は`open_vscode.cmd`から開き、第2節のPython確認と第3節の出力確認を行います。

セッションを初期化する場合は`marimo: Restart notebook kernel`を使い、必要なセルを最初から実行します。
拡張機能が管理する実行では、統合ターミナルの`Ctrl+C`をカーネル停止の手順にしません。
終了できない場合は対象のカーネルを確認し、他のPythonプロセスをまとめて終了しないでください。

## 9. 困ったとき

| 症状 | 対処 |
| --- | --- |
| 通常のPythonコードとして開く | `marimo: Open as marimo notebook`を実行する |
| 拡張機能が動かない | 専用ウィンドウの拡張機能、フォルダの信頼、`marimo: Show diagnostics`を確認する |
| 違うPythonが動く | Notebook側のカーネル選択と`sys.executable`を確認する |
| 依存が不足する | uv利用手順で`uv sync --locked --group notebook`を実行する |
| 入力変更後に更新されない | `Cell changes`、stale、停止中のセル、上流エラーを確認する |
| ボタンを押す前に計算される | 停止判定が重い処理より前にあるか確認する |
| パス・保存先が違う | 専用ウィンドウを閉じ、`open_vscode.cmd`から開き直す |
| 元の.ipynbや旧APIで失敗する | 書籍16本は未移植。元ファイルを上書きせず、移植作業で対応する |

## 参考資料

- [公式marimo拡張機能](https://marketplace.visualstudio.com/items?itemName=marimo-team.vscode-marimo)
- [拡張機能の公式リポジトリ](https://github.com/marimo-team/marimo-lsp)
- [marimo 0.25.0のセル依存関係](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/reactivity.md)
- [marimo 0.25.0の実行設定](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/configuration/runtime_configuration.md)
- [marimo 0.25.0の重いNotebookの扱い](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/expensive_notebooks.md)
