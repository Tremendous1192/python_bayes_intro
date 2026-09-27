# marimo利用手順

- VS Codeの公式marimo拡張機能で、Notebookを作成・編集・実行・保存します。
- 作成日：2026-09-26。文章・参照先の更新日：2026-09-27。
- 対象：Windows 11 AMD64、CPython 3.14.7、marimo 0.25.0、拡張機能`marimo-team.vscode-marimo` 0.18.1。
- 初回準備は[uv利用手順](HowToUse_uv.md)、検証結果は[検証記録](log/移植検証.md)を参照します。

## 1. 現在のウィンドウで準備する

1. [uv利用手順](HowToUse_uv.md)の第1～4節で、環境同期と`call env.cmd configure`を済ませます。
2. 既存のVS Codeウィンドウをそのまま使います。
   - 親フォルダを開いている場合は、対象の`2026`フォルダを同じワークスペースへ追加します。
   - 初回の拡張機能・フォルダの信頼・設定範囲は、uv利用手順の第4節に従います。
3. 起動済みのNotebookカーネルがあれば停止し、登録後のPythonで起動し直します。

- 保存先・並列数は専用`.venv`の起動フックから設定します。
- VS Codeの別ウィンドウや専用ユーザーデータ領域を作成しません。

## 2. 用意されたNotebookを開く

1. エクスプローラーで[examples/basic_usage.py](examples/basic_usage.py)を選びます。
2. コマンドパレットで`marimo: Open as marimo notebook`を実行します。
3. Notebookのカーネル選択で、次のPythonを選びます。

```text
C:\dev\python_bayes_intro\marimo\2026\.venv\Scripts\python.exe
```

4. 全8セルを実行し、最初の出力の`Python executable`・`Python version`を確認します。
   - 指定の`.venv`、Python 3.14.7、GIL有効が条件です。版・GILが違う場合は確認セルがエラーになります。
   - 別の環境で動いた場合はカーネルを停止し、選択を直します。

- `Python: Select Interpreter`の設定と、Notebookで実際に選んだカーネルは別々に確認します。
- `marimo sandbox`は選びません。候補がない場合は環境の同期・拡張機能の環境選択を確認します。
- この版には旧設定`marimo.pythonPath`がありません。`python.defaultInterpreterPath`とNotebookのカーネル選択を使います。

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

1. スライダーを20へ変えます。
   - 平均10.5、表1～20、図の平均線10.5を確認します。
   - 等差数列の平均`(n + 1) / 2`と照合できます。
2. アクティブなNotebookで`marimo: Show notebook menu`を開き、`Cell changes`を`Auto-run`にします。
3. 初期状態の`Click Run calculation.`を確認し、ボタンを押します。
   - `Confirmed mean`が表示されます。スライダー変更後は再びボタンを押します。

- `Lazy`では影響するセルが実行待ちになります。`marimo: Run stale cells`で実行できます。
- この版ではボタンの未押下への復帰が`Auto-run`と異なる場合があるため、ボタン例は`Auto-run`で確認します。
- 実行設定だけで計算の正しさ・収束は保証されません。

## 4. 自分のNotebookを作る

1. コマンドパレットで`Create: New marimo notebook`を実行します。
2. このフォルダ内に`first_notebook.py`などの名前で保存します。同名ファイルがあれば別名にします。
3. 第2節と同じカーネルを選び、次を1つのPythonセルに入力して実行します。

```python
# - 画面表示と、外部データを使わない軽い平均計算を準備する。
import marimo as mo
import numpy as np

# - 単位のない整数1～10をfloat64で扱い、平均5.5を確認する。
values = np.arange(1, 11, dtype=np.float64)
mean_value = float(np.mean(values))
mo.md(f"Mean: **{mean_value:.1f}**")
```

- `@app.cell`・`def`・`return`などのファイル構造は拡張機能が扱います。セルにはセル本体だけを入力します。
- 同じ名前のimport・変数を別セルで重複定義しません。
- 8セルを練習する場合は、`basic_usage.py`を別名でコピーできます。
- [検証用コード](examples/test_basic_usage.py)は元の例を検証します。元ファイルの計算を変えると検証結果も変わります。

## 5. 編集・保存・セルの依存関係

1. コードを編集し、必要なセルを実行します。
2. `Ctrl+S`で保存し、エクスプローラーで`.py`ファイルを確認します。

- marimoは変数の依存関係から実行順を決めます。
- 例の流れは「ライブラリ → スライダー → 観測値・平均 → 表・図」です。

| 規則 | 理由・対処 |
| --- | --- |
| 同じ名前の定義は1つのセルへ | 変数・import・関数定義の重複を避ける |
| セル内だけの一時変数は`_`で始める | 他セルから参照する変数と区別する |
| 他セルのオブジェクトを直接変更しない | `append()`・列の書換えだけでは依存先へ変更が伝わらない場合がある |
| 循環参照を作らない | 入力→計算→表示の一方向にする |
| `stale`を実行済み結果と取り違えない | 入力変更後、まだ計算していないセルを確認する |

- テキストエディターとNotebookエディターで、同じファイルを同時に編集しません。
- 説明・コメントは日本語、画面出力・図の文字は英語にします。

## 6. 検査と再現確認

1. 現在のウィンドウの設定済み統合ターミナルで、静的検査と数値確認を実行します。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
call env.cmd
call env.cmd check
uv run --locked --group notebook marimo check examples/basic_usage.py
uv run --locked --group notebook python examples/test_basic_usage.py
```

- 静的検査はセルの重複定義・依存の循環などを確認します。
- 検証コードは、初期値10・変更後20・変更後のボタン押下相当を、別々の新規プロセスで順に実行します。
- 平均5.5／10.5、表・図の値、未押下時の停止を確認します。失敗時は終了コードも失敗になります。
- 入力変更・押下は定義の差替えによる模擬です。画面操作の確認には含めません。

2. 画面で第2・3節の入力操作を確認します。
3. 保存・再読込を行い、第8節に従ってカーネル停止・再起動も確認します。

## 7. 重い計算を追加するとき

- PyMCなどの推論は、実行ボタンの停止判定より後に置きます。

```python
# - ボタン未押下の再実行では、このセルと下流の重い計算を止める。
mo.stop(not run_calculation.value, mo.md("Click Run calculation."))
```

- この断片だけではボタンは作成されません。例のセル7・8のように、ボタン定義と値を読むセルを分けます。
- ボタンは計算開始用です。実行中の計算を中断するボタンではありません。
- 乱数シード・chains・draws・tune・coresを記録し、まず`cores=1`で確認します。
- `env.cmd`のスレッド上限を維持し、精度・収束・時間はモデルごとに検証します。
- 依存追加はuv利用手順に従います。セル・拡張機能の導入機能からは追加しません。

## 8. 終了・再開する

1. 実行中の処理を停止し、`Ctrl+S`で保存します。
2. アクティブなNotebookで`marimo: Shut Down Kernel`を実行します。
3. 作業を終える場合は、必要に応じてターミナルやNotebookを閉じます。
4. 再開時も現在のウィンドウを使い、第2節のPython・第3節の出力を確認します。

- セッション初期化には`marimo: Restart notebook kernel`を使い、必要なセルを最初から実行します。
- 拡張機能が管理するカーネルは、統合ターミナルの`Ctrl+C`で停止する手順にしません。
- 終了できない場合は対象のカーネルを確認します。他のPythonプロセスをまとめて終了しません。

## 9. 困ったとき

| 症状 | 対処 |
| --- | --- |
| 通常のPythonコードとして開く | `marimo: Open as marimo notebook`を実行する |
| 拡張機能が動かない | 現在のウィンドウの拡張機能、フォルダの信頼、`marimo: Show diagnostics`を確認する |
| 違うPythonが動く | Notebook側のカーネル選択・`sys.executable`を確認する |
| 依存が不足する | uv利用手順で`uv sync --locked --group notebook`を実行する |
| 入力変更後に更新されない | `Cell changes`、stale、停止中のセル、上流エラーを確認する |
| ボタンを押す前に計算される | 停止判定が重い処理より前にあるか確認する |
| パス・保存先が違う | `call env.cmd check`、必要なら再登録してカーネルを再起動する |
| 元の.ipynbや旧APIで失敗する | [READMEの対応表](README.md#書籍notebookの移植)から移植済みの.pyを開く。元の.ipynbは変更しない |

## 10. 同じウィンドウでの受入確認

1. 既存ウィンドウのCommand Promptで次を実行します。

```cmd
cd /d C:\dev\python_bayes_intro\marimo\2026
call env.cmd
call env.cmd check
```

2. 第2・3節でカーネル選択、スライダー10→20、平均5.5→10.5、ボタン押下を確認します。
3. `Ctrl+S`で保存し、Notebookを閉じて同じウィンドウで開き直します。
4. 第8節でカーネル停止・再起動を行い、初期状態から再実行します。
5. この操作を理由とする追加ログイン要求がなく、既存ウィンドウ・既存アカウントを維持できたことを確認します。

- 画面確認は自動テストと区別し、実施日・結果・失敗した操作をREADMEへ記録します。
- 起動スクリプト廃止の最終判定には、この画面確認も必要です。

## 参考資料

- [公式marimo拡張機能](https://marketplace.visualstudio.com/items?itemName=marimo-team.vscode-marimo)
- [拡張機能の公式リポジトリ](https://github.com/marimo-team/marimo-lsp)
- [marimo 0.25.0のセル依存関係](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/reactivity.md)
- [marimo 0.25.0の実行設定](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/configuration/runtime_configuration.md)
- [marimo 0.25.0の重いNotebookの扱い](https://github.com/marimo-team/marimo/blob/0.25.0/docs/guides/expensive_notebooks.md)
