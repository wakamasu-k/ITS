import json
import ast

# 読み込むファイルのパス
file_path = './output_dir/results_mcdrop.txt'

try:
    # ファイルを読み込む
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 文字列データをPythonの辞書（リスト）に変換
    try:
        data = json.loads(content.replace("'", '"'))
    except json.JSONDecodeError:
        data = ast.literal_eval(content)

    print("="*30)
    print(" 🏆 TULIP 最終評価スコア（平均値）")
    print("="*30)

    # 各指標（maeなど）のリストを取り出して平均を計算
    for metric_name, values_list in data.items():
        if isinstance(values_list, list) and len(values_list) > 0:
            # 平均の計算: 合計 ÷ 個数
            average = sum(values_list) / len(values_list)
            print(f"{metric_name.upper()} : {average:.5f}")
            
    print("="*30)

except FileNotFoundError:
    print("エラー: 評価結果のファイルが見つかりません。パスを確認してください。")
except Exception as e:
    print(f"エラーが発生しました: {e}")