import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

# 🌟 你可以在這裡調整 Ridge 的橡皮筋強度 (懲罰項)
# 數字越大，越能壓制 V2 的極端共線數值，並迫使權重吐回給 U1 的基礎價值。
ALPHA = 1000 

def main():
    print("正在讀取高品質特徵考卷...")
    try:
        df = pd.read_csv('processed_dataset_hq_only.csv')
    except FileNotFoundError:
        print("錯誤：找不到 processed_dataset_hq_only.csv，請先執行 extract_new.py")
        return

    # 切割題目與答案
    X = df.drop(columns=['label', 'weight'])
    y = df['label']
    weights = df['weight']

    print(f"啟動 Ridge 引擎 (完全解除 U1 訓練限制，Alpha = {ALPHA})...")
    # 西洋棋為完全對稱賽制，不加常數項 (fit_intercept=False)
    model = Ridge(alpha=ALPHA, fit_intercept=False)
    model.fit(X, y, sample_weight=weights)

    r2 = model.score(X, y, sample_weight=weights)

    # 四捨五入並放大 1000 倍給 C++ 整數引擎
    SCALE_FACTOR = 1000  
    coefs = np.round(model.coef_ * SCALE_FACTOR).astype(int)

    # 解析 191 維特徵係數
    u1 = coefs[0:5]      # V1 (基礎價值)
    u2 = coefs[5:10]     # V2 (敵王距離)
    u3_all = coefs[10:190] # V3 (180維 獨立PST)
    k1 = coefs[190]      # a1 (小兵推進)

    print("\n" + "="*50)
    print("🎯 訓練完成！180維度獨立大腦還原版參數：")
    print("="*50)
    print(f"模型擬合度 (R^2 Score): {r2:.4f}\n")
    print(f"// u1 (V1 兵,馬,象,車,后 基礎價值)")
    print(f"int u1[5] = {{{u1[0]}, {u1[1]}, {u1[2]}, {u1[3]}, {u1[4]}}};\n")
    print(f"// u2 (V2 距離敵王威脅權重)")
    print(f"int u2[5] = {{{u2[0]}, {u2[1]}, {u2[2]}, {u2[3]}, {u2[4]}}};\n")
    print(f"// k1 (a1 小兵推進加分)")
    print(f"int k1 = {k1};\n")
    print("-" * 50)
    print("// u3 (V3 6套獨立棋子 PST 戰略分數, 5x6 陣列)")
    print("-" * 50)

    pieces_names = ['Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King']
    
    for idx, name in enumerate(pieces_names):
        start = idx * 30
        end = start + 30
        # 將該棋子的 30 維特徵還原回 6列 x 5行 的棋盤格
        p_matrix = u3_all[start:end].reshape(6, 5)
        
        print(f"\n// {name} 專屬 PST 陣列")
        print(f"int u3_{name.lower()}[30] = {{")
        for r in range(6):
            row_str = ", ".join(f"{p_matrix[r, c]:>4}" for c in range(5))
            # 加上逗號結尾，最後一行也補上方便排版
            print(f"    {row_str},")
        print("};")
        
    print("="*50)

if __name__ == "__main__":
    main()