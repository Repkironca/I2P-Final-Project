import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
import os

def main():
    csv_filename = 'processed_dataset_ultra.csv'
    if not os.path.exists(csv_filename):
        print(f"錯誤：找不到特徵矩陣 {csv_filename}，請先執行 extract_ultra.py！")
        return

    print(f"正在載入 Ultra 高維特徵考卷 ({csv_filename})...")
    df = pd.read_csv(csv_filename)

    y = df['label'].values
    weights = df['weight'].values

    # 自動識別所有欄位群組
    all_cols = list(df.columns)
    v2_cols = [c for c in all_cols if c.startswith('v2_')]
    v3_cols = [c for c in all_cols if c.startswith('v3_')]
    v4_cols = [c for c in all_cols if c.startswith('v4_')]
    v5_cols = [c for c in all_cols if c.startswith('v5_')]
    v6_cols = [c for c in all_cols if c.startswith('v6_')]
    v7_cols = [c for c in all_cols if c.startswith('v7_')]

    # 🛠️ 戰略部署：全面釋放！除 V1 歸零外，其餘 V2~V7 全部參戰
    feature_cols = v2_cols + v3_cols + v4_cols + v5_cols + v6_cols + v7_cols
    X = df[feature_cols].copy().values

    print(f"📊 [終極實驗] V1 歸零，V2~V7 全面集結！當前訓練特徵維度: {X.shape[1]} 維")

    # 手動領域縮放：對大尺度特徵 (V2, V6, V7) 進行防禦性配平
    scales = np.ones(X.shape[1])
    for idx, col in enumerate(feature_cols):
        if col in v2_cols or col in v6_cols or col in v7_cols:
            X[:, idx] /= 10.0
            scales[idx] = 10.0

    # 🛠️ 釋放瘋狂網格：從 10^-3 到 10^3 密探 300 組不同的懲罰力度！
    alphas_to_test = np.logspace(-3, 3, 300)
    print(f"🏋️‍♂️ 正在啟動 300 道 Alpha 網格雷達掃描 (範圍: {alphas_to_test[0]:.4f} ~ {alphas_to_test[-1]:.1f})...")
    
    model = RidgeCV(alphas=alphas_to_test, fit_intercept=True)
    model.fit(X, y, sample_weight=weights)

    print(f"\n🎯 【最佳 Alpha 誕生】 = {model.alpha_:.4f}")
    r2_score = model.score(X, y, sample_weight=weights)
    print(f"📈 【終極極限擬合度】 R^2 Score: {r2_score:.5f}")

    # 尺度逆向還原
    raw_coefs = model.coef_ / scales
    raw_intercept = model.intercept_

    # 放大 1000 倍轉化為 C++ 純整數
    SCALE_FACTOR = 1000
    coef_dict = {col: raw_coefs[i] for i, col in enumerate(feature_cols)}

    print(f"\n=======================================================")
    print(f"   🎉 C++ 原始碼專用陣列產生器 (全特徵網格極限版)   ")
    print(f"=======================================================")
    print(f"// 請直接複製以下程式碼至 duckyQuack_v7.cpp 中\n")

    # 聽從指揮：V1 強制歸零
    print(f"// V1: 子力差距權重 (已完美歸零核銷，完全由 V3 PST 承接靈魂)")
    print(f"const int u1[5] = {{0, 0, 0, 0, 0}};")

    # V2: 敵王距離和 (5維)
    cpp_u2 = [int(np.round(coef_dict[f'v2_{i}'] * SCALE_FACTOR)) for i in range(5)]
    print(f"\n// V2: 敵王曼哈頓距離和權重 (順序: 兵, 馬, 象, 車, 后)")
    print(f"const int u2[5] = {{{', '.join(map(str, cpp_u2))}}};")

    # V3: 靜態 PST 陣地戰大腦 (180維)
    print(f"\n// V3: PST 靜態位階權重")
    pieces = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
    for p in pieces:
        cpp_u3 = [int(np.round(coef_dict[f'v3_{p}_{i}'] * SCALE_FACTOR)) for i in range(30)]
        print(f"const int u3_{p}[30] = {{{', '.join(map(str, cpp_u3))}}};")

    # V4: 立即戰術威脅權重 (6維)
    v4_names = ['v4_check', 'v4_cap_p', 'v4_cap_n', 'v4_cap_b', 'v4_cap_r', 'v4_cap_q']
    cpp_u4 = [int(np.round(coef_dict[name] * SCALE_FACTOR)) for name in v4_names]
    print(f"\n// V4: 立即戰術威脅權重 (順序: Check, 吃兵, 吃馬, 吃象, 吃車, 吃后)")
    print(f"const int u4_tactical[6] = {{{', '.join(map(str, cpp_u4))}}};")

    # V5: 4階段完整升變階梯 (4維)
    v5_names = ['v5_pawn_step1', 'v5_pawn_step2', 'v5_pawn_step3', 'v5_pawn_step4']
    cpp_u5 = [int(np.round(coef_dict[name] * SCALE_FACTOR)) for name in v5_names]
    print(f"\n// V5: 完整小兵升變階梯權重 (順序: 剩1步, 剩2步, 剩3步, 剩4步)")
    print(f"const int u5_pawn_stages[4] = {{{', '.join(map(str, cpp_u5))}}};")

    # V6: 零成本己方行動力 (1維)
    cpp_u6 = int(np.round(coef_dict['v6_mobility'] * SCALE_FACTOR))
    print(f"\n// V6: 零成本己方行動力權重 (直接乘以合法步數)")
    print(f"const int u6_mobility = {cpp_u6};")

    # V7: 國王時序交互動態特徵 (30維)
    cpp_u7 = [int(np.round(coef_dict[f'v7_king_step_{i}'] * SCALE_FACTOR)) for i in range(30)]
    print(f"\n// V7: 國王時序動態格位權重 (在 C++ 中需乘以當前 step)")
    print(f"const int u7_king_step[30] = {{{', '.join(map(str, cpp_u7))}}};")

    # 基礎偏置項
    cpp_intercept = int(np.round(raw_intercept * SCALE_FACTOR))
    print(f"\n// 盤面基礎偏置分數 (Intercept)")
    print(f"const int u_intercept = {cpp_intercept};")
    print(f"\n=======================================================")

if __name__ == '__main__':
    main()