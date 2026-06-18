import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

print("正在讀取清洗好的資料...")
df = pd.read_csv('processed_dataset.csv')

# =====================================================================
# 🌟 【遞進式訓練魔法】：固定已知知識 (Anchoring)
# =====================================================================
# 我們不准模型亂動兵、馬、象、車、后的基礎價值，避免它被舊 AI 的壞習慣帶偏。
# 這裡填入你在 C++ duckyQuack_v4.hpp 裡設定的 piece_values (除了國王)
fixed_u1_cpp = np.array([10, 30, 30, 50, 90])  # P, N, B, R, Q

# 由於我們希望最後產出的 C++ 整數係數是放大 1000 倍的結果，
# 所以在 ML 的世界裡 (勝率是 0.0 ~ 1.0)，這些固定價值必須先縮小 1000 倍。
SCALE_FACTOR = 1000.0
fixed_u1_ml = fixed_u1_cpp / SCALE_FACTOR

# 切割資料：把 V1 (子力數量差) 和其他特徵分開
X_V1 = df.iloc[:, 0:5]   # 前 5 個欄位是 V1
X_rest = df.iloc[:, 5:42] # 後面 37 個欄位是 V2, V3, a1, a2
y = df['label']
weights = df['weight']

# 🔥 核心數學移項：修正目標 (Target Modification)
# 把我們「已經知道的分數」從標準答案中扣除。
# 模型接下來的任務，就只剩下「預測那些子力價值無法解釋的勝率差距」。
y_adjusted = y - np.dot(X_V1, fixed_u1_ml)


# =====================================================================
# 🌟 【防呆魔法】：Ridge Regression (脊迴歸 / L2 正規化)
# =====================================================================
print("啟動 Ridge Regression (脊迴歸) 引擎...")
# alpha 是懲罰力度 (Regularization Strength)。
# 當 alpha > 0 時，模型在計算分數時會受到一個無形的壓力：「係數絕對值不准太大！」
# 這能完美解決「特徵互相打架 (例如把車的分數藏進距離裡)」的共線性問題。
# 如果你覺得算出來的數字還是太大，可以把 alpha 調成 500.0 或 1000.0；若太小則調成 10.0。
model = Ridge(alpha=100.0, fit_intercept=False)

# 讓模型開始學習！(注意：我們是用 X_rest 和修正後的 y_adjusted 來訓練)
model.fit(X_rest, y_adjusted, sample_weight=weights)


# =====================================================================
# 結算與轉換係數
# =====================================================================
# 把模型新學到的 37 個係數放大 1000 倍，並轉成整數
coefs_rest = np.round(model.coef_ * SCALE_FACTOR).astype(int)

# 把固定不變的基礎價值，以及新學到的係數切分開來
u1 = fixed_u1_cpp        # V1 (直接沿用我們固定的數字)
u2 = coefs_rest[0:5]     # V2 (敵王距離：5 維)
u3 = coefs_rest[5:35]    # V3 (PST 棋盤位置：30 維)
k1 = coefs_rest[35]      # a1 (小兵推進)
k2 = coefs_rest[36]      # a2 (通路兵)

print("\n" + "="*50)
print("🎯 遞進式訓練大功告成！這組數據絕對正常多了：")
print("="*50)

print(f"u1 (V1 兵,馬,象,車,后 基礎價值) = {list(u1)}  <-- 這是你強迫它記住的")
print(f"u2 (V2 距離敵王威脅權重) = {list(u2)}")
print(f"k1 (a1 小兵推進加分) = {k1}")
print(f"k2 (a2 通路兵加分) = {k2}")

print("\n--------------------------------------------------")
print("u3 (V3 PST 棋盤格位戰略分數, 5x6 陣列):")
for r in range(6):
    print("    " + ", ".join(f"{val:4d}" for val in u3[r*5 : (r+1)*5]) + ",")
print("="*50)

# 計算 R^2 Score (注意：這裡是看「剩餘特徵」對「勝率殘差」的解釋能力)
score = model.score(X_rest, y_adjusted, sample_weight=weights)
print(f"\n[進階指標] 剩餘特徵擬合度 (R^2 Score): {score:.4f}")
print("*(註：因為我們已經拔掉了最具決定性的『子力數量』特徵，所以這裡的 R^2 會比之前更小，這是完全正常的！)*")