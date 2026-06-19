import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

print("正在讀取包含 8 倍權重 HQ 資料的考卷...")
df = pd.read_csv('processed_dataset_mixed.csv')

# =====================================================================
# 1. 切割資料 (41維特徵)
# =====================================================================
# a2 已經在 extract 階段拔除了，所以直接全拿
X = df.drop(columns=['label', 'weight'])
y = df['label']
weights = df['weight']

# =====================================================================
# 2. 🌟 特徵標準化 (Standardization)
# =====================================================================
print("正在對特徵進行標準化縮放 (避免 V2 數值過大影響模型)...")
# with_mean=False 保留特徵的正負號(敵我關係)，只除以標準差縮小數值尺度。
# 這樣 V2 (距離和動輒 30) 和 V1 (子力動輒 1) 就會被壓到同樣的起跑線！
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)

# =====================================================================
# 3. 機器學習訓練 (全解鎖版)
# =====================================================================
print("啟動 Ridge 引擎，並完全解鎖 U1 的訓練限制...")
# 這次我們不固定 U1 了，讓含有 v5, Lucifer, BOSS 這些高手的 HQ 數據，
# 親自告訴我們「車」跟「主教」的真實價值。
model = Ridge(alpha=500.0, fit_intercept=False)
model.fit(X_scaled, y, sample_weight=weights)

# =====================================================================
# 4. 🌟 權重還原與 C++ 整數化魔法
# =====================================================================
# 因為剛才 X 被縮小了，算出來的權重是「對應縮小版 X 的權重」。
# 為了讓你的 C++ 程式碼 0 改動，我們要在 Python 裡把權重「除回原比例」！
raw_coefs = model.coef_ / scaler.scale_

# 統一乘上 1000 倍，達成 C++ 裡「免 Float 的小數點後三位精準度」
SCALE_FACTOR = 1000.0
final_coefs = np.round(raw_coefs * SCALE_FACTOR).astype(int)

# 依序把這 41 個數字切開 (再也沒有 a2 了)
u1 = final_coefs[0:5]    # V1: 兵, 馬, 象, 車, 后
u2 = final_coefs[5:10]   # V2: 距離和
u3 = final_coefs[10:40]  # V3: PST 棋盤位置
k1 = final_coefs[40]     # a1: 小兵推進

# =====================================================================
# 5. 產出給鴨子的神級參數
# =====================================================================
print("\n" + "="*50)
print("🎯 訓練完成！混合高強度資料 + 標準化還原版參數：")
print("="*50)

print(f"u1 (V1 兵,馬,象,車,后 基礎價值) = {list(u1)}")
print(f"u2 (V2 距離敵王威脅權重) = {list(u2)}")
print(f"k1 (a1 小兵推進加分) = {k1}")
print("(a2 通路兵已被捨棄，以換取 C++ 搜尋深度！)")

print("\n--------------------------------------------------")
print("u3 (V3 PST 棋盤格位戰略分數, 5x6 陣列):")
for r in range(6):
    print("    " + ", ".join(f"{val:4d}" for val in u3[r*5 : (r+1)*5]) + ",")
print("="*50)

score = model.score(X_scaled, y, sample_weight=weights)
print(f"\n[模型擬合度 (R^2 Score): {score:.4f}]")