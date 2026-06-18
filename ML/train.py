import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

print("正在讀取清洗好的資料...")
df = pd.read_csv('processed_dataset.csv')

# 【切割資料集】
# X 是考卷的題目 (42 維度的盤面特徵)
X = df.drop(columns=['label', 'weight'])
# y 是考卷的標準答案 (1.0 會贏, 0.5 和局, 0.0 會輸)
y = df['label']
# weights 是這題佔幾分 (靠近終盤的步數、有分出勝負的對局，分數佔比較重)
weights = df['weight']

print("啟動 Linear Regression (線性迴歸) 引擎...")
# 🌟 【ML 數學小知識】: 為什麼要 fit_intercept=False？
# 線性迴歸預設會加上一個常數項 (Intercept，即 y = ax + b 裡面的 b)。
# 但西洋棋是「完全對稱」的遊戲。在初始盤面 (X 幾乎等於 0 的狀態)，勝率理當是 50%。
# 如果允許模型產生 b，它可能會因為某方贏比較多盤，而產生無來由的偏見 (Bias)。
# 設定 False，等於強迫模型「只能從棋子的特徵去找勝負原因，不能瞎猜」。
model = LinearRegression(fit_intercept=False)

# 讓模型依照我們設計好的權重 (sample_weight) 進行訓練！
model.fit(X, y, sample_weight=weights)

# 【準備給 C++ 的禮物】
# 模型算出來的係數 (coef_) 會是極小的小數 (例如 0.021)。
# 但你在 C++ 裡是用「整數」算分數，因為整數運算比浮點數快非常多！
# 所以我們把小數統一乘上 1000 倍放大，並四捨五入成整數 (int)。
SCALE_FACTOR = 1000  
coefs = np.round(model.coef_ * SCALE_FACTOR).astype(int)

# 把 42 個數字切回它們原本對應的特徵意義
u1 = coefs[0:5]    # V1 (子力價值：兵, 馬, 象, 車, 后)
u2 = coefs[5:10]   # V2 (敵王距離：數值越小代表越不想靠近敵王，數值越大代表越喜歡包圍敵王)
u3 = coefs[10:40]  # V3 (PST：30 格的位置分數)
k1 = coefs[40]     # a1 (小兵每往前推進一格，加多少分)
k2 = coefs[41]     # a2 (擁有一隻強大的通路兵，加多少分)

print("\n" + "="*50)
print("🎯 訓練大功告成！請將以下陣列直接貼入你的 DuckyQuack_v4 評估函數：")
print("="*50)

print(f"u1 (V1 兵,馬,象,車,后 基礎價值) = {list(u1)}")
print(f"u2 (V2 距離敵王威脅權重) = {list(u2)}")
print(f"k1 (a1 小兵推進加分) = {k1}")
print(f"k2 (a2 通路兵加分) = {k2}")

print("\n--------------------------------------------------")
print("u3 (V3 PST 棋盤格位戰略分數, 5x6 陣列):")
# 為了讓你方便貼到 C++ 裡的二維陣列 (如 int pst[6][5])，我幫你排版成 6 行
for r in range(6):
    print("    " + ", ".join(f"{val:4d}" for val in u3[r*5 : (r+1)*5]) + ",")
print("="*50)

# 【驗收成果】R-squared (R平方) 是評估模型好壞的統計指標
# 它的範圍通常是 0 到 1。在棋類靜態評估中，通常落在 0.05 ~ 0.3 之間都是很正常的。
# (因為這只是「靜態評分」，還沒加上你強大的 C++ PVS 搜尋引擎)
score = model.score(X, y, sample_weight=weights)
print(f"\n[模型擬合度 (R^2 Score): {score:.4f}]")