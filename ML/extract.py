import pandas as pd
import numpy as np
import math

# 【定義棋子價值代號】
# 在機器學習裡，我們喜歡用「正負號」來區分敵我，這可以讓數學運算變得很優雅。
# 正數 = 當前玩家 (我方)，負數 = 對手 (敵方)
P, N, B, R, Q, K = 1, 2, 3, 4, 5, 6

class MiniChessSimulator:
    def __init__(self):
        # 建立 6列(row) x 5行(col) 的二維矩陣
        self.board = np.zeros((6, 5), dtype=int)
        self.reset()

    def reset(self):
        """將棋盤重置為 5x6 迷你象棋的初始陣型"""
        self.board[:] = 0
        self.board[0, :] = [R, N, B, Q, K] # 白方底線
        self.board[1, :] = P               # 白方兵線
        self.board[4, :] = -P              # 黑方兵線
        self.board[5, :] = [-K, -Q, -B, -N, -R] # 黑方底線

    def parse_move(self, move_str):
        """將人類看的座標 (如 'c2c3') 轉成陣列的 (row, col) 索引"""
        return int(move_str[1])-1, ord(move_str[0])-ord('a'), int(move_str[3])-1, ord(move_str[2])-ord('a')

    def push_move(self, move_str):
        """在虛擬棋盤上推演這一步棋"""
        from_r, from_c, to_r, to_c = self.parse_move(move_str)
        piece = self.board[from_r, from_c]
        
        # 移動並吃子 (直接覆蓋目標格)
        self.board[from_r, from_c] = 0
        self.board[to_r, to_c] = piece
        
        # 【規則處理：小兵升變】走到對方底線就變成皇后
        if piece == P and to_r == 5: 
            self.board[to_r, to_c] = Q
        elif piece == -P and to_r == 0: 
            self.board[to_r, to_c] = -Q

    def extract_features(self, is_white_turn):
        """
        🔥 [核心靈魂] 特徵萃取器：找出決定勝負的關鍵變數
        """
        # side 是方向轉換器。白棋看自己是 1，黑棋看自己也是 1 (把敵方變成負的)。
        side = 1 if is_white_turn else -1
        
        # ----------------------------------------------------
        # V1: 子力數量差 (我方數量 - 敵方數量) [長度: 5]
        # 為什麼要算這個？告訴模型每種棋子的基本戰鬥力。
        # ----------------------------------------------------
        V1 = np.zeros(5)
        for idx, p_type in enumerate([P, N, B, R, Q]):
            # 數盤面上有幾顆符合條件的棋子
            V1[idx] = np.sum(self.board == (p_type * side)) - np.sum(self.board == (-p_type * side))

        # ----------------------------------------------------
        # V2: 敵王漢密頓距離和 [長度: 5]
        # 為什麼要算這個？教 AI 學會「包圍敵王」，離敵王越近越有威脅。
        # ----------------------------------------------------
        V2 = np.zeros(5)
        king_pos = np.argwhere(self.board == -K * side) # 找敵王座標
        if len(king_pos) > 0:
            kr, kc = king_pos[0]
            for idx, p_type in enumerate([P, N, B, R, Q]):
                friendly_positions = np.argwhere(self.board == (p_type * side))
                if len(friendly_positions) > 0:
                    # 漢密頓距離 (Manhattan Distance)：X座標差的絕對值 + Y座標差的絕對值
                    V2[idx] = np.sum(np.abs(friendly_positions[:, 0] - kr) + np.abs(friendly_positions[:, 1] - kc))

        # ----------------------------------------------------
        # V3: 棋盤格位分佈 (PST) [長度: 30]
        # 【ML 視角統一神技】：黑白共用特徵，防過擬合 (Overfitting)！
        # 如果是黑棋，我們把棋盤「上下顛倒 (np.flipud)」並「正負反轉」。
        # 這樣一來，AI 永遠覺得自己是從下面往上進攻，資料量等於翻倍！
        # ----------------------------------------------------
        perspective_board = self.board.copy() if is_white_turn else -np.flipud(self.board)
        V3 = perspective_board.flatten() # 把 6x5 壓平成 30 維的一維陣列

        # ----------------------------------------------------
        # a1 & a2: 戰術純量特徵 (小兵推進差 & 通路兵數量差)
        # 為什麼要特別抓小兵？因為西洋棋的殘局，小兵升變往往是唯一贏法。
        # ----------------------------------------------------
        w_adv = sum(r - 1 for r in range(6) for c in range(5) if self.board[r, c] == P)
        b_adv = sum(4 - r for r in range(6) for c in range(5) if self.board[r, c] == -P)
        a1 = (w_adv - b_adv) if is_white_turn else (b_adv - w_adv)

        w_pass, b_pass = 0, 0
        for r in range(6):
            for c in range(5):
                p = self.board[r, c]
                if abs(p) != P: continue
                
                is_passed = True
                cols = [c-1, c, c+1]
                rows = range(r+1, 6) if p == P else range(0, r)
                enemy = -P if p == P else P
                
                for tc in cols:
                    if 0 <= tc < 5 and any(self.board[tr, tc] == enemy for tr in rows):
                        is_passed = False; break
                if is_passed:
                    if p == P: w_pass += 1
                    else: b_pass += 1
                    
        a2 = (w_pass - b_pass) if is_white_turn else (b_pass - w_pass)

        return V1, V2, V3, a1, a2

def process_and_save_dataset(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    all_features = []
    simulator = MiniChessSimulator()
    
    print("開始將棋譜翻譯成 ML 數據，並啟動千日手過濾器...")
    for index, row in df.iterrows():
        result = float(row['Result'])
        moves = str(row['Moves']).split()
        total_moves = len(moves)
        if total_moves == 0: continue
        
        # 🌟 【權重魔法 1】: 和局價值貶值！
        # 如果是平手，基礎權重只剩 0.2。教 AI 專注於「會贏或會輸」的決策。
        base_weight = 0.2 if result == 0.5 else 1.0
        simulator.reset()
        
        # 🌟 【千日手剋星】: 盤面歷史記憶體
        # 使用 Python 的 set() (集合) 來記錄看過的盤面。尋找速度是 O(1)，超級快。
        seen_positions = set()
        
        for i, move in enumerate(moves):
            move = move.strip()
            # 🌟【資料清洗防呆機制】：長度不到 4 的絕對是髒數據（殘缺步數），直接跳過！
            if len(move) < 4: 
                continue
                
            is_white_turn = (i % 2 == 0)
            
            # 將目前的盤面陣列轉換成 Byte 字串 (Hash 的一種)，這代表當前唯一的盤面狀態
            board_hash = simulator.board.tobytes()
            
            # 如果這個盤面「沒看過」，我們才學習它！
            # (如果看過了，代表兩隻鴨子在跳恰恰、拖時間，我們直接無視，不放入考卷中)
            if board_hash not in seen_positions:
                seen_positions.add(board_hash) # 記到腦袋裡
                
                # 擷取 42 維特徵
                V1, V2, V3, a1, a2 = simulator.extract_features(is_white_turn)
                feature_vector = np.concatenate([V1, V2, V3, [a1, a2]])
                
                # 視角反轉標籤 (黑棋看白勝 1.0 = 黑負 0.0)
                label = result if is_white_turn else (1.0 - result)
                
                # 🌟 【權重魔法 2】: 開根號時間衰減！
                # 越接近 100 手終盤，這步棋對勝負的決定力越大。開根號可以畫出平滑上升的曲線。
                progress_ratio = (i + 1) / total_moves
                time_weight = math.sqrt(progress_ratio)
                
                sample_weight = base_weight * time_weight
                
                row_data = np.append(feature_vector, [label, sample_weight])
                all_features.append(row_data)
                
            # 推演下一步 (不管有沒有跳恰恰，棋局還是要繼續推演下去，不然盤面會亂掉)
            simulator.push_move(move)
            
    # 幫 CSV 加上漂亮的欄位名稱
    columns = ([f'v1_{i}' for i in range(5)] + [f'v2_{i}' for i in range(5)] + 
               [f'v3_{i}' for i in range(30)] + ['a1', 'a2', 'label', 'weight'])
    new_df = pd.DataFrame(all_features, columns=columns)
    new_df.to_csv(output_csv, index=False)
    print(f"資料轉換完成！已儲存至 {output_csv}")
    print(f"原本有 {total_moves * len(df)} 手棋，過濾千日手後剩餘 {len(new_df)} 筆有效學習資料。")

if __name__ == "__main__":
    process_and_save_dataset('dataset.csv', 'processed_dataset.csv')