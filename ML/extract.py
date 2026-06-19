import pandas as pd
import numpy as np
import math
import os

P, N, B, R, Q, K = 1, 2, 3, 4, 5, 6

class MiniChessSimulator:
    def __init__(self):
        self.board = np.zeros((6, 5), dtype=int)
        self.reset()

    def reset(self):
        self.board[:] = 0
        self.board[0, :] = [R, N, B, Q, K]
        self.board[1, :] = P
        self.board[4, :] = -P
        self.board[5, :] = [-K, -Q, -B, -N, -R]

    def parse_move(self, move_str):
        return int(move_str[1])-1, ord(move_str[0])-ord('a'), int(move_str[3])-1, ord(move_str[2])-ord('a')

    def push_move(self, move_str):
        from_r, from_c, to_r, to_c = self.parse_move(move_str)
        piece = self.board[from_r, from_c]
        self.board[from_r, from_c] = 0
        self.board[to_r, to_c] = piece
        if piece == P and to_r == 5: self.board[to_r, to_c] = Q
        elif piece == -P and to_r == 0: self.board[to_r, to_c] = -Q

    def extract_features(self, is_white_turn):
        side = 1 if is_white_turn else -1
        
        # V1: 子力數量差
        V1 = np.zeros(5)
        for idx, p_type in enumerate([P, N, B, R, Q]):
            V1[idx] = np.sum(self.board == (p_type * side)) - np.sum(self.board == (-p_type * side))

        # V2: 敵方國王距離和 (先保留給模型標準化用)
        V2 = np.zeros(5)
        king_pos = np.argwhere(self.board == -K * side)
        if len(king_pos) > 0:
            kr, kc = king_pos[0]
            for idx, p_type in enumerate([P, N, B, R, Q]):
                friendly_positions = np.argwhere(self.board == (p_type * side))
                if len(friendly_positions) > 0:
                    V2[idx] = np.sum(np.abs(friendly_positions[:, 0] - kr) + np.abs(friendly_positions[:, 1] - kc))

        # V3: PST 棋盤分佈
        perspective_board = self.board.copy() if is_white_turn else -np.flipud(self.board)
        V3 = perspective_board.flatten()

        # a1: 兵的推進加分 (徹底拔除 a2 通路兵計算，省下時間！)
        w_adv = sum(r - 1 for r in range(6) for c in range(5) if self.board[r, c] == P)
        b_adv = sum(4 - r for r in range(6) for c in range(5) if self.board[r, c] == -P)
        a1 = (w_adv - b_adv) if is_white_turn else (b_adv - w_adv)

        # 現在特徵只剩下 V1(5) + V2(5) + V3(30) + a1(1) = 41 維
        return V1, V2, V3, a1

def process_file(filepath, simulator, all_features, is_hq=False):
    if not os.path.exists(filepath):
        print(f"找不到檔案: {filepath}，跳過。")
        return

    df = pd.read_csv(filepath)
    print(f"正在處理 {filepath} (總局數: {len(df)}，HQ模式: {is_hq})...")

    for index, row in df.iterrows():
        result = float(row['Result'])
        moves = str(row['Moves']).split()
        total_moves = len(moves)
        if total_moves == 0: continue
        
        # 🌟【全新權重演算法】：你指定的 1, 0.2, 乘 8 邏輯！
        # 舊版且有勝負: 1.0，沒有勝負: 0.2
        base_weight = 0.2 if result == 0.5 else 1.0
        
        # 新版 (HQ) 的權重統一乘 8
        if is_hq:
            base_weight *= 8.0
            
        simulator.reset()
        seen_positions = set()
        
        for i, move in enumerate(moves):
            move = move.strip()
            if len(move) < 4: continue # 防呆
                
            is_white_turn = (i % 2 == 0)
            board_hash = simulator.board.tobytes()
            
            if board_hash not in seen_positions:
                seen_positions.add(board_hash)
                
                # 萃取 41 維特徵 (沒有 a2 了)
                V1, V2, V3, a1 = simulator.extract_features(is_white_turn)
                feature_vector = np.concatenate([V1, V2, V3, [a1]])
                label = result if is_white_turn else (1.0 - result)
                
                # 結合開根號時間衰減
                progress_ratio = (i + 1) / total_moves
                time_weight = math.sqrt(progress_ratio)
                sample_weight = base_weight * time_weight
                
                row_data = np.append(feature_vector, [label, sample_weight])
                all_features.append(row_data)
                
            simulator.push_move(move)

def main():
    all_features = []
    simulator = MiniChessSimulator()
    
    # 處理高品質資料集 (is_hq=True)
    process_file('dataset_hq.csv', simulator, all_features, is_hq=True)
    # 處理舊版資料集 (is_hq=False)
    process_file('dataset.csv', simulator, all_features, is_hq=False)
    
    # a2 已經移除了，欄位總共 41 個特徵 + label + weight
    columns = ([f'v1_{i}' for i in range(5)] + [f'v2_{i}' for i in range(5)] + 
               [f'v3_{i}' for i in range(30)] + ['a1', 'label', 'weight'])
    
    new_df = pd.DataFrame(all_features, columns=columns)
    new_df.to_csv('processed_dataset_mixed.csv', index=False)
    print(f"\n✅ 雙資料集轉換完成！已儲存至 processed_dataset_mixed.csv")

if __name__ == "__main__":
    main()