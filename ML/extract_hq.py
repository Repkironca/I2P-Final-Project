import pandas as pd
import numpy as np
import math
import os

# 定義棋子價值代號（正數我方，負數敵方）
P, N, B, R, Q, K = 1, 2, 3, 4, 5, 6

class MiniChessSimulator:
    def __init__(self):
        self.board = np.zeros((6, 5), dtype=int)
        self.reset()

    def reset(self):
        self.board[:] = 0
        self.board[0, :] = [R, N, B, Q, K] # 白方底線 (王在最右邊)
        self.board[1, :] = P               # 白方兵線
        self.board[4, :] = -P              # 黑方兵線
        self.board[5, :] = [-K, -Q, -B, -N, -R] # 黑方底線

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
        # 視角轉換：只做前後（上下）翻轉與正負號反轉，絕對不進行左右鏡像！
        # 這樣能精準保留左側(車馬翼)與右側(王翼)的不對稱戰略價值。
        perspective_board = self.board.copy() if is_white_turn else -np.flipud(self.board)
        
        # --------------------------------------------------
        # V1: 子力差 (5維)
        # --------------------------------------------------
        V1 = np.zeros(5)
        for idx, p_type in enumerate([P, N, B, R, Q]):
            V1[idx] = np.sum(perspective_board == p_type) - np.sum(perspective_board == -p_type)

        # --------------------------------------------------
        # V2: 敵方國王距離和 (5維)
        # --------------------------------------------------
        V2 = np.zeros(5)
        king_pos = np.argwhere(perspective_board == -K)
        if len(king_pos) > 0:
            kr, kc = king_pos[0]
            for idx, p_type in enumerate([P, N, B, R, Q]):
                friendly_positions = np.argwhere(perspective_board == p_type)
                if len(friendly_positions) > 0:
                    V2[idx] = np.sum(np.abs(friendly_positions[:, 0] - kr) + np.abs(friendly_positions[:, 1] - kc))

        # --------------------------------------------------
        # 🌟 V3: 180維獨立大腦 (6種棋子 x 30格)
        # --------------------------------------------------
        V3_list = []
        for p_type in [P, N, B, R, Q, K]:
            piece_mask = np.zeros((6, 5))
            piece_mask[perspective_board == p_type] = 1   # 我方該棋子所在格
            piece_mask[perspective_board == -p_type] = -1 # 敵方該棋子所在格
            V3_list.append(piece_mask.flatten())
        V3 = np.concatenate(V3_list)

        # --------------------------------------------------
        # a1: 小兵推進加分 (1維)
        # --------------------------------------------------
        adv_friendly = sum(r - 1 for r in range(6) for c in range(5) if perspective_board[r, c] == P)
        adv_enemy = sum(4 - r for r in range(6) for c in range(5) if perspective_board[r, c] == -P)
        a1 = adv_friendly - adv_enemy

        # 總共 5 + 5 + 180 + 1 = 191 維特徵
        return V1, V2, V3, a1

def main():
    if not os.path.exists('dataset_hq.csv'):
        print("錯誤：找不到高品質資料集 dataset_hq.csv！")
        return

    all_features = []
    simulator = MiniChessSimulator()
    df = pd.read_csv('dataset_hq.csv')
    print(f"開始處理高品質資料集 (共 {len(df)} 局)...")

    for index, row in df.iterrows():
        result = float(row['Result'])
        moves = str(row['Moves']).split()
        total_moves = len(moves)
        if total_moves == 0: continue
        
        base_weight = 0.2 if result == 0.5 else 1.0
        simulator.reset()
        seen_positions = set()
        
        for i, move in enumerate(moves):
            move = move.strip()
            if len(move) < 4: continue
                
            is_white_turn = (i % 2 == 0)
            board_hash = simulator.board.tobytes()
            
            if board_hash not in seen_positions:
                seen_positions.add(board_hash)
                
                # 萃取 191 維終極不對稱特徵
                V1, V2, V3, a1 = simulator.extract_features(is_white_turn)
                feature_vector = np.concatenate([V1, V2, V3, [a1]])
                
                label = result if is_white_turn else (1.0 - result)
                
                # 🌟 恢復經典的開根號時間衰減權重
                progress_ratio = (i + 1) / total_moves
                time_weight = math.sqrt(progress_ratio)
                sample_weight = base_weight * time_weight
                
                row_data = np.append(feature_vector, [label, sample_weight])
                all_features.append(row_data)
                
            simulator.push_move(move)

    # 建立 CSV 欄位標籤
    pieces_names = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
    v3_cols = [f'v3_{p}_{i}' for p in pieces_names for i in range(30)]
    columns = ([f'v1_{i}' for i in range(5)] + [f'v2_{i}' for i in range(5)] + 
               v3_cols + ['a1', 'label', 'weight'])
    
    new_df = pd.DataFrame(all_features, columns=columns)
    new_df.to_csv('processed_dataset_hq_only.csv', index=False)
    print(f"✅ 特徵萃取完畢！產出樣本數: {len(new_df)} 步，已存至 processed_dataset_hq_only.csv")

if __name__ == "__main__":
    main()