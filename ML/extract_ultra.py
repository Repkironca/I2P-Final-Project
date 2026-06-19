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
        self.board[0, :] = [R, N, B, Q, K] # 白方底線
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

    def get_pseudo_moves(self, p_board):
        """ 在獨立視角棋盤上生成我方(正數)的所有偽合法步 """
        moves = []
        for r in range(6):
            for c in range(5):
                piece = p_board[r, c]
                if piece <= 0: continue
                
                if piece == P: # 兵 (前進與斜吃)
                    if r + 1 < 6 and p_board[r + 1, c] == 0:
                        moves.append((r + 1, c))
                    for dc in [-1, 1]:
                        if r + 1 < 6 and 0 <= c + dc < 5 and p_board[r + 1, c + dc] < 0:
                            moves.append((r + 1, c + dc))
                            
                elif piece == N: # 馬
                    offsets = [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]
                    for dr, dc in offsets:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 6 and 0 <= nc < 5 and p_board[nr, nc] <= 0:
                            moves.append((nr, nc))
                            
                elif piece == B or piece == Q: # 象、后 (斜向滑行)
                    dirs = [(-1,-1), (-1,1), (1,-1), (1,1)]
                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        while 0 <= nr < 6 and 0 <= nc < 5:
                            if p_board[nr, nc] == 0: moves.append((nr, nc)); nr += dr; nc += dc
                            elif p_board[nr, nc] < 0: moves.append((nr, nc)); break
                            else: break
                            
                if piece == R or piece == Q: # 車、后 (直線滑行)
                    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        while 0 <= nr < 6 and 0 <= nc < 5:
                            if p_board[nr, nc] == 0: moves.append((nr, nc)); nr += dr; nc += dc
                            elif p_board[nr, nc] < 0: moves.append((nr, nc)); break
                            else: break
                            
                elif piece == K: # 王
                    dirs = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 6 and 0 <= nc < 5 and p_board[nr, nc] <= 0:
                            moves.append((nr, nc))
        return moves

    def extract_ultra_features(self, is_white_turn, step):
        # 絕對不進行左右翻轉，精準保留不對稱生態
        p_board = self.board.copy() if is_white_turn else -np.flipud(self.board)
        
        # 1. V1: 子力差 (5維)
        V1 = np.zeros(5)
        for idx, p_type in enumerate([P, N, B, R, Q]):
            V1[idx] = np.sum(p_board == p_type) - np.sum(p_board == -p_type)

        # 2. V2: 敵方國王距離和 (5維)
        V2 = np.zeros(5)
        king_pos = np.argwhere(p_board == -K)
        if len(king_pos) > 0:
            kr, kc = king_pos[0]
            for idx, p_type in enumerate([P, N, B, R, Q]):
                friendly_positions = np.argwhere(p_board == p_type)
                if len(friendly_positions) > 0:
                    V2[idx] = np.sum(np.abs(friendly_positions[:, 0] - kr) + np.abs(friendly_positions[:, 1] - kc))

        # 3. V3: 180維度獨立 PST (6種棋子 x 30格)
        V3_list = []
        for p_type in [P, N, B, R, Q, K]:
            piece_mask = np.zeros((6, 5))
            piece_mask[p_board == p_type] = 1
            piece_mask[p_board == -p_type] = -1
            V3_list.append(piece_mask.flatten())
        V3 = np.concatenate(V3_list)

        # 🚀 4. V4: 戰術威脅特徵 (6維)
        pseudo_moves = self.get_pseudo_moves(p_board)
        v4_check = 0
        v4_captures = {1:0, 2:0, 3:0, 4:0, 5:0} # 兵馬象車后
        
        for to_r, to_c in pseudo_moves:
            target = p_board[to_r, to_c]
            if target == -K:
                v4_check = 1
            elif -Q <= target <= -P:
                v4_captures[abs(target)] += 1
                
        V4 = np.array([v4_check, v4_captures[1], v4_captures[2], v4_captures[3], v4_captures[4], v4_captures[5]])

        # 🚀 5. V5: 階層小兵升變 (改為完整的 4 維！)
        # 我方視角下，兵永遠在 row 1 出生，往 row 5 升變
        v5_step1 = np.sum(p_board[4, :] == P) # 在第 4 列，離升變剩 1 步
        v5_step2 = np.sum(p_board[3, :] == P) # 在第 3 列，離升變剩 2 步
        v5_step3 = np.sum(p_board[2, :] == P) # 在第 2 列，離升變剩 3 步
        v5_step4 = np.sum(p_board[1, :] == P) # 在第 1 列，離升變剩 4 步 (初始位置)
        V5 = np.array([v5_step1, v5_step2, v5_step3, v5_step4])

        # 🚀 6. V6: 零成本行動力差 (1維)
        V6 = np.array([len(pseudo_moves)])

        # 🚀 7. V7: 國王時序交互特徵 (30維)
        V7_mask = np.zeros((6, 5))
        V7_mask[p_board == K] = float(step)
        V7_mask[p_board == -K] = -float(step)
        V7 = V7_mask.flatten()

        # 總共 5 + 5 + 180 + 6 + 4 + 1 + 30 = 231 維特徵
        return np.concatenate([V1, V2, V3, V4, V5, V6, V7])

def main():
    if not os.path.exists('dataset_hq.csv'):
        print("錯誤：找不到高品質資料集 dataset_hq.csv！")
        return

    all_features = []
    simulator = MiniChessSimulator()
    df = pd.read_csv('dataset_hq.csv')
    print(f"啟動 Ultra 特徵探針，正在精煉資料 (共 {len(df)} 局)...")

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
                
                current_step = i + 1
                feature_vector = simulator.extract_ultra_features(is_white_turn, current_step)
                
                label = result if is_white_turn else (1.0 - result)
                
                progress_ratio = current_step / total_moves
                time_weight = math.sqrt(progress_ratio)
                sample_weight = base_weight * time_weight
                
                row_data = np.append(feature_vector, [label, sample_weight])
                all_features.append(row_data)
                
            simulator.push_move(move)

    # 動態建立 231 維欄位標籤
    pieces = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
    v3_cols = [f'v3_{p}_{i}' for p in pieces for i in range(30)]
    v4_cols = ['v4_check', 'v4_cap_p', 'v4_cap_n', 'v4_cap_b', 'v4_cap_r', 'v4_cap_q']
    v5_cols = ['v5_pawn_step1', 'v5_pawn_step2', 'v5_pawn_step3', 'v5_pawn_step4']
    v7_cols = [f'v7_king_step_{i}' for i in range(30)]
    
    columns = ([f'v1_{i}' for i in range(5)] + [f'v2_{i}' for i in range(5)] + 
               v3_cols + v4_cols + v5_cols + ['v6_mobility'] + v7_cols + ['label', 'weight'])
    
    new_df = pd.DataFrame(all_features, columns=columns)
    new_df.to_csv('processed_dataset_ultra.csv', index=False)
    print(f"\n✅ Ultra 特徵工廠運作完畢！")
    print(f"產出終局高維矩陣: {new_df.shape[0]} 列 x {new_df.shape[1]} 欄位 (231維特徵 + label + weight)")
    print("已成功存至 processed_dataset_ultra.csv！")

if __name__ == "__main__":
    main()