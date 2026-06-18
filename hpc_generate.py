import subprocess
import random
import multiprocessing
import os
import sys

# 參賽選手陣容
PLAYERS = [
    {"name": "v2", "path": "build/minichess-ubgi", "algo": "duckyQuack_v2"},
    {"name": "v3", "path": "build/minichess-ubgi", "algo": "duckyQuack_v3"},
    {"name": "v4", "path": "build/minichess-ubgi", "algo": "duckyQuack_v4"},
    {"name": "Lucifer", "path": "build/Lucifer", "algo": "minimax"},
    {"name": "BOSS", "path": "build/boss-ubgi", "algo": "minimax"}
]

DATA_FILE = "dataset.csv"
DEPTHS = [6, 7, 8, 9] # 訓練資料多樣性：不同戰術深度
MAX_MOVES = 100 

class Engine:
    """封裝與 AI 執行檔的溝通管道"""
    def __init__(self, path, algo):
        self.proc = subprocess.Popen(
            [f"./{path}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
            stderr=subprocess.DEVNULL, text=True
        )
        self.send("ubgi")
        self.send(f"setoption name Algorithm value {algo}")
        self.send("isready")
        self.wait_for("readyok")

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def wait_for(self, target):
        while True:
            line = self.proc.stdout.readline().strip()
            if line == target or line == "ubgiok":
                break

    def get_move(self, moves_list, depth):
        if moves_list:
            self.send(f"position startpos moves {' '.join(moves_list)}")
        else:
            self.send("position startpos")
        
        self.send(f"go depth {depth}")

        while True:
            line = self.proc.stdout.readline().strip()
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) >= 2 else "none"

    def quit(self):
        self.send("quit")
        self.proc.terminate()

def play_game(task_id, w_idx, b_idx):
    w_info = PLAYERS[w_idx]
    b_info = PLAYERS[b_idx]
    
    # 1. 隨機開局 (前 4 步)：徹底打破 AI 的確定性
    moves = []
    rand_eng = Engine("build/minichess-ubgi", "random")
    for _ in range(4):
        mv = rand_eng.get_move(moves, depth=1)
        if mv != "none":
            moves.append(mv)
    rand_eng.quit()
    
    # 2. 為這盤棋隨機選定一個探索深度
    game_depth = random.choice(DEPTHS)
    
    # 3. 雙方實力 AI 接手盤面
    w_eng = Engine(w_info["path"], w_info["algo"])
    b_eng = Engine(b_info["path"], b_info["algo"])
    
    result = "0.5" # 預設平局 (步數超過上限)
    
    for ply in range(len(moves), MAX_MOVES):
        is_white_turn = (ply % 2 == 0)
        active_eng = w_eng if is_white_turn else b_eng
        
        bestmove = active_eng.get_move(moves, game_depth)
        
        # 遊戲結束條件
        if bestmove in ["none", "(none)", "0000", ""]:
            result = "0.0" if is_white_turn else "1.0"
            break
            
        moves.append(bestmove)

    w_eng.quit()
    b_eng.quit()
    
    game_str = f"{result},{' '.join(moves)}"
    print(f"Game {task_id:04d} | {w_info['name']} vs {b_info['name']} | Depth {game_depth} | Result: {result}", flush = true)
    return game_str

def worker(task):
    task_id, w_idx, b_idx = task
    try:
        return play_game(task_id, w_idx, b_idx)
    except Exception as e:
        print(f"Game {task_id} failed: {e}")
        return None

if __name__ == "__main__":
    tasks = []
    task_id = 0
    
    # 建立對戰任務列隊
    for w in range(len(PLAYERS)):
        for b in range(len(PLAYERS)):
            # 只要有 BOSS 參與的局，我們就收集 500 盤，其餘 200 盤
            is_boss_game = (PLAYERS[w]["name"] == "BOSS" or PLAYERS[b]["name"] == "BOSS")
            num_games = 500 if is_boss_game else 200
            for _ in range(num_games):
                tasks.append((task_id, w, b))
                task_id += 1
                
    # 打亂順序，避免一直看同兩台 AI 下棋
    random.shuffle(tasks)
    
    # 抓取 CPU 總核心數，全速運轉
    cores = multiprocessing.cpu_count()
    print(f"🚀 啟動單機練蠱！共 {len(tasks)} 盤任務，火力全開使用 {cores} 核心...")
    
    # 準備 CSV 檔案頭部
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w") as f:
            f.write("Result,Moves\n")

    # 啟動並行運算池
    with multiprocessing.Pool(cores) as pool:
        for data in pool.imap_unordered(worker, tasks):
            if data:
                # 每下一盤立刻寫入，隨時可以 Ctrl+C 中斷
                with open(DATA_FILE, "a") as f:
                    f.write(data + "\n")
                    
    print(f"✅ 訓練資料收集完畢！所有資料皆已儲存至 {DATA_FILE}")