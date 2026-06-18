import subprocess
import random
import multiprocessing
import os
import sys

# ==========================================
# 跨平台相容性設定
# ==========================================
IS_WINDOWS = sys.platform.startswith('win')
EXT = ".exe" if IS_WINDOWS else ""

# 定義高強度引擎與其路徑 (自動根據 OS 加上 .exe)
ENGINES = {
    "v5": {"path": f"build/minichess-ubgi{EXT}", "algo": "duckyQuack_v5"}, 
    "Lucifer": {"path": f"build/Lucifer{EXT}", "algo": "minimax"},
    "BOSS": {"path": f"build/boss-ubgi{EXT}", "algo": "minimax"}
}

# 指定的高端局對戰組合 (白 vs 黑)
MATCHUPS = [
    ("BOSS", "BOSS"),
    ("Lucifer", "BOSS"),
    ("BOSS", "Lucifer"),
    ("v5", "BOSS"),
    ("BOSS", "v5"),
    ("v5", "Lucifer"),
    ("Lucifer", "v5")
]

DATA_FILE = "dataset_hq.csv"
DEPTHS = [8, 9, 0] # 0 代表 movetime 2000
MAX_MOVES = 100 
GAMES_PER_MATCHUP = 500 # 每個組合各跑 500 盤，總計 3500 盤

class Engine:
    """封裝與 AI 執行檔的溝通管道"""
    def __init__(self, path, algo):
        # 跨平台處理路徑前綴 (Linux 需要 ./ 才能執行當前目錄檔案，Windows 則可以直接呼叫或用 .\)
        exe_cmd = path if IS_WINDOWS else f"./{path}"
        
        self.proc = subprocess.Popen(
            [exe_cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
            stderr=subprocess.DEVNULL, text=True
        )
        self.send("ubgi")
        self.send(f"setoption name Algorithm value {algo}")
        self.send("isready")
        self.wait_for("readyok")

    def send(self, cmd):
        if self.proc.poll() is not None:
            raise BrokenPipeError("引擎已崩潰")
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def wait_for(self, target):
        while True:
            line = self.proc.stdout.readline().strip()
            if not line: raise BrokenPipeError("引擎無回應")
            if line == target or line == "ubgiok":
                break

    def get_move(self, moves_list, depth):
        if moves_list:
            self.send(f"position startpos moves {' '.join(moves_list)}")
        else:
            self.send("position startpos")
        
        if depth == 0:
            self.send("go movetime 2000")
        else:
            self.send(f"go depth {depth}")

        while True:
            line = self.proc.stdout.readline().strip()
            if not line: raise BrokenPipeError("引擎無回應")
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) >= 2 else "none"

    def quit(self):
        try:
            self.send("quit")
            self.proc.terminate()
            self.proc.wait(timeout=1)
        except:
            self.proc.kill()

def play_game(task_id, w_name, b_name):
    w_info = ENGINES[w_name]
    b_info = ENGINES[b_name]
    
    # 1. 隨機開局 (前 4 步)
    moves = []
    rand_eng = Engine(f"build/minichess-ubgi{EXT}", "random")
    for _ in range(4):
        mv = rand_eng.get_move(moves, depth=1)
        if mv != "none":
            moves.append(mv)
    rand_eng.quit()
    
    # 2. 為這盤棋隨機選定一個探索深度/時間
    game_depth = random.choice(DEPTHS)
    
    # 3. 雙方實力 AI 接手盤面
    w_eng = Engine(w_info["path"], w_info["algo"])
    b_eng = Engine(b_info["path"], b_info["algo"])
    
    result = "0.5"
    try:
        for ply in range(len(moves), MAX_MOVES):
            is_white_turn = (ply % 2 == 0)
            active_eng = w_eng if is_white_turn else b_eng
            bestmove = active_eng.get_move(moves, game_depth)
            
            if bestmove in ["none", "(none)", "0000", ""]:
                result = "0.0" if is_white_turn else "1.0"
                break
            moves.append(bestmove)
    finally:
        w_eng.quit()
        b_eng.quit()
    
    depth_str = f"Time 2000ms" if game_depth == 0 else f"Depth {game_depth}"
    print(f"Game {task_id:04d} | {w_name} vs {b_name} | {depth_str} | Result: {result}", flush=True)
    
    # 輸出格式加入黑白雙方名稱
    return f"{result},{w_name},{b_name},{' '.join(moves)}"

def worker(task):
    task_id, w_name, b_name = task
    try:
        return play_game(task_id, w_name, b_name)
    except Exception as e:
        print(f"Game {task_id} failed: {e}", flush=True)
        return None

if __name__ == "__main__":
    # Windows 的 multiprocessing 需要這行保護，防止無限遞迴產生存取違規
    multiprocessing.freeze_support()
    
    tasks = []
    task_id = 0
    
    # 建立對戰任務列隊
    for w_name, b_name in MATCHUPS:
        for _ in range(GAMES_PER_MATCHUP):
            tasks.append((task_id, w_name, b_name))
            task_id += 1
                
    random.shuffle(tasks)
    
    cores = multiprocessing.cpu_count()
    print(f"🚀 啟動高端局練蠱！共 {len(tasks)} 盤，使用 {cores} 核心...", flush=True)
    
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w") as f:
            f.write("Result,White,Black,Moves\n")

    with multiprocessing.Pool(cores) as pool:
        for data in pool.imap_unordered(worker, tasks):
            if data:
                with open(DATA_FILE, "a") as f:
                    f.write(data + "\n")
                    
    print(f"✅ 完成！儲存至 {DATA_FILE}", flush=True)