#include <utility>
#include "state.hpp"
#include "duckyQuack_v6.hpp"

namespace DuckyQuackV6 {

// *=============================================*
// State，嗯笑死我沒有要用你寫的唷，而且我還塞在同一份檔案
static int custom_evaluate(State* state) {
    int score = 0;
    int K_r[2] = {-1, -1}, K_c[2] = {-1, -1};

    // 尋找雙方國王座標
    for (int p = 0; p < 2; ++p) {
        for (int r = 0; r < BOARD_H; ++r) {
            for (int c = 0; c < BOARD_W; ++c) {
                if (state->board.board[p][r][c] == 6) {
                    K_r[p] = r; K_c[p] = c;
                }
            }
        }
    }

    // 計算各項特徵
    for (int p = 0; p < 2; ++p) {
        int sign = (p == 0) ? 1 : -1;
        int opp = 1 - p;

        for (int r = 0; r < BOARD_H; ++r) {
            for (int c = 0; c < BOARD_W; ++c) {
                int piece = state->board.board[p][r][c];
                if (piece == 0) continue;

                // 1. 基礎價值
                score += sign * piece_values[piece];

                if (piece != 6) {
                    // 2. 棋盤 PST 權重 (黑棋視角反轉，把 r 變成 5 - r)
                    int pst_idx = (p == 0) ? (r * 5 + c) : ((5 - r) * 5 + c);
                    score += sign * pst[pst_idx];

                    // 3. 漢密頓距離
                    if (K_r[opp] != -1) {
                        int dist = std::abs(r - K_r[opp]) + std::abs(c - K_c[opp]);
                        score += sign * (dist * dist_weights[piece]);
                    }
                }

                // 4. 小兵推進進度
                if (piece == 1) {
                    int push_dist = (p == 0) ? r : (5 - r);
                    score += sign * (push_dist * PAWN_PUSH_WEIGHT);
                }
            }
        }
    }
    // 轉換為 Minimax 視角：輪到誰，分數就是正的
    return (state->player == 0) ? score : -score;
}
// ==============================================*

// *=============================================*
// Transposition Table
enum TTFlag { TT_EXACT, TT_LOWERBOUND, TT_UPPERBOUND };

struct TTEntry {
    uint64_t hash = 0;       // 用來核對身分，避免雜湊碰撞
    int depth = -1;          // 當時往下算了幾層 (深度不夠的分數不能直接用)
    int score = 0;           // 算出來的評估分數
    TTFlag flag = TT_EXACT;  // 分數的精確度種類 (很重要，下方解釋)
    Move best_move;          // 在這個盤面找到的最佳步
};

// 開啟 2^23 格，約佔用 400MB 記憶體
// 使用 2 的次方是為了可以用 & 來取代超級慢的 % 運算。
const int TT_SIZE = 1 << 23; 
static std::vector<TTEntry> tt_table(TT_SIZE);

// *=============================================*

static Move killer_moves[MAX_PLY][NUM_KILLERS];

// *=============================================*
// MVV-LVA + Killer Heuristic
static int score_move(State* state, const Move& move, const Move& tt_move, int ply) {
    // 如果 TT 裡面就有了，直接給他最高分，不用看了
    if (move == tt_move) return 2000000;

    int opp = 1 - state->player;
    int p = state->player;
    
    int fr = move.first.first; // from row
    int fc = move.first.second; // from column
    int tr = move.second.first % BOARD_H; // to row 
    int tc = move.second.second; // to column

    int attacker = state->board.board[p][fr][fc];
    int victim = state->board.board[opp][tr][tc];

    // 吃子步：價值 = 基礎高分 + (受害者價值*10) - 攻擊者價值，因為我要盡可能殺人
    if (victim != 0) {
        return 1000000 + (piece_values[victim] * 10) - piece_values[attacker];
    }

    // 升變：哥們多一隻皇后你要不要，反正我要
    if(attacker == 1){
        bool promotes = (state->player == 0 && tr == 0) || (state->player == 1 && tr == BOARD_H-1);
        if(promotes) return 80000;
    }

    // 非吃子步
    // 殺手步 (僅非吃子步)
    if (ply >= 0 && ply < MAX_PLY) {
        for (int i = 0; i < NUM_KILLERS; ++i) {
            if (move == killer_moves[ply][i]) {
                // 放心，這個宣告的型態是 static 所以可以這樣寫
                return KILLER_BASE - (i * KILLER_CONST); 
            }
        }
    }

    return NORMAL_MOVE;
}
// *=============================================*

int Policy::eval_ctx(
    State *state, int depth, GameHistory& history, int ply,
    SearchContext& ctx, const MMParams& p, int alpha, int beta
){  
    // 取得當前盤面的 Hash
    uint64_t hash_val = state->hash();
    // 使用 & 來取得陣列 index
    TTEntry& tte = tt_table[hash_val & (TT_SIZE - 1)];

    // 紀錄剛進來時的 original_alpha，這對於最後判斷 UPPER/LOWER BOUND 很有用
    int original_alpha = alpha;

    // 查表
    // 條件：Hash 必須吻合，且快取裡的計算深度必須 >= 我們現在需要的深度
    if (tte.hash == hash_val && tte.depth >= depth) {
        if (tte.flag == TT_EXACT) {
            return tte.score;
        } else if (tte.flag == TT_LOWERBOUND) {
            alpha = std::max(alpha, tte.score); // 下界可以幫我們推高 alpha
        } else if (tte.flag == TT_UPPERBOUND) {
            beta = std::min(beta, tte.score);   // 上界可以幫我們壓低 beta
        }
        
        // 如果經過快取更新後，alpha 已經把 beta 壓扁了，代表這裡可以直接剪枝，不用算了！
        if (alpha >= beta) {
            return tte.score;
        }
    }

    ctx.nodes++;

    if(ply > ctx.seldepth) ctx.seldepth = ply;
    if(ctx.stop) return 0;

    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    if(state->game_state == WIN) return 1000000 - ply;
    if(state->game_state == DRAW) return 0;
    if(state->legal_actions.empty()) return 0; 

    int rep_score;
    if(state->check_repetition(history, rep_score)) return rep_score;
    history.push(state->hash());

    // 停下來了，那就換 QS - Search 接手
    if(depth <= 0){
        int score = q_search(state, history, ply, ctx, p, alpha, beta);
        history.pop(state->hash());
        return score;
    }

    int best_score = -10000000;
    Move best_move; // 用來記錄這個盤面的最佳步

    // 從 TT 表提取先前的最佳步，作為排序首選
    Move tt_move = {{-1, -1}, {-1, -1}};
    if (tte.hash == hash_val) {
        tt_move = tte.best_move;
    }

    // 先進可能排序，有利剪枝（MVV-LVA）
    std::sort(state->legal_actions.begin(), state->legal_actions.end(),
        [&](const Move& a, const Move& b) {
            return score_move(state, a, tt_move, ply) > score_move(state, b, tt_move, ply);
        }
    );

    bool is_first_move = true; // 如果是第一步我們才要完整處理 - PVS
    for(auto& action : state->legal_actions){
        if (ctx.stop) break;
        State* next = state->next_state(action);
        int score;

        if (is_first_move) { // 全視窗搜尋 - PVS
            score = -eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
            is_first_move = false;
        } else { // 否則我們看一步就好
            score = -eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -alpha - 1, -alpha);
            
            // 假設真的翻車，那我們再處理
            if (score > alpha && score < beta) {
                score = -eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
            }
        }
        delete next;

        if(score > best_score){
            best_score = score;
            best_move = action;
        }
        if(best_score > alpha) alpha = best_score;

        // Killer Heuristic
        if(alpha >= beta) {
            int opp = 1 - state->player;
            int tr = action.second.first % BOARD_H;
            int tc = action.second.second;

            // 引發剪枝的是安靜步，且沒有超過最大層數限制
            if (state->board.board[opp][tr][tc] == 0 && ply < MAX_PLY) {
                bool exists = false;
                for (int i = 0; i < NUM_KILLERS; ++i) {
                    if (killer_moves[ply][i] == action) {
                        exists = true;
                        break;
                    }
                }

                if (!exists) {
                    // 所有殺手步往後擠一格，淘汰最後一名
                    for (int i = NUM_KILLERS - 1; i > 0; --i) {
                        killer_moves[ply][i] = killer_moves[ply][i - 1];
                    }
                    killer_moves[ply][0] = action;
                }
            }
            break;          
        }
    }

    if (!ctx.stop) {
        // 存表
        tte.hash = hash_val;
        tte.depth = depth;
        tte.score = best_score;
        tte.best_move = best_move;

        // 判斷精確度旗標
        if (best_score <= original_alpha) {
            // 如果連最初的 alpha 都沒辦法超越，代表所有選項都很爛，這是一個上限值
            tte.flag = TT_UPPERBOUND;
        } else if (best_score >= beta) {
            // 如果分數突破了 beta，代表這步太強了，對手絕對不會讓你走到這，這是一個下限值
            tte.flag = TT_LOWERBOUND;
        } else {
            // 介於中間，這是精確算出來的分數
            tte.flag = TT_EXACT;
        }
    }

    history.pop(state->hash());
    return best_score;
}

int Policy::q_search(
    State *state, GameHistory& history, int ply,
    SearchContext& ctx, const MMParams& p, int alpha, int beta
){
    // 基本防禦與中斷檢查
    ctx.nodes++;
    if(ctx.stop) return 0;
    if(ply > ctx.seldepth) ctx.seldepth = ply; // seldepth 用來記錄 QS 鑽到了多深

    if(state->legal_actions.empty() && state->game_state == UNKNOWN) state->get_legal_actions();
    if(state->game_state == WIN) return 1000000 - ply;
    if(state->game_state == DRAW) return 0;

    // 我們有權利＂選擇不吃子＂。如果當前盤面的靜態分數就已經夠高了，
    // 我們不一定要為了吃子而去冒險，當我模擬到這裡時，我也能選擇不動
    int stand_pat = custom_evaluate(state);
    
    // 想太美了，stand_pat 大於 Beta，根本不會發生，可以剪了
    if(stand_pat >= beta) return beta;

    if(alpha < stand_pat) alpha = stand_pat;

    int opp = 1 - state->player;

    Move empty_move = {{-1, -1}, {-1, -1}}; // 這裡沒有 TT 所以隨便傳一步
    std::sort(state->legal_actions.begin(), state->legal_actions.end(),
        [&](const Move& a, const Move& b) {
            // QS 不考慮 Killer Step 所以隨便傳個 -1 就好
            return score_move(state, a, empty_move, -1) > score_move(state, b, empty_move, -1);
        }
    ); // MVV-LVA

    // 延長賽：遞迴到底了，但我們把吃子看完
    for(auto& action : state->legal_actions){
        int tr = action.second.first % BOARD_H;
        int tc = action.second.second;
        
        // 判斷這步棋是不是吃子
        bool is_capture = (state->board.board[opp][tr][tc] != 0);
        if(!is_capture) continue;

        State* next = state->next_state(action);
        
        int score = -q_search(next, history, ply + 1, ctx, p, -beta, -alpha);
        delete next;
        if(score >= beta) return beta; // 比 beta 好就剪了吧
        if(score > alpha) alpha = score;
    }
    return alpha;
}

SearchResult Policy::search(
    State *state, int depth, GameHistory& history, SearchContext& ctx
){
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    // 清空殺手步陣列
    for(int i = 0; i < MAX_PLY; ++i){
        for(int j = 0; j < NUM_KILLERS; ++j){
            killer_moves[i][j] = {{-1, -1}, {-1, -1}};
        }
    }

    if(!state->legal_actions.size()) state->get_legal_actions();
    if(state->legal_actions.empty()) return result; // 防呆機制

    int alpha = -10000000;
    int beta = 10000000;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();

    // 去查表看看上一層有沒有留下最佳解
    Move tt_move = {{-1, -1}, {-1, -1}};
    uint64_t hash_val = state->hash();
    if (tt_table[hash_val & (TT_SIZE - 1)].hash == hash_val) {
        tt_move = tt_table[hash_val & (TT_SIZE - 1)].best_move;
    }

    std::sort(state->legal_actions.begin(), state->legal_actions.end(),
        [&](const Move& a, const Move& b) {
            return score_move(state, a, tt_move, 0) > score_move(state, b, tt_move, 0);
        }
    );

    // 用區域變數暫存，不要提早寫入 result，會出事
    Move current_best_move = state->legal_actions[0];
    int current_best_score = -10000000;

    bool is_first_move = true; 
    for(auto& action : state->legal_actions){
        if (ctx.stop) break; // 時間到，立刻打斷迴圈！
        
        State* next = state->next_state(action); 
        int score; 
        
        if (is_first_move) { 
            score = -eval_ctx(next, depth - 1, history, 1, ctx, p, -beta, -alpha);
            is_first_move = false;
        } else { 
            score = -eval_ctx(next, depth - 1, history, 1, ctx, p, -alpha - 1, -alpha); 
            if (score > alpha && score < beta) {
              score = -eval_ctx(next, depth - 1, history, 1, ctx, p, -beta, -alpha);
            }
        }
        delete next;

        if(score > current_best_score){
            current_best_score = score;
            current_best_move = action; 
            
            // 報告給 GUI 看的，不影響實質 result
            if(p.report_partial && ctx.on_root_update){
               ctx.on_root_update({current_best_move, current_best_score, depth, move_index + 1, total_moves});
            }
        }
        if(current_best_score > alpha) alpha = current_best_score;
        move_index++;
    }

    // 確保這層是「完整算完」的，才核准寫入 result
    if (!ctx.stop) {
        result.score = current_best_score; 
        result.best_move = current_best_move;
        result.nodes = ctx.nodes;  
        result.pv = {current_best_move}; 
    } else {
        // 如果被打斷了，給出一個極端的無效分數，讓 UBGI 外殼知道這層是廢棄的
        // UBGI 收到這個爛分數後，就會放棄這一層，直接採用上一層的完美結果
        result.score = -20000000; 
        result.nodes = ctx.nodes;
    }

    return result;
}

ParamMap Policy::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> Policy::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}

} // for namespace DuckyQuackV5