#include <utility>
#include <chrono>
#include "state.hpp"
#include "duckyQuack_v3.hpp"

namespace DuckyQuackV3 {

// *=============================================*
// Transposition Table
enum TTFlag { TT_EXACT, TT_LOWERBOUND, TT_UPPERBOUND };

struct TTEntry {
    uint64_t hash = 0;       // 用來核對身分，避免雜湊碰撞
    int depth = -1;          // 當時往下算了幾層 (深度不夠的分數不能直接用)
    int score = 0;           // 算出來的評估分數
    TTFlag flag = TT_EXACT;  // 分數的精確度種類 
    Move best_move;          // 在這個盤面找到的最佳步
};

// 開啟 2^23 格，約佔用 400MB 記憶體
// 使用 2 的次方是為了可以用 & 來取代超級慢的 % 運算。
const int TT_SIZE = 1 << 23; 
static std::vector<TTEntry> tt_table(TT_SIZE);

// *=============================================*

// *=================MVV-LVA=================*
static const int piece_values[7] = {0, 10, 50, 30, 30, 90, 1000};

static int score_move(State* state, const Move& move, const Move& tt_move) {
    // 如果這個盤面以前算過，直接把當時的最佳步排第一
    if (move == tt_move) return 2000000;

    int opp = 1 - state->player;
    int p = state->player;
    
    int fr = move.first.first;
    int fc = move.first.second;
    int tr = move.second.first % BOARD_H; // 處理升變
    int tc = move.second.second;

    int attacker = state->board.board[p][fr][fc];
    int victim = state->board.board[opp][tr][tc];

    // 受害者越貴越好，攻擊者越便宜越好
    if (victim != 0) {
        // 加 1000000 是為了確保所有吃子步，都絕對排在不吃子的安靜步前面
        return 1000000 + (piece_values[victim] * 10) - piece_values[attacker];
    }

    // 安靜步
    return 0;
}
// *=========================================*

int Policy::eval_ctx(
    State *state, int depth, GameHistory& history, int ply,
    SearchContext& ctx, const MMParams& p, std::chrono::time_point<std::chrono::high_resolution_clock> start_time,
    int alpha, int beta
){  
    uint64_t hash_val = state->hash();
    TTEntry& tte = tt_table[hash_val & (TT_SIZE - 1)];
    int original_alpha = alpha; // 最後我需要用這個來知道目前是精確的，或是上下界而已

    Move tt_move = {{-1, -1}, {-1, -1}}; 
    if (tte.hash == hash_val) { // 之前搜尋過了，抓 TT 的東西就好
        tt_move = tte.best_move; 
        if (tte.depth >= depth) { 
            if (tte.flag == TT_EXACT) return tte.score;
            else if (tte.flag == TT_LOWERBOUND) alpha = std::max(alpha, tte.score);
            else if (tte.flag == TT_UPPERBOUND) beta = std::min(beta, tte.score);
            if (alpha >= beta) return tte.score;
        }
    }

    ctx.nodes++;
    
    // 每 2048 次看一下手錶，確定我 1900 ms 時一定要 shutdown 所有運算
    if ((ctx.nodes & 2047) == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        if (elapsed > 1900) { 
            ctx.stop = true;
        }
    }

    if(ply > ctx.seldepth) ctx.seldepth = ply;
    if(ctx.stop) return 0; // 這代表 1900 ms 了，隨便給個 0，直接退出

    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    if(state->game_state == WIN) return 1000000 - ply;
    if(state->game_state == DRAW) return 0;
    if(state->legal_actions.empty()) return 0; 

    int rep_score;
    if(state->check_repetition(history, rep_score)) return rep_score;
    
    history.push(state->hash());

    if(depth <= 0){
        int score = q_search(state, history, ply, ctx, p, start_time, alpha, beta);
        history.pop(state->hash());
        return score;
    }

    int best_score = -10000000;
    Move best_move; 
    std::sort(state->legal_actions.begin(), state->legal_actions.end(),
        [&](const Move& a, const Move& b) {
            return score_move(state, a, tt_move) > score_move(state, b, tt_move);
        }
    );

    int move_index = 0;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        int score;

        // [刪除] 我之前亂教的 `same_player_as_parent` 判斷。完全遵照你的指示：西洋棋 100% 輪流下。
        // [新增] 乾淨俐落的純 PVS 零寬度視窗掃描 (Negamax)
        if (move_index == 0) {
            score = -eval_ctx(next, depth - 1, history, ply + 1, ctx, p, start_time, -beta, -alpha);
        } else {
            score = -eval_ctx(next, depth - 1, history, ply + 1, ctx, p, start_time, -alpha - 1, -alpha);
            if (score > alpha && score < beta) {
                score = -eval_ctx(next, depth - 1, history, ply + 1, ctx, p, start_time, -beta, -alpha);
            }
        }
        delete next;
        
        // [新增] 核心防漏機制：如果被碼錶強制砍斷，跳出前**必須**把剛剛 push 的 hash pop 掉，否則 memory 會卡死。
        if(ctx.stop) {
            history.pop(state->hash());
            return 0;
        }

        if(score > best_score){
            best_score = score;
            best_move = action;
        }
        if(best_score > alpha) alpha = best_score;
        if(alpha >= beta) break; 
        
        move_index++; 
    }

    tte.hash = hash_val;
    tte.depth = depth;
    tte.score = best_score;
    tte.best_move = best_move;

    if (best_score <= original_alpha) {
        tte.flag = TT_UPPERBOUND;
    } else if (best_score >= beta) {
        tte.flag = TT_LOWERBOUND;
    } else {
        tte.flag = TT_EXACT;
    }

    history.pop(state->hash());
    return best_score;
}

int Policy::q_search(
    State *state, GameHistory& history, int ply,
    SearchContext& ctx, const MMParams& p, 
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time,
    int alpha, int beta
){
    ctx.nodes++;
    if ((ctx.nodes & 2047) == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        if (elapsed > 1900) ctx.stop = true;
    }
    if(ctx.stop) return 0;
    if(ply > ctx.seldepth) ctx.seldepth = ply; 

    if(state->legal_actions.empty() && state->game_state == UNKNOWN) state->get_legal_actions();
    if(state->game_state == WIN) return 1000000 - ply;
    if(state->game_state == DRAW) return 0;

    int stand_pat = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
    
    if(stand_pat >= beta) return beta;
    if(alpha < stand_pat) alpha = stand_pat;

    int opp = 1 - state->player;

    Move empty_move = {{-1, -1}, {-1, -1}};
    std::sort(state->legal_actions.begin(), state->legal_actions.end(),
        [&](const Move& a, const Move& b) {
            return score_move(state, a, empty_move) > score_move(state, b, empty_move);
        }
    );

    for(auto& action : state->legal_actions){
        int tr = action.second.first % BOARD_H;
        int tc = action.second.second;
        
        bool is_capture = (state->board.board[opp][tr][tc] != 0);
        if(!is_capture) continue;

        State* next = state->next_state(action);
        
        // [刪除] `same_player_as_parent` 的多餘判斷。
        // [恢復] 最單純的 Negamax 吃子搜尋。
        int score = -q_search(next, history, ply + 1, ctx, p, start_time, -beta, -alpha);
        
        delete next;
        
        // [新增] QS 延長賽防禦：防止無限連環吃子導致超時停不下來。
        if (ctx.stop) return 0;

        if(score >= beta) return beta; 
        if(score > alpha) alpha = score;
    }
    return alpha;
}

SearchResult Policy::search(
    State *state, int depth, GameHistory& history, SearchContext& ctx
){
    // * 新增：全域計時器與逾時標記。
    // 因為外層 ubgi.cpp 會不斷呼叫 search (depth 1, 2, 3...19)
    // 如果每次都重新設定 start_time，1900ms 的限制就會被重置 19 次 (導致跑了 30 幾秒)
    static auto global_start_time = std::chrono::high_resolution_clock::now();
    static bool time_is_up = false;

    // 當 ubgi 請求 depth 1 時，代表這是一個全新的回合，重置碼錶
    if (depth == 1) {
        global_start_time = std::chrono::high_resolution_clock::now();
        time_is_up = false;
    }

    ctx.reset();
    
    // * 如果在較淺的層數已經超時，直接拉起煞車，拒絕計算後面的深層
    if (time_is_up) {
        ctx.stop = true;
    }

    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;

    // 加回確保 legal_actions 有值的檢查，避免 root 沒有步可跑
    if(state->legal_actions.empty()) state->get_legal_actions();
    if(state->legal_actions.empty()) return result;
    
    Move global_best_move = state->legal_actions[0];
    
    // * 已經超時的話，直接回傳預設步，瞬間跳過這層
    if (ctx.stop) {
        result.depth = depth;
        result.best_move = global_best_move;
        return result;
    }

    int best_score = -10000000;
    int alpha = -10000000;
    int beta = 10000000;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();
    Move current_best_move = state->legal_actions[0];

    // 在進入這層的推演前，先拿上一層的最佳步來當排序 MVP
    // * 改從 TT 表中調閱，因為這是 UBGI 外掛迴圈最穩的作法
    Move tt_move = {{-1, -1}, {-1, -1}};
    uint64_t hash_val = state->hash();
    if (tt_table[hash_val & (TT_SIZE - 1)].hash == hash_val) {
        tt_move = tt_table[hash_val & (TT_SIZE - 1)].best_move;
    }

    std::sort(state->legal_actions.begin(), state->legal_actions.end(),
        [&](const Move& a, const Move& b) {
            return score_move(state, a, tt_move) > score_move(state, b, tt_move);
        }
    );

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action); 
        int score; 
        
        if (move_index == 0) {
            // 第一步：正常視窗 [alpha, beta] 
            score = -eval_ctx(next, depth - 1, history, 1, ctx, p, global_start_time, -beta, -alpha);
        } else {
            // 後續：零寬度視窗 [-alpha-1, -alpha]
            score = -eval_ctx(next, depth - 1, history, 1, ctx, p, global_start_time, -alpha - 1, -alpha);
            // 翻車重算
            if (score > alpha && score < beta) {
                score = -eval_ctx(next, depth - 1, history, 1, ctx, p, global_start_time, -beta, -alpha);
            }
        }
        delete next;
        
        // 防禦機制：如果被碼錶強制砍斷，這步的分數不能信，立刻終止計算
        if (ctx.stop) {
            time_is_up = true; // * 標記全域超時，讓後面的 depth 也直接放棄
            break;
        }

        if(score > best_score){
            best_score = score;
            current_best_move = action; 
            if(p.report_partial && ctx.on_root_update){
               ctx.on_root_update({current_best_move, best_score, depth, move_index + 1, total_moves});
            }
        }
        if(best_score > alpha) alpha = best_score;
        move_index++;
    }

    // * 結算防禦。沒有被超時砍斷，才把這次的結果存入 TT
    if (!ctx.stop) {
        TTEntry& tte = tt_table[hash_val & (TT_SIZE - 1)];
        tte.hash = hash_val;
        tte.depth = depth;
        tte.score = best_score;
        tte.best_move = current_best_move;
        tte.flag = TT_EXACT;
    }

    result.depth = depth; 
    result.score = best_score; 
    result.nodes = ctx.nodes;  
    result.pv = {current_best_move}; 
    result.best_move = current_best_move;
    
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

} // for namespace DuckyQuackV2