#include <utility>
#include <chrono>
#include "state.hpp"
#include "duckyQuack_v1.hpp"

namespace DuckyQuackV1 {

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

int Policy::eval_ctx(
    State *state, int depth, GameHistory& history, int ply,
    SearchContext& ctx, const MMParams& p, std::chrono::time_point<std::chrono::high_resolution_clock> start_time,
    int alpha, int beta
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

    // ==============================================================
    // 每展開 2048 個節點，看一下手錶確認時間
    // 用 & 2047 取代 % 2048，速度快非常多
    if ((ctx.nodes & 2047) == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        if (elapsed > 9900) { // 超過 9900 毫秒就停
            ctx.stop = true;
        }
    }
    // ==============================================================

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
        int score = q_search(state, history, ply, ctx, p, start_time, alpha, beta);
        history.pop(state->hash());
        return score;
    }

    int best_score = -10000000;
    Move best_move; // 用來記錄這個盤面的最佳步
    for(auto& action : state->legal_actions){
    State* next = state->next_state(action);
    // 反正 MiniChess 絕對換人，直接 Negamax 視角翻轉
    int score = -eval_ctx(next, depth - 1, history, ply + 1, ctx, p, start_time, -beta, -alpha);
    delete next;

        if(score > best_score){
            best_score = score;
            best_move = action;
        }
        if(best_score > alpha) alpha = best_score;
        if(alpha >= beta) break; 
    }

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

    history.pop(state->hash());
    return best_score;
}

int Policy::q_search(
    State *state, GameHistory& history, int ply,
    SearchContext& ctx, const MMParams& p, 
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time,
    int alpha, int beta
){
    // 1. 基本防禦與中斷檢查
    ctx.nodes++;
    if ((ctx.nodes & 2047) == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        if (elapsed > 9900) ctx.stop = true;
    }
    if(ctx.stop) return 0;
    if(ply > ctx.seldepth) ctx.seldepth = ply; // seldepth 用來記錄 QS 鑽到了多深

    if(state->legal_actions.empty() && state->game_state == UNKNOWN) state->get_legal_actions();
    if(state->game_state == WIN) return 1000000 - ply;
    if(state->game_state == DRAW) return 0;

    // 我們有權利＂選擇不吃子＂。如果當前盤面的靜態分數就已經夠高了，
    // 我們不一定要為了吃子而去冒險，當我模擬到這裡時，我也能選擇不動
    int stand_pat = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
    
    // 想太美了，stand_pat 大於 Beta，根本不會發生，可以剪了
    if(stand_pat >= beta) return beta;

    if(alpha < stand_pat) alpha = stand_pat;

    int opp = 1 - state->player;
    // 延長賽：遞迴到底了，但我們把吃子看完
    for(auto& action : state->legal_actions){
        int tr = action.second.first % BOARD_H;
        int tc = action.second.second;
        
        // 判斷這步棋是不是吃子
        bool is_capture = (state->board.board[opp][tr][tc] != 0);
        if(!is_capture) continue;

        State* next = state->next_state(action);
        
        int score = -q_search(next, history, ply + 1, ctx, p, start_time, -beta, -alpha);
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

    // 一喊開始就起來設定碼錶
    auto start_time = std::chrono::high_resolution_clock::now();

    if(!state->legal_actions.size()) state->get_legal_actions();

    int best_score = -10000000;
    int alpha = -10000000;
    int beta = 10000000;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();

    if (!state->legal_actions.empty()) {
        result.best_move = state->legal_actions[0];
    }

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action); 
        bool same = next->same_player_as_parent();
        int score; 
        if(!same) {
            score = -eval_ctx(next, depth - 1, history, 1, ctx, p, start_time, -beta, -alpha);
        } else {
            score = eval_ctx(next, depth - 1, history, 1, ctx, p, start_time, alpha, beta);
        }
        delete next;

        if(score > best_score){
            best_score = score;
            result.best_move = action; 
            if(p.report_partial && ctx.on_root_update){
               ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }
        if(best_score > alpha) alpha = best_score;
        move_index++;
    }

    result.score = best_score; 
    result.nodes = ctx.nodes;  
    result.pv = {result.best_move}; 
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

} // for namespace DuckyQuackV1