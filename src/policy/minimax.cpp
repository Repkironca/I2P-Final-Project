#include <utility>
#include "state.hpp"
#include "minimax.hpp"


/*============================================================
 * MiniMax — eval_ctx
 *
 * Negamax without pruning. Caller manages memory.
 *============================================================*/
int MiniMax::eval_ctx(
    State *state,
    int depth,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p,
    int alpha,
    int beta
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    /* === Lazy move generation (sets game_state) === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */

    // [ Hackathon TODO 3-1 ]
    // return the score for a winning terminal state
    // Hint: prefer faster wins by using ply.
    if(state->game_state == WIN){
        // 反正就是我希望 ply 深度愈低愈好，如果不要遞迴太久
        // 回傳極大值，並扣除搜尋深度 ply，確保 AI 選擇最快獲勝的路徑
        return 1000000 - ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    // 逼和，它讓你沒有合法步可以走
    if(state->legal_actions.empty()){
        return 0; 
    }

    /* === Repetition check (game-specific) === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }
    history.push(state->hash());

    if(depth <= 0){
        int score = state->evaluate(
            p.use_kp_eval, p.use_eval_mobility, &history
        ); 
        history.pop(state->hash());
        return score;
    }

    /* === Negamax loop === */
    // with alpha-beta
    int best_score = -10000000;
    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();
        
        int score;
        if(!same) {
            // 換對手下：分數翻轉，且邊界也要翻轉 (-beta 變成新的 alpha, -alpha 變成新的 beta)
            score = -eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
        } else {
            // 同一個人下
            score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, alpha, beta);
        }
        delete next;

        if(score > best_score){
            best_score = score;
        }

        // Alpha-Beta 剪枝
        if(best_score > alpha){
            alpha = best_score;
        }
        // 如果保底分數已經大於等於對手的容忍上限，代表這條路對手絕對不會走，後面的合法步都不用看了
        if(alpha >= beta){
            break; 
        }
    }

    history.pop(state->hash());
    return best_score;
}


/*============================================================
 * MiniMax — search
 *
 * Iterate legal moves, call eval_ctx, return SearchResult.
 *============================================================*/
SearchResult MiniMax::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }


    int best_score = -10000000;
    int alpha = -10000000;
    int beta = 10000000;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();

    // 防呆：就算後面找不到更好的步，至少預設第一步合法步，不要回傳空值否則會被判負
    if (!state->legal_actions.empty()) {
        result.best_move = state->legal_actions[0];
    }

    for(auto& action : state->legal_actions){
        /* [ Hackathon TODO 4-1 ]
         * search this move like TODO 3, but starting from the root */
            State* next = state->next_state(action); // 偷到下一步的走法
            bool same = next->same_player_as_parent();
            int score; // 來推演未來的分數，深度減 1，ply 設為 1
            if(!same) {
                score = -eval_ctx(next, depth - 1, history, 1, ctx, p, -beta, -alpha); 
            } else {
                score = eval_ctx(next, depth - 1, history, 1, ctx, p, alpha, beta); 
            }
            delete next;

            if(score > best_score){
                // [ Hackathon TODO 4-2 ]
                // keep this move if it is the best so far
                best_score = score;
                result.best_move = action; // 記下這一步行動，因為這一部更頂
                if(p.report_partial && ctx.on_root_update){
                   ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
                }
            }

            if(best_score > alpha) alpha = best_score;
             
        move_index++;
    }

    // [ Hackathon TODO 4-3 ]
    // update result and return
      result.score = best_score; // 最終最高分
      result.nodes = ctx.nodes;  // 總共拜訪了多少個節點
      result.pv = {result.best_move}; // 最佳路徑 (Principal Variation，這裡放起手步)
      return result;
} 


/*============================================================
 * MiniMax — default_params / param_defs
 *============================================================*/
ParamMap MiniMax::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}
