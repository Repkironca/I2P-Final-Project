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
    const MMParams& p
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
    int best_score = M_MAX;

    for(auto& action : state->legal_actions){
        // [ Hackathon TODO 3-2 ]
        // create the child state after applying action
        // 利用目前取出來的合法移動，來產生下一個移動，嗯嗯嗯嗯嗯
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        // [Hackathon TODO 3-3]
        // search the child one level deeper
        // 呼叫自己，將深度減 1，並將層數加 1，往未來的回合推演
        int score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p);

        // [Hackathon TODO 3-4]
        // convert raw to the current player's perspective.
        // 對手的獲利就是我的損失，所以把對面的 score 乘上 -1，話說我之後想改這個權重
        if(!same) score = -score;
        delete next;

        // [ Hackathon TODO 3-5 ]
        // update best_score if this child is better.
        // 如果有更好的走法，當然就用更好的走法啊
        if(score > best_score) best_score = score;
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


    int best_score = M_MAX - 10;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();

    for(auto& action : state->legal_actions){
        /* [ Hackathon TODO 4-1 ]
         * search this move like TODO 3, but starting from the root */

            if(score > best_score){
                // [ Hackathon TODO 4-2 ]
                // keep this move if it is the best so far

                if(p.report_partial && ctx.on_root_update){
                   ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
                }
            }  
        move_index++;
    }

    // [ Hackathon TODO 4-3 ]
    // update result and return

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
