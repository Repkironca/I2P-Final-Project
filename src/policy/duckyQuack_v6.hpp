#pragma once
#include "search_types.hpp"
#include "game_history.hpp"
#include <chrono>

// 用 DuckyQuackV6 命名空間包起來，避免跟其他版本的 AI 撞名
namespace DuckyQuackV6 { 

constexpr int TT_MOVE = 5000000; // 被 TT 命中要加幾分
constexpr int CAPTURE_BASE = 1000000; // 吃子步要加幾分
constexpr int KILLER_BASE = 900000; // 殺手步要加幾分
constexpr int KILLER_CONST = 10000; // 第 i 個殺手步少加幾分
constexpr int NORMAL_MOVE = 0; // 一般步加幾分

constexpr int MAX_PLY = 128; // 最大深度限制
constexpr int NUM_KILLERS = 2;   // 記錄多少個殺手步，主流是 2，之後測試要不要放大

// 空格=0, 小兵=1, 騎士=2, 主教=3, 城堡=4, 皇后=5, 國王=6
// 這是 Ridge Regression 訓練出來的
static const int piece_values[7] = {0, 15, 182, 89, 65, 198, 10000};

// 距離敵王權重 (注意：這裡加上了負號！因為特徵是「真實距離」，距離越大扣分越多)
static const int dist_weights[7] = {0, 0, -8, 2, 4, -2, 0};

// 小兵推進權重
constexpr int PAWN_PUSH_WEIGHT = 27;

// 4. PST 棋盤位置戰略分數，白棋視角
static const int pst[30] = {
      -2,    3,   11,   23,   45,
      15,   34,   23,   40,   33,
      28,   22,   31,   19,   41,
      16,   31,   16,   35,   11,
       6,    5,   21,    0,   28,
      15,    7,   15,   35,   27,
};

struct MMParams {
    bool use_kp_eval = true;
    bool use_eval_mobility = true;
    bool report_partial = true;

    static MMParams from_map(const ParamMap& m){
        MMParams p;
        p.use_kp_eval       = param_bool(m, "UseKPEval", true);
        p.use_eval_mobility = param_bool(m, "UseEvalMobility", true);
        p.report_partial    = param_bool(m, "ReportPartial", true);
        return p;
    }
};

class Policy {
public:
    static int eval_ctx(
        State *state, int depth, GameHistory& history, int ply, SearchContext& ctx,
        const MMParams& p, int alpha = -10000000, int beta = 10000000
    );
    static SearchResult search(
        State *state, int depth, GameHistory& history, SearchContext& ctx
    );

    // QS - Search
    static int q_search(
        State *state, GameHistory& history, int ply, SearchContext& ctx,
        const MMParams& p, int alpha, int beta
    );

    static ParamMap default_params();
    static std::vector<ParamDef> param_defs();
};

} // namespace