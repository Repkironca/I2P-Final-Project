#pragma once
#include "search_types.hpp"
#include "game_history.hpp"
#include <chrono>

// 用 DuckyQuackV7 命名空間包起來，避免跟其他版本的 AI 撞名
namespace DuckyQuackV7 { 

constexpr int TT_MOVE = 5000000; // 被 TT 命中要加幾分
constexpr int CAPTURE_BASE = 1000000; // 吃子步要加幾分
constexpr int KILLER_BASE = 900000; // 殺手步要加幾分
constexpr int KILLER_CONST = 10000; // 第 i 個殺手步少加幾分
constexpr int NORMAL_MOVE = 0; // 一般步加幾分

constexpr int MAX_PLY = 128; // 最大深度限制
constexpr int NUM_KILLERS = 2;   // 記錄多少個殺手步，主流是 2，之後測試要不要放大

// 這樣大概 2.68 GB
// 使用 2 的次方是為了可以用 & 來取代超級慢的 % 運算。
const int TT_SIZE = 1 << 26; 

// 空格=0, 小兵=1, 騎士=2, 主教=3, 城堡=4, 皇后=5, 國王=6
// 這是 Ridge Regression (HQ資料) 訓練出來的
static const int piece_values[7] = {0, 360, 1320, 890, 1390, 2070, 100000};

// 距離敵王權重 (注意：這裡必須補上負號！因為特徵是「真實距離」，距離越大扣分越多)
static const int dist_weights[7] = {0, -19, -15, -11, -54, -7, 0};

// 小兵推進權重
constexpr int PAWN_PUSH_WEIGHT = 16;

// 4. 棋子 PST 戰略分數 (白棋視角)
// 陣列維度為 [7][30]，對應 piece 的 index
static const int pst[7][30] = {
    {0}, // [0] 空格
    // [1] 小兵
    {
           0,    0,    0,    0,    0,
         -25,  -32,  -54,  -13,  -17,
          -3,  -43,  -46,   26,   39,
           0,  -26,   26,   20,  -16,
          16,   48,   56,   51,   28,
           0,    0,    0,    0,    0
    },
    // [2] 騎士
    {
          -4,  -50,   -1,  -63,   -4,
          -2,    4,    5,   -2,    7,
          -3,   26,   57,  -13,   -3,
           1,    1,   20,   47,    6,
           5,   15,    3,   19,    2,
          -1,   29,    0,    6,   25
    },
    // [3] 主教
    {
           9,  -14,  -74,   -8,    3,
         -24,   29,   -3,   38,  -14,
           3,   -1,   84,  -23,   -3,
         -21,   36,  -17,   41,  -44,
          17,   34,   13,    4,   -3,
          -7,   10,    9,   12,    5
    },
    // [4] 城堡
    {
         -41,  -25,  -12,  -45, -126,
         -29,   -9,  -14,  -27,    1,
         -21,    4,   -3,   15,   14,
         -15,    5,   -6,    9,    9,
           7,   36,   66,   33,   60,
         104,   63,   41,   17,   30
    },
    // [5] 
    {
         -19,  -62,  -19,  -30,  -34,
           1,  -21,  -19,  -21,  -19,
           7,   20,   10,  -12,   -3,
         -16,   41,    1,   22,    4,
          27,   11,   21,   40,   56,
          80,   47,   37,   36,   19
    },
    // [6] 國王
    {
           1,   29,   31,   80,  197,
          22,   16,   72,  152,   78,
           2,  -25,   15,   25,   11,
         -11,  -60,  -10,  -15,  -20,
         -64,  -99,  -56,  -47,  -19,
         -34,  -83,  -68,   -8,   -4
    }
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
        const MMParams& p, int alpha = -10000000, int beta = 10000000, bool is_null_move = false
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