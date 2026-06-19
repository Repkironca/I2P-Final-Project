#pragma once
#include "search_types.hpp"
#include "game_history.hpp"
#include <chrono>

// 用 DuckyQuackV8 命名空間包起來，避免跟其他版本的 AI 撞名
namespace DuckyQuackV8 { 

constexpr int TT_MOVE = 5000000; // 被 TT 命中要加幾分
constexpr int CAPTURE_BASE = 1000000; // 吃子步要加幾分
constexpr int KILLER_BASE = 900000; // 殺手步要加幾分
constexpr int KILLER_CONST = 10000; // 第 i 個殺手步少加幾分
constexpr int NORMAL_MOVE = 0; // 一般步加幾分

constexpr int MAX_PLY = 128; // 最大深度限制
constexpr int NUM_KILLERS = 2;   // 記錄多少個殺手步，主流是 2，之後測試要不要放大

const int TT_SIZE = 1 << 24; 

// V2: 敵王曼哈頓距離和權重 (小兵, 騎士, 主教, 城堡, 皇后)
static const int u2[5] = {6, 8, -2, -5, -12};

// V3: PST 靜態位階權重
static const int u3[7][30] = {
    // [0] 空格
    {0}, 
    
    // [1] 小兵
    {
           0,    0,    0,    0,    0,
         135,  163,  106,  229,  113,
         125,   81,   85,  160,   97,
         112,   81,   92,  177,   98,
         137,  162,   72,  223,  105,
           0,    0,    0,    0,    0
    },
    
    // [2] 騎士
    {
        -161,  112,   58,   91,  274,
         204,  184,  192,  136,  332,
          86,  212,  191,  256,  125,
         124,  269,  221,  203,  169,
         170,  206,  168,  243,  330,
        -178,  168,   83,  121,  252
    },
    
    // [3] 主教
    {
         167,  -56,   69,   80,  165,
          14,  136,   83,  163,   -7,
          46,  197,  171,  125,   77,
          85,  156,  184,  100,  110,
          95,  170,  106,  157,  -31,
         122,   -3,   68,   56,  171
    },
    
    // [4] 城堡
    {
         303,  292,  311,  206,  153,
         204,  298,  289,  258,  312,
         168,  257,  176,  310,  303,
         167,  226,  159,  309,  363,
         179,  240,  289,  286,  296,
         267,  295,  314,  228,  155
    },
    
    // [5] 皇后
    {
         325,  290,  292,  311,  168,
         338,  253,  296,  295,  342,
         250,  364,  250,  269,  257,
         259,  399,  244,  272,  287,
         364,  248,  270,  297,  302,
         351,  245,  274,  301,  212
    },
    
    // [6] 國王
    {
        -114, -129,  -60,  254,  -81,
          86, -302, -195,   79,   55,
          -9, -306, -394,   11, -267,
         -47, -286, -398,  -25, -345,
         106, -233, -200,  116,  116,
        -157, -145,  -86,  174,  -54
    }
};

// V4: 立即戰術威脅權重 (Check, 吃兵, 吃馬, 吃主教, 吃城堡, 吃后)
static const int u4_tactical[6] = {5, 19, 6, 37, 79, 68};

// V5: 完整小兵升變階梯權重 (剩1步, 剩2步, 剩3步, 剩4步)
static const int u5_pawn_stages[4] = {-12, -14, -24, -40};

// V6: 零成本己方行動力權重
constexpr int u6_mobility = 4;

// V7: 國王時序動態格位權重
static const int u7_king_step[30] = {1, 2, 0, -4, 3, -2, 4, 4, 1, 0, 0, 3, 6, 0, 5, 1, 4, 6, 1, 6, -3, 3, 4, 0, 0, 2, 2, 1, -2, 2};

// 盤面基礎偏置分數 (Intercept)
constexpr int u_intercept = 480;

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