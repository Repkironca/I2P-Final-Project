#pragma once
/*============================================================
 * Algorithm Registry
 *
 * Each algorithm defines:
 *   - search() function
 *   - default_params() returning ParamMap
 *   - param_defs() for UCI option advertisement
 *============================================================*/

#include <string>
#include <functional>
#include <vector>
#include "search_types.hpp"
#include "game_history.hpp"
#include "minimax.hpp"
#include "random.hpp"

/*
#include "duckyQuack_v0.hpp"
#include "duckyQuack_v1.hpp"
#include "duckyQuack_v2.hpp"
#include "duckyQuack_v3.hpp"
#include "duckyQuack_v4.hpp"
*/
#include "duckyQuack_v5.hpp"
#include "duckyQuack_v6.hpp"

struct AlgoEntry {
    std::string name;
    ParamMap default_params;
    std::vector<ParamDef> param_defs;
    std::function<SearchResult(State*, int, GameHistory&, SearchContext&)> search;
};

inline const std::vector<AlgoEntry>& get_algo_table(){
    static const std::vector<AlgoEntry> table = {
        {
            "minimax",
            MiniMax::default_params(),
            MiniMax::param_defs(),
            [](State* s, int d, GameHistory& h, SearchContext& c){
                return MiniMax::search(s, d, h, c);
            }
        },
        {
            "random",
            Random::default_params(),
            Random::param_defs(),
            [](State* s, int d, GameHistory& h, SearchContext& c){
                return Random::search(s, d, h, c);
            }
        },
        /*
        {
            "duckyQuack_v0", 
            DuckyQuackV0::Policy::default_params(),
            DuckyQuackV0::Policy::param_defs(),
            [](State* s, int d, GameHistory& h, SearchContext& c){
                return DuckyQuackV0::Policy::search(s, d, h, c);
            }
        },
        {
            "duckyQuack_v1", 
            DuckyQuackV1::Policy::default_params(),
            DuckyQuackV1::Policy::param_defs(),
            [](State* s, int d, GameHistory& h, SearchContext& c){
                return DuckyQuackV1::Policy::search(s, d, h, c);
            }
        },
        {
            "duckyQuack_v2", 
            DuckyQuackV2::Policy::default_params(),
            DuckyQuackV2::Policy::param_defs(),
            [](State* s, int d, GameHistory& h, SearchContext& c){
                return DuckyQuackV2::Policy::search(s, d, h, c);
            }
        },
        {
            "duckyQuack_v3", 
            DuckyQuackV3::Policy::default_params(),
            DuckyQuackV3::Policy::param_defs(),
            [](State* s, int d, GameHistory& h, SearchContext& c){
                return DuckyQuackV3::Policy::search(s, d, h, c);
            }
        },
        {
            "duckyQuack_v4", 
            DuckyQuackV4::Policy::default_params(),
            DuckyQuackV4::Policy::param_defs(),
            [](State* s, int d, GameHistory& h, SearchContext& c){
                return DuckyQuackV4::Policy::search(s, d, h, c);
            }
        },
        */
        {
            "duckyQuack_v5", 
            DuckyQuackV5::Policy::default_params(),
            DuckyQuackV5::Policy::param_defs(),
            [](State* s, int d, GameHistory& h, SearchContext& c){
                return DuckyQuackV5::Policy::search(s, d, h, c);
            }
        },
        {
            "duckyQuack_v6", 
            DuckyQuackV6::Policy::default_params(),
            DuckyQuackV6::Policy::param_defs(),
            [](State* s, int d, GameHistory& h, SearchContext& c){
                return DuckyQuackV6::Policy::search(s, d, h, c);
            }
        },
    };
    return table;
}

inline const AlgoEntry* find_algo(const std::string& name){
    for(auto& entry : get_algo_table()){
        if(entry.name == name){
            return &entry;
        }
    }
    return nullptr;
}

inline std::string default_algo_name(){ return "minimax"; }
