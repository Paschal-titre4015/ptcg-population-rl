#pragma once
#include "../../../engine/cpp/agent.h"
#include "../../../engine/cpp/api_enums.h"
// Port of the accompanying official Notebook main.py.
namespace official::iono {
class Policy final : public Agent {
    const CardTableView& card_table;
    std::vector<int> my_deck;
    static constexpr int Iono_Voltorb = 265;
    static constexpr int Iono_Tadbulb = 268;
    static constexpr int Iono_Bellibolt_ex = 269;
    static constexpr int Iono_Wattrel = 270;
    static constexpr int Iono_Kilowattrel = 271;
    static constexpr int Buddy_Buddy_Poffin = 1086;
    static constexpr int Night_Stretcher = 1097;
    static constexpr int Max_Rod = 1110;
    static constexpr int Energy_Retrieval = 1118;
    static constexpr int Ultra_Ball = 1121;
    static constexpr int Poke_Pad = 1152;
    static constexpr int Lillie_Determination = 1227;
    static constexpr int Canari = 1233;
    static constexpr int Levincia = 1254;
    static constexpr int Basic_Lightning_Energy = 4;
    bool can_attack = false;
public:
    Policy(const CardTableView& table, const std::vector<int>& deck) : card_table(table), my_deck(deck) {}
    std::vector<int> decide(const Observation& obs) override {

        int context{};
        int my_index{};

        int op_prize{};
        std::map<int, int> field_counts{};
        std::map<int, int> field_hand_counts{};
        bool active_attacker{};
        bool bench_attacker{};
        int energy_count{};
        bool can_ability{};
        CardView p{};
        int field_pokemon1{};
        int field_pokemon2{};
        bool no_more_pokemon{};
        int stadium_id{};
        CardView c{};
        std::map<int, int> hand_counts{};
        std::vector<double> hand_scores{};
        int unused_hand_count{};
        MasterView data{};
        double score{};
        std::map<int, int> discard_counts{};
        OptionView o{};
        int op_active_hp{};
        bool no_draw{};
        std::vector<double> scores{};
        std::map<int, int> id_counts{};
        int energy{};
        std::vector<int> desc_indices{};
        // Policy order follows ../main.py:70.
        const auto& state = obs.current;
        const auto& select = obs.select;
        context = select.context;
        my_index = state.yourIndex;
        const auto& my_state = state.players.at(my_index);
        const auto& op_state = state.players.at(1 - my_index);
        op_prize = count_of(op_state.prize);
        field_counts = {};
        field_hand_counts = {};
        active_attacker = false;
        bench_attacker = false;
        energy_count = 0;
        can_ability = false;
        for (const auto& entry_1 : my_state.active) {
            p = entry_1;
            if (p == nullptr) {
                continue;
            }
            field_counts[p.id] += 1;
            field_hand_counts[p.id] += 1;
            energy_count += count_of(p.energies);
            if (p.id == Iono_Kilowattrel) {
                if (count_of(p.energies) > 0) {
                    can_ability = true;
                }
            }
            if (p.id == Iono_Voltorb) {
                if (count_of(p.energies) >= 2) {
                    active_attacker = true;
                }
            }
        }
        for (const auto& entry_2 : my_state.bench) {
            p = entry_2;
            field_counts[p.id] += 1;
            field_hand_counts[p.id] += 1;
            energy_count += count_of(p.energies);
            if (p.id == Iono_Kilowattrel) {
                if (count_of(p.energies) > 0) {
                    can_ability = true;
                }
            }
            if (p.id == Iono_Voltorb) {
                if (count_of(p.energies) >= 2) {
                    bench_attacker = true;
                }
            }
        }
        field_pokemon1 = (field_counts[Iono_Tadbulb] + field_counts[Iono_Bellibolt_ex]);
        field_pokemon2 = (field_counts[Iono_Wattrel] + field_counts[Iono_Kilowattrel]);
        no_more_pokemon = (count_of(my_state.bench) >= 5);
        if ((field_counts[Iono_Tadbulb] + field_counts[Iono_Wattrel]) >= 1) {
            no_more_pokemon = false;
        }
        stadium_id = 0;
        for (const auto& entry_3 : state.stadium) {
            c = entry_3;
            stadium_id = c.id;
        }
        hand_counts = {};
        hand_scores = {};
        unused_hand_count = 0;
        for (const auto& entry_4 : my_state.hand) {
            c = entry_4;
            data = card_table.at(c.id);
            score = (-10000);
            if (c.id == Iono_Voltorb) {
                score = 100;
            } else if (c.id == Iono_Bellibolt_ex) {
                if (field_counts[c.id] <= 1) {
                    score = 120;
                }
            } else if (c.id == Iono_Kilowattrel) {
                if (field_counts[c.id] <= 1) {
                    score = 140;
                }
            } else if (c.id == Ultra_Ball) {
                if (!no_more_pokemon) {
                    score = 10;
                }
            } else if (c.id == Night_Stretcher) {
                score = 50;
            } else if (c.id == Energy_Retrieval) {
                score = 20;
            } else if (c.id == Max_Rod) {
                score = 1000;
            } else if (c.id == Lillie_Determination) {
                score = 150;
            } else if (c.id == Canari) {
                score = 160;
            } else if (c.id == Levincia) {
                if (stadium_id != Levincia) {
                    score = 30;
                }
            } else if (c.id == Basic_Lightning_Energy) {
                score = (-10);
            }
            score -= (hand_counts[c.id] * 100);
            hand_scores.push_back(score);
            if (score < 0) {
                unused_hand_count += 1;
            }
            hand_counts[c.id] += 1;
            field_hand_counts[c.id] += 1;
        }
        discard_counts = {};
        for (const auto& entry_5 : my_state.discard) {
            c = entry_5;
            discard_counts[c.id] += 1;
        }
        if (context == SelectContext::MAIN) {
            can_attack = false;
            for (const auto& entry_6 : select.option) {
                o = entry_6;
                if (o.type == OptionType::ATTACK) {
                    can_attack = true;
                }
            }
        }
        op_active_hp = 10000;
        if (count_of(op_state.active) >= 1) {
            if (at_index(op_state.active, 0) != nullptr) {
                op_active_hp = at_index(op_state.active, 0).hp;
            }
        }
        no_draw = (my_state.deckCount <= 5);
        scores = {};
        id_counts = {};
        for (const auto& entry_7 : select.option) {
            o = entry_7;
            score = 0;
            if (o.type == OptionType::NUMBER) {
                score = o.number;
            } else if (o.type == OptionType::YES) {
                score = 1;
            } else if ((o.type == OptionType::ATTACH) || (context == SelectContext::ATTACH_FROM)) {
                if (o.type == OptionType::ATTACH) {
                    p = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index);
                } else {
                    p = get_card(obs, o.area, o.index, o.playerIndex);
                }
                score = 40000;
                if (p.id == Iono_Voltorb) {
                    if (count_of(p.energies) >= 2) {
                        if ((o.inPlayArea == AreaType::ACTIVE) && (!can_attack)) {
                            score += 3000;
                        }
                    } else if (o.inPlayArea == AreaType::ACTIVE) {
                        score += 5000;
                    } else if (bench_attacker || active_attacker) {
                        score += 100;
                    } else {
                        score += 1000;
                    }
                } else if (p.id == Iono_Tadbulb) {
                    score += (10 - count_of(p.energies));
                } else if (p.id == Iono_Bellibolt_ex) {
                    if (count_of(p.energies) >= 4) {
                        if ((o.inPlayArea == AreaType::ACTIVE) && (!can_attack)) {
                            score += 500;
                        }
                    } else if (o.inPlayArea == AreaType::ACTIVE) {
                        score += 800;
                    } else if (bench_attacker || active_attacker) {
                        score += (14 - count_of(p.energies));
                    } else {
                        score += 100;
                    }
                } else if (p.id == Iono_Wattrel) {
                    if ((count_of(p.energies) >= 1) || (o.inPlayArea == AreaType::ACTIVE)) {
                        score += (10 - count_of(p.energies));
                    } else {
                        score += 6000;
                    }
                } else if (p.id == Iono_Kilowattrel) {
                    if (count_of(p.energies) >= 1) {
                        score += (11 - count_of(p.energies));
                    } else {
                        score += 8000;
                    }
                }
            } else if (o.type == OptionType::CARD) {
                c = get_card(obs, o.area, o.index, o.playerIndex);
                if (c != nullptr) {
                    data = card_table.at(c.id);
                    if ((context == SelectContext::SWITCH) || (context == SelectContext::TO_ACTIVE) || (context == SelectContext::SETUP_ACTIVE_POKEMON)) {
                        energy = 0;
                        if (c.is_pokemon) {
                            energy = count_of(c.energies);
                            score -= c.hp;
                            score -= (energy * 100);
                        }
                        if (c.id == Iono_Voltorb) {
                            if ((20 + (energy_count * 20)) >= op_active_hp) {
                                score += 100000;
                            } else {
                                score += 1500;
                            }
                            if (energy >= 1) {
                                score += 200;
                                if (energy >= 2) {
                                    score += 10000;
                                }
                            }
                        } else if (c.id == Iono_Bellibolt_ex) {
                            score += 1000;
                            if (energy >= 4) {
                                score += 1000;
                            }
                        } else if (c.id == Iono_Tadbulb) {
                            score += 10;
                        }
                    } else if ((context == SelectContext::TO_HAND) || (context == SelectContext::TO_BENCH)) {
                        if (c.id == Basic_Lightning_Energy) {
                            score += 1;
                        } else if (c.id == Iono_Voltorb) {
                            if (o.area == AreaType::DISCARD) {
                                score += 100000;
                            }
                            if (field_counts[c.id] == 0) {
                                score += 110;
                            } else if ((field_counts[c.id] == 1) && (op_prize >= 2)) {
                                score += 5;
                            }
                        } else if (c.id == Iono_Tadbulb) {
                            if (field_pokemon1 == 0) {
                                score += 200;
                            } else if (field_pokemon1 == 1) {
                                if ((op_prize >= 3) || ((op_prize >= 2) && (field_counts[Iono_Bellibolt_ex] == 0))) {
                                    score += 20;
                                }
                            }
                        } else if (c.id == Iono_Bellibolt_ex) {
                            if (field_hand_counts[c.id] == 0) {
                                score += 250;
                                if (field_counts[Iono_Tadbulb] > 0) {
                                    score += 300;
                                }
                            } else if (field_hand_counts[c.id] == 1) {
                                if (op_prize >= 3) {
                                    score += 30;
                                    if (field_counts[Iono_Tadbulb] > 0) {
                                        score += 30;
                                    }
                                }
                            }
                        } else if (c.id == Iono_Wattrel) {
                            if (field_pokemon2 == 0) {
                                score += 320;
                            } else if (field_pokemon2 == 1) {
                                score += 15;
                            }
                        } else if (c.id == Iono_Kilowattrel) {
                            if (field_hand_counts[c.id] == 0) {
                                score += 300;
                                if (field_counts[Iono_Wattrel] > 0) {
                                    score += 250;
                                }
                            } else if (field_hand_counts[c.id] == 1) {
                                score += 25;
                                if (field_counts[Iono_Wattrel] > 0) {
                                    score += 25;
                                }
                            }
                        }
                        if (c.id != Basic_Lightning_Energy) {
                            if (hand_counts[c.id] >= 2) {
                                score -= 20000;
                            } else if (hand_counts[c.id] >= 1) {
                                score -= 2000;
                            }
                            if (id_counts[c.id] == 1) {
                                score -= 1000;
                            } else if (id_counts[c.id] >= 2) {
                                score -= 10000;
                            }
                        }
                        id_counts[c.id] += 1;
                    } else if (context == SelectContext::DISCARD) {
                        if ((o.area == AreaType::HAND) && (o.playerIndex == my_index)) {
                            score = (-at_index(hand_scores, o.index));
                        }
                    }
                }
            } else if (o.type == OptionType::PLAY) {
                c = get_card(obs, AreaType::HAND, o.index, my_index);
                data = card_table.at(c.id);
                if (data.cardType == CardType::STADIUM) {
                    if ((discard_counts[Basic_Lightning_Energy] >= 1) || can_ability) {
                        score = 85000;
                    } else {
                        score = (-1);
                    }
                } else if (data.cardType == CardType::SUPPORTER) {
                    score = 25000;
                    if (c.id == Lillie_Determination) {
                        score += 1000;
                    } else if (no_draw) {
                        score = (-1);
                    } else if (c.id == Canari) {
                        if (no_more_pokemon) {
                            score = (-1);
                        } else if ((field_counts[Iono_Voltorb] > 0) && (field_counts[Iono_Bellibolt_ex] > 0) && (field_counts[Iono_Kilowattrel] > 0)) {
                            score += 100;
                        } else {
                            score += 2000;
                        }
                    }
                } else if (data.cardType == CardType::POKEMON) {
                    score = 100000;
                    if ((c.id == Iono_Voltorb) && (field_counts[Iono_Voltorb] >= 2)) {
                        score = (-1);
                    } else if ((c.id == Iono_Tadbulb) && (field_pokemon1 >= 2)) {
                        score = (-1);
                    } else if ((c.id == Iono_Wattrel) && (field_pokemon2 >= 2)) {
                        if ((op_prize >= 2) || (field_counts[Iono_Voltorb] == 0) || (field_counts[Iono_Bellibolt_ex] == 0)) {
                            score = (-1);
                        }
                    }
                } else if (c.id == Night_Stretcher) {
                    if ((discard_counts[Iono_Voltorb] > 0) || ((discard_counts[Iono_Bellibolt_ex] > 0) && (field_counts[Iono_Tadbulb] > 0)) || ((discard_counts[Iono_Kilowattrel] > 0) && (field_counts[Iono_Wattrel] > 0))) {
                        score = 75000;
                    } else {
                        score = (-1);
                    }
                } else if (c.id == Energy_Retrieval) {
                    score = 61000;
                } else if (c.id == Max_Rod) {
                    if ((state.turn >= 3) && (discard_counts[Basic_Lightning_Energy] >= 2)) {
                        score = 55000;
                    } else {
                        score = (-1);
                    }
                } else if (no_draw) {
                    score = (-1);
                } else if (c.id == Buddy_Buddy_Poffin) {
                    score = 80000;
                } else if (c.id == Ultra_Ball) {
                    if (no_more_pokemon || (state.turn <= 2)) {
                        score = (-1);
                    } else if ((field_hand_counts[Iono_Bellibolt_ex] > 0) && (field_hand_counts[Iono_Kilowattrel] > 0)) {
                        if (unused_hand_count >= 2) {
                            score = 45000;
                        } else {
                            score = (-1);
                        }
                    } else if (unused_hand_count >= 1) {
                        score = 62000;
                    } else {
                        score = (-1);
                    }
                } else if (c.id == Poke_Pad) {
                    score = 79000;
                }
            } else if (o.type == OptionType::EVOLVE) {
                score = 110000;
            } else if (o.type == OptionType::ABILITY) {
                score = (-1);
                c = get_card(obs, o.area, o.index, my_index);
                if (c.id == Iono_Bellibolt_ex) {
                    score = 50000;
                } else if (c.id == Levincia) {
                    score = 8000;
                } else if ((!no_draw) && (c.id == Iono_Kilowattrel)) {
                    score = 30000;
                }
            } else if (o.type == OptionType::RETREAT) {
                if (bench_attacker && (!active_attacker)) {
                    score = 10000;
                } else {
                    score = (-1);
                }
            } else if (o.type == OptionType::ATTACK) {
                score = o.attackId;
            }
            scores.push_back(score);
        }
        desc_indices = ranked_indices(scores);
        return take(desc_indices, select.maxCount);
    }
};
inline std::unique_ptr<Agent> make_policy(const CardTableView& table,const std::vector<int>& deck) {
    std::map<int,int> actual;
    for(int id:deck)++actual[id];
    const std::map<int,int> expected{{4, 22}, {265, 3}, {268, 3}, {269, 3}, {270, 3}, {271, 3}, {1086, 3}, {1097, 2}, {1110, 1}, {1118, 1}, {1121, 3}, {1152, 2}, {1227, 4}, {1233, 4}, {1254, 3}};
    if(actual!=expected)throw std::runtime_error("Official deck mismatch: iono");
    return std::make_unique<Policy>(table,deck);
}
} // namespace official::iono
