#pragma once
#include "../../../engine/cpp/agent.h"
#include "../../../engine/cpp/api_enums.h"
// Port of the accompanying official Notebook main.py.
namespace official::mega_abomasnow {
class Policy final : public Agent {
    const CardTableView& card_table;
    std::vector<int> my_deck;
    static constexpr int Kyogre = 721;
    static constexpr int Snover = 722;
    static constexpr int Mega_Abomasnow_ex = 723;
    static constexpr int Ultra_Ball = 1121;
    static constexpr int Precious_Trolley = 1126;
    static constexpr int Carmine = 1192;
    static constexpr int Lillie_Determination = 1227;
    static constexpr int Surfing_Beach = 1262;
    static constexpr int Basic_Water_Energy = 3;
public:
    Policy(const CardTableView& table, const std::vector<int>& deck) : card_table(table), my_deck(deck) {}
    std::vector<int> decide(const Observation& obs) override {

        int context{};
        int my_index{};

        std::map<int, int> field_counts{};
        std::map<int, int> hand_counts{};
        std::map<int, int> discard_counts{};
        int bench_attacker_index0{};
        int bench_attacker_index1{};
        int i{};
        CardView card{};
        int op_active_hp{};
        bool prefer_ky{};
        int switch_index{};
        std::vector<double> scores{};
        OptionView o{};
        double score{};
        int energy_count{};
        CardView pokemon{};
        std::vector<int> desc_indices{};
        // Policy order follows ../main.py:62.
        const auto& state = obs.current;
        const auto& select = obs.select;
        context = select.context;
        my_index = state.yourIndex;
        const auto& my_state = state.players.at(my_index);
        field_counts = {};
        hand_counts = {};
        discard_counts = {};
        bench_attacker_index0 = (-1);
        bench_attacker_index1 = (-1);
        for (i = 0; i < count_of(my_state.bench); ++i) {
            card = at_index(my_state.bench, i);
            field_counts[card.id] += 1;
            if ((card.id == Mega_Abomasnow_ex) && (count_of(card.energies) >= 2)) {
                bench_attacker_index0 = i;
            } else if ((card.id == Kyogre) && (count_of(card.energies) >= 1)) {
                bench_attacker_index1 = i;
            }
        }
        for (const auto& entry_2 : my_state.hand) {
            card = entry_2;
            hand_counts[card.id] += 1;
        }
        for (const auto& entry_3 : my_state.discard) {
            card = entry_3;
            discard_counts[card.id] += 1;
        }
        op_active_hp = 0;
        for (const auto& entry_4 : state.players.at(1 - my_index).active) {
            card = entry_4;
            if (card == nullptr) {
                continue;
            }
            op_active_hp = card.hp;
        }
        prefer_ky = (op_active_hp <= (20 * discard_counts[Basic_Water_Energy]));
        switch_index = (-1);
        for (const auto& entry_5 : my_state.active) {
            card = entry_5;
            if (card == nullptr) {
                continue;
            }
            field_counts[card.id] += 1;
            if ((card.id == Mega_Abomasnow_ex) && (count_of(card.energies) >= 2)) {
                if (prefer_ky && (bench_attacker_index1 >= 0)) {
                    switch_index = bench_attacker_index1;
                }
            } else if ((card.id == Kyogre) && (count_of(card.energies) >= 1)) {
                if ((!prefer_ky) && (bench_attacker_index0 >= 0)) {
                    switch_index = bench_attacker_index0;
                }
            } else if (bench_attacker_index0 >= 0) {
                switch_index = bench_attacker_index0;
            }
        }
        scores = {};
        for (const auto& entry_6 : select.option) {
            o = entry_6;
            score = 0;
            if (o.type == OptionType::NUMBER) {
                score = o.number;
            } else if (o.type == OptionType::YES) {
                score = 1;
            } else if (o.type == OptionType::CARD) {
                card = get_card(obs, o.area, o.index, o.playerIndex);
                if (card != nullptr) {
                    energy_count = 0;
                    if (card.is_pokemon) {
                        energy_count = count_of(card.energies);
                    }
                    if ((context == SelectContext::SWITCH) || (context == SelectContext::TO_ACTIVE) || (context == SelectContext::SETUP_ACTIVE_POKEMON)) {
                        score += (energy_count * 2);
                        if (o.index == switch_index) {
                            score += 100;
                        }
                        if (card.id == Mega_Abomasnow_ex) {
                            score += 20;
                        } else if (card.id == Kyogre) {
                            score += 10;
                        }
                    } else if ((context == SelectContext::TO_BENCH) || (context == SelectContext::TO_HAND)) {
                        if (card.id == Snover) {
                            if (field_counts[card.id] >= 1) {
                                score += 5;
                            } else if (field_counts[Mega_Abomasnow_ex] >= 1) {
                                score += 15;
                            } else {
                                score += 30;
                            }
                        } else if (card.id == Mega_Abomasnow_ex) {
                            if ((field_counts[Snover] >= 1) && ((field_counts[card.id] + hand_counts[card.id]) == 0)) {
                                score += 100;
                            } else {
                                score += 10;
                            }
                        } else if (card.id == Kyogre) {
                            if (field_counts[card.id] >= 1) {
                                score += 1;
                            } else {
                                score += 20;
                            }
                        }
                    } else if (context == SelectContext::DISCARD) {
                        if (card.id == Basic_Water_Energy) {
                            score += 100;
                        } else if (card.id == Mega_Abomasnow_ex) {
                            score += 10;
                        } else if (card.id == Carmine) {
                            if (hand_counts[Lillie_Determination] >= 1) {
                                score += 30;
                            }
                        } else if (card.id == Lillie_Determination) {
                            score -= 20;
                        }
                        if (hand_counts[card.id] >= 2) {
                            score += 500;
                        }
                        hand_counts[card.id] -= 1;
                    }
                }
            } else if (o.type == OptionType::PLAY) {
                card = get_card(obs, AreaType::HAND, o.index, my_index);
                score = 10000;
                if (card.id == Ultra_Ball) {
                    if ((hand_counts[Basic_Water_Energy] >= 3) || ((my_state.handCount >= 4) && (((field_counts[Mega_Abomasnow_ex] + hand_counts[Mega_Abomasnow_ex]) == 0) || ((field_counts[Mega_Abomasnow_ex] + field_counts[Snover]) == 0) || (field_counts[Kyogre] == 0)))) {
                        score = 4000;
                    } else {
                        score = (-1);
                    }
                } else if (card.id == Carmine) {
                    if ((field_counts[Snover] >= 1) && (hand_counts[Mega_Abomasnow_ex] >= 1)) {
                        score = (-1);
                    } else {
                        score = 3000;
                    }
                } else if (card.id == Lillie_Determination) {
                    if ((field_counts[Snover] >= 1) && (field_counts[Mega_Abomasnow_ex] == 0) && (hand_counts[Mega_Abomasnow_ex] >= 1)) {
                        score = (-1);
                    } else {
                        score = 3100;
                    }
                }
            } else if (o.type == OptionType::ATTACH) {
                pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index);
                score = 5000;
                energy_count = count_of(pokemon.energies);
                if (energy_count == 0) {
                    if (o.inPlayArea == AreaType::BENCH) {
                        score += 1;
                    }
                }
                if (pokemon.id == Snover) {
                    score += 1;
                    if (energy_count == 1) {
                        score -= 100;
                    } else if (energy_count >= 2) {
                        score -= 400;
                    }
                    if (bench_attacker_index0 >= 0) {
                        score -= 300;
                    }
                } else if (pokemon.id == Mega_Abomasnow_ex) {
                    score += 10;
                    if (energy_count == 1) {
                        score += 30;
                    } else if (energy_count >= 2) {
                        score -= 300;
                    }
                    if (bench_attacker_index0 >= 0) {
                        score -= 200;
                    }
                } else if (pokemon.id == Kyogre) {
                    score += 5;
                    if (count_of(pokemon.energies) >= 1) {
                        score -= 200;
                    }
                    if (bench_attacker_index1 >= 0) {
                        score -= 200;
                    }
                }
                if (o.inPlayArea == AreaType::ACTIVE) {
                    if ((bench_attacker_index0 >= 0) && (bench_attacker_index1 >= 0) && (energy_count <= 2)) {
                        score += 200;
                    }
                }
            } else if (o.type == OptionType::EVOLVE) {
                pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index);
                score = (10000 + count_of(pokemon.energies));
            } else if (o.type == OptionType::ABILITY) {
                card = get_card(obs, o.area, o.index, my_index);
                if ((card.id == Surfing_Beach) && (switch_index >= 0)) {
                    score = 2000;
                } else {
                    score = (-1);
                }
            } else if (o.type == OptionType::RETREAT) {
                if (switch_index >= 0) {
                    score = 1500;
                } else {
                    score = (-1);
                }
            } else if (o.type == OptionType::ATTACK) {
                score = 1000;
                if (o.attackId == 1042) {
                    score += ((discard_counts[Basic_Water_Energy] * 20) - 90);
                } else if (o.attackId == 1046) {
                    if (op_active_hp <= 200) {
                        score -= 100;
                    } else {
                        score += 100;
                    }
                }
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
    const std::map<int,int> expected{{3, 34}, {721, 2}, {722, 4}, {723, 4}, {1121, 4}, {1126, 1}, {1192, 4}, {1227, 4}, {1262, 3}};
    if(actual!=expected)throw std::runtime_error("Official deck mismatch: mega_abomasnow");
    return std::make_unique<Policy>(table,deck);
}
} // namespace official::mega_abomasnow
