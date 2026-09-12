#pragma once
#include "../../../engine/cpp/agent.h"
#include "../../../engine/cpp/api_enums.h"
// Port of the accompanying official Notebook main.py.
namespace official::mega_lucario {
class Policy final : public Agent {
    const CardTableView& card_table;
    std::vector<int> my_deck;
    static constexpr int Makuhita = 673;
    static constexpr int Hariyama = 674;
    static constexpr int Lunatone = 675;
    static constexpr int Solrock = 676;
    static constexpr int Riolu = 677;
    static constexpr int Mega_Lucario_ex = 678;
    static constexpr int Dusk_Ball = 1102;
    static constexpr int Switch = 1123;
    static constexpr int Premium_Power_Pro = 1141;
    static constexpr int Fighting_Gong = 1142;
    static constexpr int Poke_Pad = 1152;
    static constexpr int Hero_Cape = 1159;
    static constexpr int Boss_Orders = 1182;
    static constexpr int Carmine = 1192;
    static constexpr int Lillie_Determination = 1227;
    static constexpr int Gravity_Mountain = 1252;
    static constexpr int Basic_Fighting_Energy = 6;
    int pre_turn = 0;
    bool ability_used = false;
    struct AttackPlan { int attacker=-1, target=-1, attack_index=-1, remain_hp=-1; bool energy=false; };
    AttackPlan plan;
public:
    Policy(const CardTableView& table, const std::vector<int>& deck) : card_table(table), my_deck(deck) {}
    int prize_count(const CardView& pokemon) {
        MasterView data{};
        int count{};
        CardView card{};
        // Policy order follows ../main.py:84.
        data = card_table.at(pokemon.id);
        count = (data.megaEx ? 3 : (data.ex ? 2 : 1));
        for (const auto& entry_1 : pokemon.energyCards) {
            card = entry_1;
            if (card.id == 12) {
                count -= 1;
            }
        }
        for (const auto& entry_2 : pokemon.tools) {
            card = entry_2;
            if ((card.id == 1172) && (data.name.find("Lillie") != std::string::npos)) {
                count -= 1;
            }
        }
        return std::max<double>(0, count);
    }
    double pokemon_score(const CardView& pokemon) {
        MasterView data{};
        double score{};
        int id{};
        // Policy order follows ../main.py:97.
        data = card_table.at(pokemon.id);
        score = (prize_count(pokemon) * 1000);
        score += (count_of(pokemon.energies) * 150);
        score += (count_of(pokemon.tools) * 100);
        if (data.stage2) {
            score += 250;
        } else if (data.stage1) {
            score += 130;
        }
        id = pokemon.id;
        if ((id == 173) || (id == 174) || (id == 190) || (id == 1071)) {
            score -= 200;
        }
        if ((id == 112) && (count_of(pokemon.energies) >= 1)) {
            score += 300;
        }
        score += pokemon.hp;
        return score;
    }
    std::vector<int> decide(const Observation& obs) override {

        int context{};
        int my_index{};

        int my_prize{};
        std::map<int, int> field_counts{};
        std::map<int, int> hand_counts{};
        std::map<int, int> discard_counts{};
        bool attacker1{};
        bool attacker2{};
        CardView card{};
        int stadium_id{};
        bool can_attack{};
        bool can_switch{};
        bool can_op_switch{};
        bool can_use_mega_brave{};
        OptionView o{};
        std::vector<CardView> my_cards{};
        CardView pokemon{};
        std::vector<CardView> op_cards{};
        double best_score{};
        int i{};
        CardView my_pokemon{};
        int a{};
        int energy_required{};
        int base_damage{};
        double base_score{};
        int index{};
        bool more_energy{};
        int energy_count{};
        int j{};
        CardView op_pokemon{};
        int damage{};
        MasterView data{};
        int prize{};
        double score{};
        std::vector<double> scores{};
        std::vector<int> desc_indices{};
        // Policy order follows ../main.py:118.
        const auto& state = obs.current;
        const auto& select = obs.select;
        context = select.context;
        my_index = state.yourIndex;
        const auto& my_state = state.players.at(my_index);
        const auto& op_state = state.players.at(1 - my_index);
        my_prize = count_of(my_state.prize);
        if (pre_turn != state.turn) {
            pre_turn = state.turn;
            plan = AttackPlan{};
            ability_used = false;
        }
        field_counts = {};
        hand_counts = {};
        discard_counts = {};
        attacker1 = false;
        attacker2 = false;
        for (const auto& entry_3 : concat(my_state.active, my_state.bench)) {
            card = entry_3;
            if (card == nullptr) {
                continue;
            }
            field_counts[card.id] += 1;
            if ((card.id == Makuhita) || (card.id == Hariyama)) {
                if (count_of(card.energies) >= 3) {
                    attacker2 = true;
                }
            } else if ((card.id == Riolu) || (card.id == Mega_Lucario_ex)) {
                if (count_of(card.energies) >= 2) {
                    attacker1 = true;
                }
            }
        }
        for (const auto& entry_4 : my_state.hand) {
            card = entry_4;
            hand_counts[card.id] += 1;
        }
        for (const auto& entry_5 : my_state.discard) {
            card = entry_5;
            discard_counts[card.id] += 1;
        }
        stadium_id = 0;
        for (const auto& entry_6 : state.stadium) {
            card = entry_6;
            stadium_id = card.id;
        }
        can_attack = false;
        if (context == SelectContext::MAIN) {
            can_switch = false;
            can_op_switch = false;
            can_use_mega_brave = false;
            for (const auto& entry_7 : select.option) {
                o = entry_7;
                if (o.type == OptionType::PLAY) {
                    card = get_card(obs, AreaType::HAND, o.index, my_index);
                    if (card.id == Switch) {
                        can_switch = true;
                    } else if (card.id == Boss_Orders) {
                        can_op_switch = true;
                    }
                } else if (o.type == OptionType::EVOLVE) {
                    card = get_card(obs, AreaType::HAND, o.index, my_index);
                    if (card.id == Hariyama) {
                        can_op_switch = true;
                    }
                } else if (o.type == OptionType::RETREAT) {
                    can_switch = true;
                } else if (o.type == OptionType::ATTACK) {
                    can_attack = true;
                    if (o.attackId == 983) {
                        can_use_mega_brave = true;
                    }
                }
            }
            my_cards = {at_index(my_state.active, 0)};
            for (const auto& entry_8 : my_state.bench) {
                pokemon = entry_8;
                my_cards.push_back(pokemon);
            }
            op_cards = {at_index(op_state.active, 0)};
            for (const auto& entry_9 : op_state.bench) {
                pokemon = entry_9;
                op_cards.push_back(pokemon);
            }
            if (state.turn >= 2) {
                best_score = (-1);
                for (i = 0; i < count_of(my_cards); ++i) {
                    my_pokemon = at_index(my_cards, i);
                    if ((i != 0) && (!can_switch)) {
                        break;
                    }
                    for (a = 0; a < 2; ++a) {
                        energy_required = 0;
                        base_damage = 0;
                        base_score = 0;
                        if (my_pokemon.id == Mega_Lucario_ex) {
                            if (a == 0) {
                                energy_required = 1;
                                base_damage = 130;
                                base_score += (60 * std::min<double>(3, discard_counts[Basic_Fighting_Energy]));
                            } else {
                                energy_required = 2;
                                base_damage = 270;
                            }
                            if ((my_prize == 2) || (my_prize == 3)) {
                                base_score -= 500;
                            }
                        } else if (a == 1) {
                            break;
                        } else if (my_pokemon.id == Hariyama) {
                            energy_required = 3;
                            base_damage = 210;
                        } else if (my_pokemon.id == Makuhita) {
                            bool loop_completed_12 = true;
                            for (const auto& entry_12 : select.option) {
                                o = entry_12;
                                if (o.type == OptionType::EVOLVE) {
                                    index = o.inPlayIndex;
                                    if (o.inPlayArea == AreaType::BENCH) {
                                        index += 1;
                                    }
                                    if (index == i) {
                                        loop_completed_12 = false;
                                        break;
                                    }
                                }
                            }
                            if (loop_completed_12) {
                                break;
                            }
                            base_score -= 100;
                            energy_required = 3;
                            base_damage = 210;
                        } else if (my_pokemon.id == Solrock) {
                            if (field_counts[Lunatone] >= 1) {
                                energy_required = 1;
                                base_damage = 70;
                            }
                        }
                        if (base_damage <= 0) {
                            continue;
                        }
                        more_energy = false;
                        energy_count = count_of(my_pokemon.energies);
                        if ((a == 1) && (i == 0) && (energy_count >= 2) && (!can_use_mega_brave)) {
                            break;
                        }
                        if (energy_count < energy_required) {
                            if ((hand_counts[Basic_Fighting_Energy] >= 1) && (!state.energyAttached)) {
                                energy_count += 1;
                                if (energy_count < energy_required) {
                                    continue;
                                } else {
                                    more_energy = true;
                                }
                            } else {
                                continue;
                            }
                        }
                        for (j = 0; j < count_of(op_cards); ++j) {
                            op_pokemon = at_index(op_cards, j);
                            if ((j != 0) && (!can_op_switch)) {
                                break;
                            }
                            damage = base_damage;
                            data = card_table.at(op_pokemon.id);
                            if (data.weakness == EnergyType::FIGHTING) {
                                damage *= 2;
                            } else if (data.resistance == EnergyType::FIGHTING) {
                                damage -= 30;
                            }
                            prize = 0;
                            score = pokemon_score(op_pokemon);
                            if (op_pokemon.hp <= damage) {
                                prize = prize_count(op_pokemon);
                            } else {
                                score *= (static_cast<double>(damage) / op_pokemon.hp);
                            }
                            score += base_score;
                            if (count_of(op_state.prize) <= prize) {
                                score = 50000;
                            }
                            if (i == 0) {
                                score += 220;
                            }
                            if (j == 0) {
                                score += 300;
                            }
                            score += energy_count;
                            if (best_score < score) {
                                best_score = score;
                                plan.attacker = i;
                                plan.target = j;
                                plan.attack_index = a;
                                plan.remain_hp = (op_pokemon.hp - damage);
                                plan.energy = more_energy;
                            }
                        }
                    }
                }
            }
        }
        std::function<double(const CardView&, bool)> energy_score = [&](const CardView& pokemon, bool active) -> double {
            int energy_count{};
            double score{};
            // Policy order follows ../main.py:300.
            energy_count = count_of(pokemon.energies);
            score = 8000;
            if (active) {
                score += 10;
            }
            if ((pokemon.id == Makuhita) || (pokemon.id == Hariyama)) {
                if (pokemon.id == Hariyama) {
                    score += 1;
                }
                if (energy_count < 3) {
                    score += 100;
                }
                if (attacker2) {
                    score -= 50;
                }
            } else if (pokemon.id == Lunatone) {
                score -= 100;
            } else if (pokemon.id == Solrock) {
                if (energy_count < 1) {
                    score += 20;
                } else {
                    score -= 100;
                }
            } else if ((pokemon.id == Riolu) || (pokemon.id == Mega_Lucario_ex)) {
                if (pokemon.id == Mega_Lucario_ex) {
                    score += 1;
                }
                if (energy_count < 2) {
                    score += 100;
                }
                if (attacker1) {
                    score -= 50;
                }
            }
            return score;
        };
        scores = {};
        for (const auto& entry_14 : select.option) {
            o = entry_14;
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
                    if ((context == SelectContext::SWITCH) || (context == SelectContext::TO_ACTIVE)) {
                        if (o.playerIndex == my_index) {
                            score += (energy_count * 2);
                            if (o.index == (plan.attacker - 1)) {
                                score += 100;
                            }
                            if (card.id == Mega_Lucario_ex) {
                                if ((my_prize == 2) || (my_prize == 3)) {
                                    score += 8;
                                } else {
                                    score += 20;
                                }
                            } else if ((card.id == Hariyama) && (energy_count >= 2)) {
                                score += 15;
                            } else if ((card.id == Makuhita) && (energy_count >= 2)) {
                                score += 10;
                            } else if (card.id == Solrock) {
                                score += 5;
                            } else if (card.id == Riolu) {
                                score += 4;
                            }
                        } else if (o.index == (plan.target - 1)) {
                            score += 100;
                        }
                    } else if (context == SelectContext::SETUP_ACTIVE_POKEMON) {
                        if (card.id == Solrock) {
                            if (state.firstPlayer == my_index) {
                                score = 2;
                            } else {
                                score = 4;
                            }
                        } else if (card.id == Riolu) {
                            score = 3;
                        } else if (card.id == Makuhita) {
                            score = 1;
                        }
                    } else if (context == SelectContext::TO_HAND) {
                        score = (200 - (hand_counts[card.id] * 100));
                        if (card.id == Makuhita) {
                            if (field_counts[card.id] >= 1) {
                                score -= 10;
                            } else {
                                score += 10;
                            }
                        } else if (card.id == Hariyama) {
                            if (field_counts[Makuhita] >= 1) {
                                score += 20;
                            } else {
                                score -= 20;
                            }
                        } else if (card.id == Lunatone) {
                            if (field_counts[card.id] >= 1) {
                                score -= 250;
                            } else {
                                score += 60;
                            }
                        } else if (card.id == Solrock) {
                            if (field_counts[card.id] >= 1) {
                                score -= 250;
                            } else {
                                score += 50;
                            }
                        } else if (card.id == Riolu) {
                            if ((field_counts[card.id] + field_counts[Mega_Lucario_ex]) >= 2) {
                                score -= 150;
                            } else if ((field_counts[card.id] + field_counts[Mega_Lucario_ex]) >= 1) {
                                score -= 3;
                            } else {
                                score += 40;
                            }
                        } else if (card.id == Mega_Lucario_ex) {
                            if (field_counts[Riolu] >= 1) {
                                score += 40;
                            } else {
                                score -= 15;
                            }
                        } else if (card.id == Basic_Fighting_Energy) {
                            if ((!ability_used) || (!state.energyAttached)) {
                                score += 30;
                            } else {
                                score -= 1;
                            }
                        }
                    } else if (context == SelectContext::ATTACH_FROM) {
                        score = energy_score(card, (o.area == AreaType::ACTIVE));
                    }
                }
            } else if (o.type == OptionType::PLAY) {
                card = get_card(obs, AreaType::HAND, o.index, my_index);
                data = card_table.at(card.id);
                if (data.cardType == CardType::POKEMON) {
                    score = 20000;
                    if ((card.id == Lunatone) || (card.id == Solrock)) {
                        if (field_counts[card.id] >= 1) {
                            score = (-1);
                        }
                    } else if (card.id == Riolu) {
                        if ((field_counts[card.id] + field_counts[Mega_Lucario_ex]) >= 2) {
                            score = (-1);
                        }
                    }
                } else {
                    score = 10000;
                    if (card.id == Switch) {
                        if (plan.attacker <= 0) {
                            score = (-1);
                        } else {
                            score = 6000;
                        }
                    } else if (card.id == Premium_Power_Pro) {
                        if (state.supporterPlayed && (plan.remain_hp <= 0)) {
                            score = (-1);
                        } else if (!can_attack) {
                            if ((!state.supporterPlayed) && (hand_counts[Carmine] > 0) && (hand_counts[Lillie_Determination] == 0)) {
                                score = 3050;
                            } else {
                                score = (-1);
                            }
                        } else {
                            score = 5000;
                        }
                    } else if (card.id == Boss_Orders) {
                        if (plan.target >= 1) {
                            score = 3200;
                        } else {
                            score = (-1);
                        }
                    } else if (card.id == Carmine) {
                        score = 3000;
                    } else if (card.id == Lillie_Determination) {
                        score = 3100;
                    } else if (card.id == Gravity_Mountain) {
                        if (stadium_id == 0) {
                            score = (-1);
                        }
                    }
                }
            } else if (o.type == OptionType::ATTACH) {
                card = get_card(obs, AreaType::HAND, o.index, my_index);
                pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index);
                if (card.id == Hero_Cape) {
                    score = 7000;
                    if (pokemon.id == Riolu) {
                        score += 100;
                    } else if (pokemon.id == Mega_Lucario_ex) {
                        score += 200;
                    }
                } else {
                    score = energy_score(pokemon, (o.inPlayArea == AreaType::ACTIVE));
                    if (o.inPlayArea == AreaType::ACTIVE) {
                        if ((plan.attacker == 0) && plan.energy) {
                            score += 200;
                        }
                    } else if ((plan.attacker == (1 + o.inPlayIndex)) && plan.energy) {
                        score += 200;
                    }
                }
            } else if (o.type == OptionType::EVOLVE) {
                pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index);
                score = (9000 + count_of(pokemon.energies));
                if ((pokemon.id == Makuhita) && (plan.target == 0)) {
                    score = (-1);
                }
            } else if (o.type == OptionType::ABILITY) {
                card = get_card(obs, o.area, o.index, my_index);
                if (card.id == 1267) {
                    score = 1;
                } else {
                    score = 30000;
                }
            } else if (o.type == OptionType::RETREAT) {
                if (plan.attacker >= 1) {
                    score = 2000;
                } else {
                    score = (-1);
                }
            } else if (o.type == OptionType::ATTACK) {
                score = 1000;
                if (plan.attack_index == 1) {
                    if (o.attackId == 983) {
                        score += 100;
                    }
                } else if (o.attackId != 983) {
                    score += 100;
                }
            }
            scores.push_back(score);
        }
        desc_indices = ranked_indices(scores);
        if (context == SelectContext::MAIN) {
            o = at_index(select.option, at_index(desc_indices, 0));
            if (o.type == OptionType::ABILITY) {
                card = get_card(obs, o.area, o.index, my_index);
                if (card.id == Lunatone) {
                    ability_used = true;
                }
            }
        }
        return take(desc_indices, select.maxCount);
    }
};
} // namespace official::mega_lucario
