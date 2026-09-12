#pragma once
#include "../../../engine/cpp/agent.h"
#include "../../../engine/cpp/api_enums.h"
// Port of the accompanying official Notebook main.py.
namespace official::dragapult {
class Policy final : public Agent {
    const CardTableView& card_table;
    std::vector<int> my_deck;
    static constexpr int Dreepy = 119;
    static constexpr int Drakloak = 120;
    static constexpr int Dragapult_ex = 121;
    static constexpr int Fezandipiti_ex = 140;
    static constexpr int Latias_ex = 184;
    static constexpr int Budew = 235;
    static constexpr int Meowth_ex = 1071;
    static constexpr int Rare_Candy = 1079;
    static constexpr int Unfair_Stamp = 1080;
    static constexpr int Buddy_Buddy_Poffin = 1086;
    static constexpr int Night_Stretcher = 1097;
    static constexpr int Crushing_Hammer = 1120;
    static constexpr int Ultra_Ball = 1121;
    static constexpr int Poke_Pad = 1152;
    static constexpr int Lucky_Helmet = 1156;
    static constexpr int Boss_Orders = 1182;
    static constexpr int Crispin = 1198;
    static constexpr int Brock_Scouting = 1210;
    static constexpr int Lillie_Determination = 1227;
    static constexpr int Team_Rocket_Watchtower = 1256;
    static constexpr int Basic_Fire_Energy = 2;
    static constexpr int Basic_Psychic_Energy = 5;
    static constexpr int UNNECESSARY = (-10000000);
    bool can_switch = false;
    bool can_attack = false;
    bool can_main_attack = false;
    bool can_energy_attach = false;
    int use_support = 0;
    bool bench_attacker = false;
    std::vector<LogView> pre_turn_log;
    std::vector<LogView> current_turn_log;
    std::vector<int> prize;
    std::map<int, int> card_counts;
    std::set<int> serial_set;
    struct AttackPlan { int attack=0; std::vector<int> counter; };
    AttackPlan plan_a, plan_b;
public:
    Policy(const CardTableView& table, const std::vector<int>& deck) : card_table(table), my_deck(deck) {}
    bool no_damage_dex(int id) {
        // Policy order follows ../main.py:74.
        return ((id == 158) || (id == 207) || (id == 330) || (id == 345));
    }
    bool no_damage_counter(const CardView& pokemon) {
        CardView card{};
        // Policy order follows ../main.py:80.
        if ((pokemon.id == 28) || (pokemon.id == 199) || (pokemon.id == 203) || (pokemon.id == 207) || (pokemon.id == 362) || (pokemon.id == 1136)) {
            return true;
        }
        for (const auto& entry_1 : pokemon.energyCards) {
            card = entry_1;
            if ((card.id == 11) || (card.id == 20)) {
                return true;
            }
        }
        return false;
    }
    int prize_count(const CardView& pokemon, bool is_attack_damage) {
        MasterView data{};
        int count{};
        CardView card{};
        // Policy order follows ../main.py:92.
        data = card_table.at(pokemon.id);
        count = (data.megaEx ? 3 : (data.ex ? 2 : 1));
        if (is_attack_damage) {
            for (const auto& entry_2 : pokemon.energyCards) {
                card = entry_2;
                if (card.id == 12) {
                    count -= 1;
                }
            }
            for (const auto& entry_3 : pokemon.tools) {
                card = entry_3;
                if ((card.id == 1172) && (data.name.find("Lillie") != std::string::npos)) {
                    count -= 1;
                }
            }
        }
        return std::max<double>(0, count);
    }
    double pokemon_score(const CardView& pokemon, bool is_attack_damage) {
        MasterView data{};
        double score{};
        int id{};
        // Policy order follows ../main.py:106.
        data = card_table.at(pokemon.id);
        score = (prize_count(pokemon, is_attack_damage) * 1000);
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
    void add_card_count(const CardView& card, int my_index) {
        CardView c{};
        // Policy order follows ../main.py:127.
        if (card == nullptr) {
            return;
        }
        if (card.is_pokemon || (card.playerIndex == my_index)) {
            if (!serial_set.contains(card.serial)) {
                card_counts[card.id] -= 1;
                serial_set.insert(card.serial);
            }
        }
        if (card.is_pokemon) {
            for (const auto& entry_4 : card.energyCards) {
                c = entry_4;
                add_card_count(c, my_index);
            }
            for (const auto& entry_5 : card.tools) {
                c = entry_5;
                add_card_count(c, my_index);
            }
            for (const auto& entry_6 : card.preEvolution) {
                c = entry_6;
                add_card_count(c, my_index);
            }
        }
    }
    void set_card_counts(const Observation& obs, int my_index) {
        int id{};

        CardView card{};
        // Policy order follows ../main.py:142.
        card_counts.clear();
        serial_set.clear();
        for (const auto& entry_7 : my_deck) {
            id = entry_7;
            card_counts[id] += 1;
        }
        const auto& state = obs.current;
        const auto& my_state = state.players.at(my_index);
        for (const auto& entry_8 : my_state.hand) {
            card = entry_8;
            add_card_count(card, my_index);
        }
        for (const auto& entry_9 : my_state.discard) {
            card = entry_9;
            add_card_count(card, my_index);
        }
        for (const auto& entry_10 : my_state.bench) {
            card = entry_10;
            add_card_count(card, my_index);
        }
        for (const auto& entry_11 : my_state.active) {
            card = entry_11;
            add_card_count(card, my_index);
        }
        for (const auto& entry_12 : state.stadium) {
            card = entry_12;
            add_card_count(card, my_index);
        }
        if (state.has_looking) {
            for (const auto& entry_13 : state.looking) {
                card = entry_13;
                add_card_count(card, my_index);
            }
        }
        add_card_count(obs.select.effect, my_index);
    }
    void main_option_proc(const Observation& obs, int damage) {

        int my_index{};

        OptionView o{};
        std::vector<CardView> cards{};
        CardView pokemon{};
        std::vector<std::vector<int>> counter_indices{};
        std::vector<int> ci{};
        int remain_damage{};
        int index{};
        int hp{};
        int remain_prize{};
        double plan_score{};
        int i{};
        int base_prize_count{};
        double base_score{};
        int active_damage{};
        double max_score{};
        std::vector<int> indices{};
        int prize{};
        double score{};
        // Policy order follows ../main.py:189.
        const auto& state = obs.current;
        const auto& select = obs.select;
        my_index = state.yourIndex;
        const auto& my_state = state.players.at(my_index);
        const auto& op_state = state.players.at(1 - my_index);
        can_switch = false;
        can_attack = false;
        can_main_attack = false;
        can_energy_attach = false;
        for (const auto& entry_14 : select.option) {
            o = entry_14;
            if (o.type == OptionType::RETREAT) {
                can_switch = true;
            } else if (o.type == OptionType::ATTACK) {
                can_attack = true;
                if (o.attackId == 154) {
                    can_main_attack = true;
                }
            }
        }
        plan_a.attack = (-1);
        plan_b.attack = (-1);
        if ((!can_main_attack) && (!(bench_attacker && can_switch))) {
            return;
        }
        cards = {at_index(op_state.active, 0)};
        for (const auto& entry_15 : op_state.bench) {
            pokemon = entry_15;
            cards.push_back(pokemon);
        }
        counter_indices = {};
        ci = {};
        ci.push_back(0);
        remain_damage = 60;
        while (!ci.empty()) {
            index = at_index(ci, (-1));
            hp = at_index(cards, index).hp;
            if (remain_damage >= hp) {
                counter_indices.push_back(ci);
                if (index < (count_of(cards) - 1)) {
                    remain_damage -= hp;
                    ci.push_back(index + 1);
                    continue;
                }
            }
            if (index == (count_of(cards) - 1)) {
                ci.pop_back();
                if (!ci.empty()) {
                    remain_damage += at_index(cards, at_index(ci, (-1))).hp;
                }
            }
            if (!ci.empty()) {
                at_index(ci, (-1)) += 1;
            }
        }
        counter_indices.push_back({});
        remain_prize = count_of(my_state.prize);
        plan_score = 0;
        for (i = 0; i < count_of(cards); ++i) {
            pokemon = at_index(cards, i);
            base_prize_count = 0;
            base_score = pokemon_score(pokemon, true);
            active_damage = (no_damage_dex(pokemon.id) ? 0 : damage);
            if (pokemon.hp <= active_damage) {
                base_prize_count += prize_count(pokemon, true);
            } else {
                base_score *= (static_cast<double>(active_damage) / pokemon.hp);
            }
            ci = {};
            max_score = base_score;
            if (remain_prize <= base_prize_count) {
                max_score = 50000;
            } else {
                for (const auto& entry_17 : counter_indices) {
                    indices = entry_17;
                    if (contains(indices, i)) {
                        continue;
                    }
                    prize = base_prize_count;
                    score = base_score;
                    for (const auto& entry_18 : indices) {
                        index = entry_18;
                        prize += prize_count(at_index(cards, index), false);
                        score += pokemon_score(at_index(cards, index), false);
                    }
                    if (remain_prize <= prize) {
                        score = 50000;
                    } else if (prize >= 2) {
                        if (remain_prize <= 4) {
                            score -= 1200;
                        }
                    } else if (prize == 1) {
                        score -= 300;
                    } else {
                        score += 1200;
                    }
                    if (max_score < score) {
                        max_score = score;
                        ci = indices;
                    }
                }
            }
            if (plan_score < max_score) {
                plan_score = max_score;
                plan_a.attack = i;
                plan_a.counter = ci;
            }
            if (i == 0) {
                plan_b.attack = plan_a.attack;
                plan_b.counter = plan_a.counter;
            }
        }
    }
    std::vector<int> decide(const Observation& obs) override {

        int context{};
        int my_index{};

        LogView log{};
        bool pre_ko{};
        bool no_item{};
        CardView card{};
        int id{};
        int _{};
        std::map<int, int> deck_counts{};
        int prize_diff{};
        std::map<int, int> field_counts{};
        std::map<int, int> hand_counts{};
        std::map<int, int> discard_counts{};
        int active_id{};
        bool can_evolve_dreepy{};
        int evolve_dreepy_count{};
        bool can_evolve_drakloak{};
        int damage{};
        int main_pokemon_count{};
        bool no_more_dex{};
        int stadium_id{};
        int support_count{};
        double support_score{};
        OptionView o{};
        double score{};
        std::vector<double> hand_scores{};
        int negative_hand_count{};
        bool no_draw{};
        bool do_switch{};
        int effect_card_id{};
        int context_card_id{};
        std::vector<double> scores{};
        int energy_count{};
        int hp{};
        int index{};
        int remain_damage{};
        double card_score{};
        CardView pokemon{};
        std::vector<int> output{};
        std::vector<std::pair<int, double>> sorted_scores{};
        int i{};
        // Policy order follows ../main.py:286.
        const auto& state = obs.current;
        const auto& select = obs.select;
        context = select.context;
        my_index = state.yourIndex;
        const auto& my_state = state.players.at(my_index);
        const auto& op_state = state.players.at(1 - my_index);
        if (state.turn == 0) {
            prize.clear();
            pre_turn_log.clear();
            current_turn_log.clear();
        } else {
            for (const auto& entry_19 : obs.logs) {
                log = entry_19;
                current_turn_log.push_back(log);
                if (log.type == LogType::TURN_END) {
                    pre_turn_log = current_turn_log;
                    current_turn_log = {};
                }
            }
        }
        pre_ko = false;
        no_item = false;
        for (const auto& entry_20 : pre_turn_log) {
            log = entry_20;
            if (log.type == LogType::ATTACK) {
                if (log.attackId == 323) {
                    no_item = true;
                }
            } else if (log.type == LogType::MOVE_CARD) {
                if ((log.playerIndex == my_index) && ((log.fromArea == AreaType::BENCH) || (log.fromArea == AreaType::ACTIVE)) && (log.toArea == AreaType::DISCARD)) {
                    pre_ko = true;
                }
            }
        }
        if (select.has_deck) {
            set_card_counts(obs, my_index);
            for (const auto& entry_21 : select.deck) {
                card = entry_21;
                card_counts[card.id] -= 1;
            }
            prize.clear();
            for (const auto& entry_22 : card_counts) {
                id = entry_22.first;
                for (_ = 0; _ < card_counts[id]; ++_) {
                    prize.push_back(id);
                }
            }
        }
        set_card_counts(obs, my_index);
        for (const auto& entry_24 : prize) {
            id = entry_24;
            card_counts[id] -= 1;
        }
        deck_counts = card_counts;
        prize_diff = (count_of(my_state.prize) - count_of(op_state.prize));
        field_counts = {};
        hand_counts = {};
        discard_counts = {};
        active_id = 0;
        bench_attacker = false;
        can_evolve_dreepy = false;
        evolve_dreepy_count = 0;
        can_evolve_drakloak = false;
        damage = 200;
        for (const auto& entry_25 : my_state.active) {
            card = entry_25;
            if (card == nullptr) {
                continue;
            }
            active_id = card.id;
            field_counts[card.id] += 1;
            if (!card.appearThisTurn) {
                if (card.id == Dreepy) {
                    can_evolve_dreepy = true;
                    evolve_dreepy_count += 1;
                } else if (card.id == Drakloak) {
                    can_evolve_drakloak = true;
                }
            }
        }
        for (const auto& entry_26 : my_state.bench) {
            card = entry_26;
            field_counts[card.id] += 1;
            if (!card.appearThisTurn) {
                if (card.id == Dreepy) {
                    can_evolve_dreepy = true;
                    evolve_dreepy_count += 1;
                } else if (card.id == Drakloak) {
                    can_evolve_drakloak = true;
                }
            }
            if ((card.id == Dragapult_ex) && (count_of(card.energies) >= 2)) {
                bench_attacker = true;
            }
        }
        main_pokemon_count = ((field_counts[Dreepy] + field_counts[Drakloak]) + field_counts[Dragapult_ex]);
        no_more_dex = ((field_counts[Dragapult_ex] * 2) >= count_of(op_state.prize));
        stadium_id = 0;
        for (const auto& entry_27 : state.stadium) {
            card = entry_27;
            stadium_id = card.id;
        }
        support_count = 0;
        for (const auto& entry_28 : my_state.discard) {
            card = entry_28;
            discard_counts[card.id] += 1;
        }
        std::function<double(int, const CardView&, bool)> attach_score = [&](int attach_id, const CardView& pokemon, bool active) -> double {
            int energy_count{};
            double score{};
            // Policy order follows ../main.py:399.
            energy_count = count_of(pokemon.energies);
            if (card_table.at(attach_id).cardType == CardType::TOOL) {
                score = 60000;
                if (active) {
                    score += 1000;
                }
                return score;
            }
            if (pokemon.id == Budew) {
                return (-1);
            } else if ((pokemon.id == Meowth_ex) || (pokemon.id == Fezandipiti_ex) || (pokemon.id == Latias_ex)) {
                if (active && (!can_switch) && (!my_state.asleep) && (!my_state.paralyzed)) {
                    if (bench_attacker || (field_counts[Budew] >= 1)) {
                        return 22000;
                    } else {
                        return 18000;
                    }
                } else {
                    return (-1);
                }
            }
            if (active && can_main_attack) {
                return (-1);
            }
            score = 20000;
            if (energy_count >= 2) {
                if (active && (!can_switch) && (!my_state.asleep) && (!my_state.paralyzed)) {
                    score += 200;
                } else {
                    return (-1);
                }
            } else if (energy_count == 1) {
                if (attach_id == at_index(pokemon.energyCards, 0).id) {
                    return (-1);
                }
                if (pokemon.id == Dragapult_ex) {
                    score += 250;
                } else if (pokemon.id == Dreepy) {
                    score -= 150;
                } else {
                    score -= 200;
                }
                if (active) {
                    score += 200;
                }
            } else if (active) {
                if (bench_attacker) {
                    score += 400;
                }
            } else {
                if (pokemon.id == Dragapult_ex) {
                    score += 150;
                } else if (pokemon.id == Dreepy) {
                    score += 100;
                } else {
                    score += 50;
                }
                if (bench_attacker) {
                    score -= 200;
                }
            }
            if (no_more_dex && ((pokemon.id == Dreepy) || (pokemon.id == Drakloak))) {
                score -= 500;
            }
            return score;
        };
        std::function<double(int, bool)> hand_score = [&](int id, bool ignore_count) -> double {
            double score{};
            int count{};
            int i{};
            int card_type{};
            double max_score{};
            CardView pokemon{};
            // Policy order follows ../main.py:455.
            score = 0;
            if (id == Dreepy) {
                if (main_pokemon_count >= 3) {
                    score = 1000;
                } else {
                    score = 18000;
                }
            } else if (id == Drakloak) {
                if (can_evolve_dreepy) {
                    score = 20000;
                } else {
                    score = 3000;
                }
            } else if (id == Dragapult_ex) {
                if (no_more_dex) {
                    score = UNNECESSARY;
                } else if (can_evolve_dreepy && (hand_counts[Rare_Candy] >= 1) && (!no_item)) {
                    score = 40000;
                } else if (can_evolve_drakloak) {
                    if (field_counts[id] == 0) {
                        score = 30000;
                    } else if (field_counts[id] == 1) {
                        score = 10000;
                    } else {
                        score = 50;
                    }
                } else if (field_counts[id] >= 2) {
                    score = 50;
                } else {
                    score = 2000;
                }
            } else if (id == Fezandipiti_ex) {
                if (pre_ko) {
                    score = 50000;
                } else if (prize_diff <= (-2)) {
                    score = 5;
                } else if (count_of(op_state.prize) == 1) {
                    score = UNNECESSARY;
                }
            } else if (id == Latias_ex) {
                if ((active_id == Fezandipiti_ex) || (active_id == Meowth_ex) || (active_id == Dreepy)) {
                    if ((field_counts[Drakloak] + field_counts[Dragapult_ex]) == 0) {
                        score = 28000;
                    } else {
                        score = 15000;
                    }
                } else {
                    score = 10;
                }
            } else if (id == Budew) {
                if (((field_counts[id] + field_counts[Drakloak]) + field_counts[Dragapult_ex]) >= 1) {
                    score = UNNECESSARY;
                } else if (state.turn >= 2) {
                    score = 30000;
                }
            } else if (id == Meowth_ex) {
                if ((support_count > hand_counts[Boss_Orders]) || (stadium_id == Team_Rocket_Watchtower)) {
                    score = 5;
                } else if (state.supporterPlayed) {
                    score = 40;
                } else {
                    score = 35000;
                }
            } else if (id == Rare_Candy) {
                if (no_more_dex) {
                    score = UNNECESSARY;
                } else if (can_evolve_dreepy && (hand_counts[Dragapult_ex] >= 1)) {
                    score = 40000;
                }
            } else if (id == Unfair_Stamp) {
                if (pre_ko) {
                    score = 80000;
                } else if (count_of(op_state.prize) == 1) {
                    score = UNNECESSARY;
                } else {
                    score = 80;
                }
            } else if (id == Buddy_Buddy_Poffin) {
                count = deck_counts[Dreepy];
                if (count == 0) {
                    score = UNNECESSARY;
                } else {
                    if ((state.turn <= 2) && (field_counts[Budew] == 0) && (deck_counts[Budew] >= 1)) {
                        count += 1;
                    }
                    if (count >= 2) {
                        score = 35000;
                    }
                }
            } else if (id == Night_Stretcher) {
                for (const auto& entry_29 : discard_counts) {
                    i = entry_29.first;
                    if (discard_counts[i] >= 1) {
                        card_type = card_table.at(i).cardType;
                        if ((card_type == CardType::POKEMON) || (card_type == CardType::BASIC_ENERGY)) {
                            score = std::max<double>(score, hand_score(i, ignore_count));
                        }
                    }
                }
            } else if (id == Crushing_Hammer) {
                score = 20;
            } else if (id == Ultra_Ball) {
                if ((main_pokemon_count <= 2) || (field_counts[Dreepy] >= 1)) {
                    score = 70;
                } else {
                    score = 5;
                }
            } else if (id == Poke_Pad) {
                score = std::max<double>(hand_score(Dreepy, ignore_count), hand_score(Drakloak, ignore_count));
            } else if (id == Lucky_Helmet) {
                score = 15;
            } else if (id == Boss_Orders) {
                if (plan_a.attack > 0) {
                    score = 60000;
                }
            } else if (id == Crispin) {
                if ((!ignore_count) || (support_count == 0)) {
                    if ((deck_counts[Basic_Fire_Energy] == 0) || (deck_counts[Basic_Psychic_Energy] == 0)) {
                        score = 10;
                    }
                    if ((!can_main_attack) && (!bench_attacker) && (field_counts[Dragapult_ex] >= 1)) {
                        score = 55000;
                    } else {
                        score = 25000;
                    }
                }
            } else if (id == Brock_Scouting) {
                if ((!ignore_count) || (support_count == 0)) {
                    if ((state.turn == 2) && ((field_counts[Budew] + field_counts[Latias_ex]) == 0)) {
                        score = 50000;
                    } else {
                        score = 30000;
                    }
                }
            } else if (id == Lillie_Determination) {
                if ((!ignore_count) || (support_count == 0)) {
                    score = 45000;
                }
            } else if (id == Team_Rocket_Watchtower) {
                if ((stadium_id != 0) && (stadium_id != Team_Rocket_Watchtower)) {
                    score = 4000;
                }
            } else if ((id == Basic_Fire_Energy) || (id == Basic_Psychic_Energy)) {
                if (can_main_attack && ((count_of(op_state.prize) <= 2) || (bench_attacker && (count_of(op_state.prize) <= 4)))) {
                    score = UNNECESSARY;
                } else {
                    max_score = (-10000);
                    for (const auto& entry_30 : my_state.active) {
                        pokemon = entry_30;
                        if (pokemon == nullptr) {
                            continue;
                        }
                        max_score = std::max<double>(max_score, attach_score(id, pokemon, true));
                    }
                    for (const auto& entry_31 : my_state.bench) {
                        pokemon = entry_31;
                        max_score = std::max<double>(max_score, attach_score(id, pokemon, false));
                    }
                    score = (max_score - 5000);
                    if (can_main_attack || bench_attacker) {
                        score /= 10;
                    }
                }
            }
            if ((!ignore_count) && (hand_counts[id] > 0)) {
                if ((id == Drakloak) && (hand_counts[id] < evolve_dreepy_count)) {
                    score -= 10;
                } else if (id == Dreepy) {
                    score -= 100;
                } else {
                    score -= 100000;
                }
            }
            return score;
        };
        if (context == SelectContext::MAIN) {
            main_option_proc(obs, damage);
            use_support = 0;
            if (!state.supporterPlayed) {
                support_score = 0;
                for (const auto& entry_32 : select.option) {
                    o = entry_32;
                    if (o.type == OptionType::PLAY) {
                        card = get_card(obs, AreaType::HAND, o.index, state.yourIndex);
                        if (card_table.at(card.id).cardType == CardType::SUPPORTER) {
                            score = hand_score(card.id, true);
                            if (support_score < score) {
                                support_score = score;
                                use_support = card.id;
                            }
                        }
                    }
                }
            }
        }
        hand_scores = {};
        negative_hand_count = 0;
        for (const auto& entry_33 : my_state.hand) {
            card = entry_33;
            score = hand_score(card.id, false);
            hand_scores.push_back(score);
            if (score < 0) {
                negative_hand_count += 1;
            }
            hand_counts[card.id] += 1;
            if ((card_table.at(card.id).cardType == CardType::SUPPORTER) && (card.id != Boss_Orders)) {
                support_count += 1;
            }
        }
        no_draw = (my_state.deckCount <= 8);
        do_switch = ((!can_main_attack) && (bench_attacker || ((active_id != Budew) && (field_counts[Budew] >= 1) && (state.turn >= 2))));
        effect_card_id = ((select.effect == nullptr) ? 0 : select.effect.id);
        context_card_id = ((select.contextCard == nullptr) ? 0 : select.contextCard.id);
        scores = {};
        for (const auto& entry_34 : select.option) {
            o = entry_34;
            score = 0;
            if (o.type == OptionType::NUMBER) {
                score = o.number;
            } else if (o.type == OptionType::YES) {
                if (context == SelectContext::IS_FIRST) {
                    score = (-1);
                } else {
                    score = 1;
                }
            } else if (o.type == OptionType::CARD) {
                card = get_card(obs, o.area, o.index, o.playerIndex);
                if (card != nullptr) {
                    energy_count = 0;
                    hp = 0;
                    if (card.is_pokemon) {
                        energy_count = count_of(card.energies);
                        hp = card.hp;
                    }
                    if ((context == SelectContext::SWITCH) || (context == SelectContext::TO_ACTIVE) || (context == SelectContext::SETUP_ACTIVE_POKEMON)) {
                        if (o.playerIndex == my_index) {
                            if (card.id == Dreepy) {
                                score += 10000;
                            } else if (card.id == Drakloak) {
                                if (energy_count >= 1) {
                                    score += 20000;
                                } else {
                                    score -= 10000;
                                }
                            } else if (card.id == Dragapult_ex) {
                                score += 50000;
                            } else if (card.id == Budew) {
                                if (context != SelectContext::SWITCH) {
                                    score += 100000;
                                } else if (!bench_attacker) {
                                    score += 30000;
                                }
                            } else if (card.id == Fezandipiti_ex) {
                                score -= 1000;
                            } else if (card.id == Meowth_ex) {
                                score -= 2000;
                            }
                        } else if (plan_a.attack == (o.index + 1)) {
                            score += 100000;
                        }
                        score += (energy_count * 1000);
                        score += hp;
                    } else if (context == SelectContext::SETUP_BENCH_POKEMON) {
                        if ((my_index == state.firstPlayer) || (card.id != Dreepy)) {
                            score = (-1);
                        }
                    } else if ((context == SelectContext::TO_BENCH) || (context == SelectContext::TO_HAND)) {
                        score = hand_score(card.id, false);
                        hand_counts[card.id] += 1;
                        if (effect_card_id == Crispin) {
                            score = (100000 - hand_score(card.id, true));
                        }
                    } else if (context == SelectContext::DISCARD) {
                        hand_counts[card.id] -= 1;
                        if (card_table.at(card.id).cardType == CardType::SUPPORTER) {
                            support_count -= 1;
                        }
                        score = (-hand_score(card.id, false));
                    } else if ((context == SelectContext::DAMAGE_COUNTER) || (context == SelectContext::DAMAGE_COUNTER_ANY)) {
                        if (hp > 0) {
                            score = ((100000 - (10 * hp)) + pokemon_score(card, false));
                            if (context == SelectContext::DAMAGE_COUNTER) {
                                if ((210 <= hp) && (hp <= 230)) {
                                    score += (20000 + (hp * 20));
                                    if (o.area == AreaType::ACTIVE) {
                                        score += 10000;
                                    }
                                } else if ((40 <= hp) && (hp <= 90)) {
                                    score += (10000 + (hp * 20));
                                } else if (hp <= 30) {
                                    score += ((-10000) + (hp * 20));
                                }
                                if ((card.id == 133) || (card.id == 351)) {
                                    score += 30000;
                                }
                            } else {
                                index = (o.index + 1);
                                if (contains(plan_b.counter, index)) {
                                    score += 100000;
                                } else {
                                    remain_damage = (select.remainDamageCounter * 10);
                                    if ((210 <= hp) && (hp <= (200 + remain_damage))) {
                                        score += 30000;
                                    } else if ((20 <= hp) && (hp <= (60 + remain_damage))) {
                                        score += 10000;
                                    } else if (hp == 10) {
                                        score -= 100000;
                                    }
                                }
                                if (no_damage_counter(card)) {
                                    score = (-1);
                                }
                            }
                        }
                    } else if (context == SelectContext::ATTACH_FROM) {
                        score = attach_score(context_card_id, card, (o.area == AreaType::ACTIVE));
                        if (card.id == Dragapult_ex) {
                            score += 200;
                        }
                    }
                }
            } else if ((o.type == OptionType::ENERGY_CARD) || (o.type == OptionType::ENERGY)) {
                if (o.playerIndex != state.yourIndex) {
                    if (o.area == AreaType::BENCH) {
                        score = 20;
                    } else {
                        score = 10;
                    }
                    card = get_card(obs, o.area, o.index, o.playerIndex);
                    if (card_table.at(card.id).cardType == CardType::SPECIAL_ENERGY) {
                        score += 1;
                    }
                }
            } else if (o.type == OptionType::PLAY) {
                card = get_card(obs, AreaType::HAND, o.index, my_index);
                card_score = at_index(hand_scores, o.index);
                if (card.id == Dreepy) {
                    score = 51000;
                } else if (card.id == Fezandipiti_ex) {
                    if (card_score > 0) {
                        score = 53000;
                    } else {
                        score = (-1);
                    }
                } else if (card.id == Latias_ex) {
                    if ((active_id != Drakloak) && (active_id != Dragapult_ex)) {
                        score = 51000;
                    } else {
                        score = (-1);
                    }
                } else if (card.id == Budew) {
                    if ((field_counts[Budew] == 0) && (field_counts[Dragapult_ex] == 0)) {
                        score = 52000;
                    } else {
                        score = (-1);
                    }
                } else if (card.id == Meowth_ex) {
                    if (state.supporterPlayed || (stadium_id == Team_Rocket_Watchtower)) {
                        score = (-1);
                    } else if (support_count == 0) {
                        score = 50000;
                    } else if ((support_count == hand_counts[Boss_Orders]) && (!(plan_a.attack <= 0))) {
                        score = 50000;
                    } else {
                        score = (-1);
                    }
                } else if (card.id == Rare_Candy) {
                    if (no_more_dex) {
                        score = (-1);
                    } else {
                        score = 75000;
                    }
                } else if (card.id == Unfair_Stamp) {
                    score = 15000;
                } else if (card.id == Night_Stretcher) {
                    if (card_score >= 18000) {
                        score = 42000;
                    } else {
                        score = (-1);
                    }
                } else if (card.id == Crushing_Hammer) {
                    score = 40000;
                } else if (card.id == Boss_Orders) {
                    if (card.id == use_support) {
                        score = 35000;
                    } else {
                        score = (-1);
                    }
                } else if (card.id == Lillie_Determination) {
                    if (card.id == use_support) {
                        score = 14000;
                    } else {
                        score = (-1);
                    }
                } else if (card.id == Team_Rocket_Watchtower) {
                    if ((stadium_id > 0) || (state.turn == 1)) {
                        score = 80000;
                    } else {
                        score = (-1);
                    }
                } else if (no_draw) {
                    score = (-1);
                } else if (card.id == Buddy_Buddy_Poffin) {
                    if (deck_counts[Dreepy] > 0) {
                        score = 46000;
                    } else {
                        score = (-1);
                    }
                } else if (card.id == Ultra_Ball) {
                    if (negative_hand_count >= 2) {
                        score = 44000;
                    } else {
                        score = (-1);
                    }
                } else if (card.id == Poke_Pad) {
                    if ((deck_counts[Dreepy] + deck_counts[Drakloak]) > 0) {
                        score = 45000;
                    } else {
                        score = (-1);
                    }
                } else if ((card.id == Crispin) || (card.id == Brock_Scouting)) {
                    if (card.id == use_support) {
                        score = 35000;
                    } else {
                        score = (-1);
                    }
                }
            } else if (o.type == OptionType::ATTACH) {
                card = get_card(obs, o.area, o.index, my_index);
                pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index);
                score = attach_score(card.id, pokemon, (o.inPlayArea == AreaType::ACTIVE));
            } else if (o.type == OptionType::EVOLVE) {
                pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index);
                score += count_of(pokemon.energies);
                if (pokemon.id == Dreepy) {
                    score += 30000;
                } else if ((field_counts[Dragapult_ex] >= 2) || ((field_counts[Dragapult_ex] == 1) && (count_of(op_state.prize) <= 2))) {
                    score = (-1);
                } else {
                    score += 70000;
                }
            } else if (o.type == OptionType::ABILITY) {
                card = get_card(obs, o.area, o.index, my_index);
                if (no_draw) {
                    score = (-1);
                } else if (card.id == 1267) {
                    score = 1;
                } else {
                    score = 40000;
                }
            } else if (o.type == OptionType::RETREAT) {
                if (do_switch) {
                    score = 10000;
                } else {
                    score = (-1);
                }
            } else if (o.type == OptionType::ATTACK) {
                score = o.attackId;
            }
            scores.push_back(score);
        }
        output = {};
        if (count_of(scores) >= 1) {
            sorted_scores = ranked_scores(scores);
            for (i = 0; i < select.maxCount; ++i) {
                if ((at_index(sorted_scores, i).second >= 0) || (select.minCount > i) || ((context != SelectContext::TO_BENCH) && (context != SelectContext::SETUP_BENCH_POKEMON))) {
                    output.push_back(at_index(sorted_scores, i).first);
                }
            }
        }
        return output;
    }
};
inline std::unique_ptr<Agent> make_policy(const CardTableView& table,const std::vector<int>& deck) {
    std::map<int,int> actual;
    for(int id:deck)++actual[id];
    const std::map<int,int> expected{{2, 4}, {5, 4}, {119, 4}, {120, 4}, {121, 3}, {140, 1}, {184, 1}, {235, 2}, {1071, 1}, {1079, 2}, {1080, 1}, {1086, 4}, {1097, 2}, {1120, 4}, {1121, 4}, {1152, 3}, {1156, 1}, {1182, 3}, {1198, 4}, {1210, 2}, {1227, 4}, {1256, 2}};
    if(actual!=expected)throw std::runtime_error("Official deck mismatch: dragapult");
    return std::make_unique<Policy>(table,deck);
}
} // namespace official::dragapult
