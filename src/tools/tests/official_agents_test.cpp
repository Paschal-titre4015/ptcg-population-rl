// Contract checks build without All.h, cg-lib, Python embedding, or assets.
#include <iostream>
#include "../../agents/cpp/registry.h"

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void expect_action(official::Agent& agent, const official::Observation& obs, std::vector<int> expected) {
    check(agent.decide(obs) == expected, "Unexpected policy action");
}
}
int main() {
    using namespace official;
    try {
        CardTableView table;
        const std::vector<int> deck;
        mega_lucario::Policy lucario(table, deck);
        dragapult::Policy dragapult(table, deck);
        iono::Policy iono(table, deck);
        mega_abomasnow::Policy abomasnow(table, deck);
        Observation obs;
        obs.select.context = SelectContext::IS_FIRST;
        obs.select.minCount = obs.select.maxCount = 2;
        for (int value : {2, 2, 3}) {
            OptionView option;
            option.type = OptionType::NUMBER;
            option.number = value;
            obs.select.option.push_back(option);
        }
        for (Agent* agent : std::array<Agent*, 4>{&lucario, &dragapult, &iono, &abomasnow})
            expect_action(*agent, obs, {2, 0}); // Stable tie order, multi-selection.

        // Dragapult may stop optional bench selection, but must obey minCount.
        CardView dreepy;
        dreepy.present = true; dreepy.id = 119; dreepy.serial = 0; dreepy.playerIndex = 0;
        table[119].cardType = CardType::POKEMON;
        obs.current.players[0].hand = {dreepy};
        obs.current.players[0].handCount = 1;
        obs.select.context = SelectContext::SETUP_BENCH_POKEMON;
        obs.select.minCount = 0; obs.select.maxCount = 1;
        OptionView choose;
        choose.type = OptionType::CARD; choose.area = AreaType::HAND; choose.index = 0; choose.playerIndex = 0;
        obs.select.option = {choose};
        expect_action(dragapult, obs, {});
        obs.select.minCount = 1;
        expect_action(dragapult, obs, {0});
        obs.select.minCount = 0;
        obs.current.firstPlayer = 1;
        expect_action(dragapult, obs, {0});

        // Iono's cached can_attack flag belongs to one seat/game instance.
        iono::Policy attacking(table, deck), fresh(table, deck);
        Observation main;
        main.select.context = SelectContext::MAIN;
        main.select.minCount = main.select.maxCount = 1;
        OptionView attack; attack.type = OptionType::ATTACK; attack.attackId = 1;
        main.select.option = {attack};
        expect_action(attacking, main, {0});
        CardView voltorb; voltorb.present = true; voltorb.is_pokemon = true;
        voltorb.id = 265; voltorb.energies = {4, 4};
        main.current.players[0].active = {voltorb};
        CardView wattrel = voltorb; wattrel.id = 270; wattrel.energies.clear();
        main.current.players[0].bench = {wattrel};
        main.select.context = SelectContext::ATTACH_FROM;
        OptionView active; active.type = OptionType::ATTACH;
        active.inPlayArea = AreaType::ACTIVE; active.inPlayIndex = 0;
        OptionView bench = active; bench.inPlayArea = AreaType::BENCH;
        // Active Voltorb scores +3000 only when can_attack is false.
        // Benched Wattrel scores +6000, so use a Tadbulb (+10) as comparator.
        main.current.players[0].bench[0].id = 268;
        main.select.option = {active, bench};
        expect_action(attacking, main, {1});
        expect_action(fresh, main, {0});

        bool rejected = false;
        try { make_factory("dragapult", table, std::vector<int>(60, 1), ""); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "Wrong deck accepted");
        rejected = false;
        try { make_factory("missing", table, deck, ""); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "Unknown policy accepted");
        std::cout << "Stable ties, multi-selection, optional STOP, seat isolation, deck contracts: OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
