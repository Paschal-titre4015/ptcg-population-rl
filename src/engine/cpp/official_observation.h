#pragma once
#include "All.h"
#include "agent.h"

// Projection equivalent to the fields used from ToJsonApi(state, ..., start).
// Keep this adapter outside the policies: State contains private information.
namespace arena {
inline official::CardView card_view(const State& state, CardRef ref, bool reverse = false) {
    if (ref.isNull()) return {};
    const auto& card = state.getCard(ref);
    if (reverse && card.reverse) return {};
    official::CardView out;
    out.present = true;
    out.id = card.getMaster().cardId;
    out.serial = ref.cardIndex;
    out.playerIndex = card.playerIndex;
    return out;
}
template<class List> std::vector<official::CardView> card_list(const State& state, const List& refs, bool reverse = false) {
    std::vector<official::CardView> out;
    for (auto ref : refs) out.push_back(card_view(state, ref, reverse));
    return out;
}
inline official::CardView pokemon_view(const State& state, CardRef ref) {
    auto out = card_view(state, ref, true);
    if (!out.present) return out;
    const auto& card = state.getCard(ref);
    out.is_pokemon = true;
    out.hp = state.getHp(card);
    out.maxHp = state.getMaxHp(card);
    out.appearThisTurn = card.appear;
    std::vector<EnergyType> energies;
    state.getEnergies(card.playerIndex, ref, energies);
    for (auto energy : energies) out.energies.push_back(EnergyTypeIndex(energy));
    std::vector<CardRef> refs;
    state.getEnergyCards(ref, refs);
    out.energyCards = card_list(state, refs);
    out.tools = card_list(state, state.getAttachedToolRef(card));
    out.preEvolution = card_list(state, state.getPreEvolutions(card));
    return out;
}
inline official::CardTableView master_view() {
    official::CardTableView table;
    for (const auto& [id, card] : CardTable) {
        auto& out = table[id];
        out.cardType = static_cast<int>(card.cardType);
        out.weakness = EnergyWeakness(card.weakness);
        out.resistance = EnergyWeakness(card.resistance);
        out.megaEx = card.pokemonType == PokemonType::MegaEx;
        out.ex = card.pokemonType == PokemonType::Ex;
        out.stage1 = card.evolutionType == EvolutionType::Stage1;
        out.stage2 = card.evolutionType == EvolutionType::Stage2;
        out.name = reinterpret_cast<const char*>(card.nameEn.c_str());
    }
    return table;
}
inline official::Observation observe(const State& state, int start_log) {
    official::Observation obs;
    auto& cur = obs.current;
    const int actor = state.selectPlayer;
    if (actor < 0 || actor > 1) throw std::runtime_error("Invalid acting player");
    cur.turn = state.turn;
    cur.yourIndex = actor;
    cur.firstPlayer = state.firstPlayer;
    cur.supporterPlayed = state.supporterPlayed;
    cur.energyAttached = state.energyPlayed;
    cur.stadium = card_list(state, state.stadium);
    if (!state.looking.empty()) {
        if (state.lookingPlayer == actor || state.lookingPlayer == 2) {
            cur.has_looking = true;
            cur.looking = card_list(state, state.looking);
        } else if (state.lookingPlayer == actor + 3) {
            cur.has_looking = true;
            cur.looking.resize(state.looking.size());
        }
    }
    for (int pi = 0; pi < 2; ++pi) {
        const auto& ps = state.players[pi];
        auto& out = cur.players[pi];
        for (auto ref : ps.active) out.active.push_back(pokemon_view(state, ref));
        for (auto ref : ps.bench) out.bench.push_back(pokemon_view(state, ref));
        out.discard = card_list(state, ps.trash);
        out.prize = card_list(state, ps.prize, true);
        out.handCount = ps.hand.size();
        out.deckCount = ps.deck.size();
        if (pi == actor) out.hand = card_list(state, ps.hand);
        out.asleep = ps.badStatus == BadStatusType::Asleep;
        out.paralyzed = ps.badStatus == BadStatusType::Paralyzed;
    }
    auto& sel = obs.select;
    sel.context = std::max(0, static_cast<int>(state.selectContext) - 1);
    sel.minCount = state.selectMin;
    sel.maxCount = state.selectMax;
    sel.remainDamageCounter = state.remainDamageCounter;
    sel.has_deck = state.selectDeck;
    if (sel.has_deck) sel.deck = card_list(state, state.players[actor].deck);
    sel.contextCard = card_view(state, state.contextCard);
    if (state.onEffect()) sel.effect = card_view(state, state.getEffectCard().card);
    for (const auto& option : state.options) {
        official::OptionView out;
        out.type = static_cast<int>(option.type);
        switch (option.type) {
        case SelectOptionType::Number: out.number = option.param0; break;
        case SelectOptionType::Card:
        case SelectOptionType::ToolCard:
        case SelectOptionType::EnergyCard:
        case SelectOptionType::Energy:
            out.area = option.param0; out.index = option.param1; out.playerIndex = option.param2; break;
        case SelectOptionType::Play: out.index = option.param0; break;
        case SelectOptionType::Attach:
        case SelectOptionType::Evolve:
            out.area = option.param0; out.index = option.param1;
            out.inPlayArea = option.param2; out.inPlayIndex = option.param3; break;
        case SelectOptionType::Ability:
        case SelectOptionType::Discard:
            out.area = option.param0; out.index = option.param1; break;
        case SelectOptionType::Attack: out.attackId = option.param0; break;
        default: break;
        }
        sel.option.push_back(out);
    }
    for (int i = start_log; i < static_cast<int>(state.logs.size()); ++i) {
        const auto& log = state.logs[i];
        if (log.logType > LogType::Result) continue;
        official::LogView out;
        out.type = static_cast<int>(log.logType);
        // Only these fields are consumed by the policies. Do not expose hidden
        // card IDs through logs; all other log types retain only their type.
        if (log.logType == LogType::TurnEnd || log.logType == LogType::Attack ||
            log.logType == LogType::MoveCard || log.logType == LogType::MoveCardReverse) out.playerIndex = log.param[0];
        if (log.logType == LogType::Attack) out.attackId = log.param[3];
        if (log.logType == LogType::MoveCard) {
            bool visible = log.param[5] == 0 || (log.param[5] == 1 && log.param[0] == actor) ||
                           (log.param[5] == 3 && actor == 0) || (log.param[5] == 4 && actor == 1);
            if (!visible) out.type = static_cast<int>(LogType::MoveCardReverse);
            out.fromArea = log.param[3]; out.toArea = log.param[4];
        } else if (log.logType == LogType::MoveCardReverse) {
            out.fromArea = log.param[1]; out.toArea = log.param[2];
        } else if (log.logType == LogType::Draw && log.param[0] != actor) {
            out.type = static_cast<int>(LogType::DrawReverse);
        }
        obs.logs.push_back(out);
    }
    return obs;
}
} // namespace arena
