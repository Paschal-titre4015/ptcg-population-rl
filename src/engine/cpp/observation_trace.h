#pragma once
#include "official_observation.h"

namespace arena {
// Serialize the actual policy input as well as the upstream API observation.
// The parity checker verifies this projection independently before actions.
inline void view_json(JsonBuilder& j, const official::CardView& card) {
    if (!card.present) { j.appendNull(); return; }
    j.append('{');
    j.appendKeyValue("id", card.id);
    j.appendCommaKeyValue("serial", card.serial);
    j.appendCommaKeyValue("playerIndex", card.playerIndex);
    if (card.is_pokemon) {
        j.appendCommaKeyValue("hp", card.hp);
        j.appendCommaKeyValue("maxHp", card.maxHp);
        j.appendCommaKeyValue("appearThisTurn", card.appearThisTurn);
        j.appendCommaKey("energies"); j.append('[');
        for (int i = 0; i < official::count_of(card.energies); ++i) { j.comma(i); j.append(card.energies[i]); }
        j.append(']');
        auto cards = [&](const auto& key, const auto& values) {
            j.appendCommaKey(key); j.append('[');
            for (int i = 0; i < official::count_of(values); ++i) { j.comma(i); view_json(j, values[i]); }
            j.append(']');
        };
        cards("energyCards", card.energyCards); cards("tools", card.tools); cards("preEvolution", card.preEvolution);
    }
    j.append('}');
}
inline void view_json(JsonBuilder& j, const official::Observation& obs) {
    auto cards = [&](const auto& key, const auto& values) {
        j.appendCommaKey(key); j.append('[');
        for (int i = 0; i < official::count_of(values); ++i) { j.comma(i); view_json(j, values[i]); }
        j.append(']');
    };
    j.append('{'); j.appendKey("current"); j.append('{');
    const auto& cur = obs.current;
    j.appendKeyValue("turn", cur.turn);
    j.appendCommaKeyValue("yourIndex", cur.yourIndex);
    j.appendCommaKeyValue("firstPlayer", cur.firstPlayer);
    j.appendCommaKeyValue("supporterPlayed", cur.supporterPlayed);
    j.appendCommaKeyValue("energyAttached", cur.energyAttached);
    cards("stadium", cur.stadium);
    if (cur.has_looking) cards("looking", cur.looking);
    else { j.appendCommaKey("looking"); j.appendNull(); }
    j.appendCommaKey("players"); j.append('[');
    for (int pi = 0; pi < 2; ++pi) {
        const auto& ps = cur.players[pi];
        j.comma(pi); j.append('{');
        j.appendKeyValue("handCount", ps.handCount);
        j.appendCommaKeyValue("deckCount", ps.deckCount);
        j.appendCommaKeyValue("asleep", ps.asleep);
        j.appendCommaKeyValue("paralyzed", ps.paralyzed);
        cards("active", ps.active); cards("bench", ps.bench); cards("discard", ps.discard); cards("prize", ps.prize);
        cards("hand", ps.hand);
        j.append('}');
    }
    j.append(']'); j.append('}');
    const auto& sel = obs.select;
    j.appendCommaKey("select"); j.append('{');
    j.appendKeyValue("context", sel.context);
    j.appendCommaKeyValue("minCount", sel.minCount);
    j.appendCommaKeyValue("maxCount", sel.maxCount);
    j.appendCommaKeyValue("remainDamageCounter", sel.remainDamageCounter);
    if (sel.has_deck) cards("deck", sel.deck);
    else { j.appendCommaKey("deck"); j.appendNull(); }
    j.appendCommaKey("contextCard"); view_json(j, sel.contextCard);
    j.appendCommaKey("effect"); view_json(j, sel.effect);
    j.appendCommaKey("option"); j.append('[');
    for (int i = 0; i < official::count_of(sel.option); ++i) {
        const auto& o = sel.option[i];
        j.comma(i); j.append('{'); j.appendKeyValue("type", o.type);
        j.appendCommaKeyValue("number", o.number); j.appendCommaKeyValue("area", o.area);
        j.appendCommaKeyValue("index", o.index); j.appendCommaKeyValue("playerIndex", o.playerIndex);
        j.appendCommaKeyValue("inPlayArea", o.inPlayArea); j.appendCommaKeyValue("inPlayIndex", o.inPlayIndex);
        j.appendCommaKeyValue("attackId", o.attackId); j.append('}');
    }
    j.append(']'); j.append('}');
    j.appendCommaKey("logs"); j.append('[');
    for (int i = 0; i < official::count_of(obs.logs); ++i) {
        const auto& log = obs.logs[i];
        j.comma(i); j.append('{'); j.appendKeyValue("type", log.type);
        j.appendCommaKeyValue("playerIndex", log.playerIndex);
        j.appendCommaKeyValue("attackId", log.attackId);
        j.appendCommaKeyValue("fromArea", log.fromArea);
        j.appendCommaKeyValue("toArea", log.toArea); j.append('}');
    }
    j.append(']'); j.append('}');
}
} // namespace arena
