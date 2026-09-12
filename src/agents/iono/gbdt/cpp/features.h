#pragma once
#include <cmath>
#include "../../../../engine/cpp/agent.h"
#include "iono_schema.h"

namespace iono_features {
using namespace official;
inline void check_deck(const std::vector<int>& deck) {
    std::map<int, int> counts;
    for (int id : deck) ++counts[id];
    std::map<int, int> expected;
    for (std::size_t i=0; i<iono_schema::IDS.size(); ++i) expected[iono_schema::IDS[i]]=iono_schema::COUNTS[i];
    if (counts != expected) throw std::runtime_error("Iono feature/model deck mismatch");
}
inline CardView card_at(const Observation& obs, int area, int index, int player) {
    if (area != 1 && area != 2 && area != 3 && area != 4 && area != 5 && area != 6 && area != 7 && area != 12) return {};
    if (area == 1 && !obs.select.has_deck) return {};
    if (area == 2 && player != obs.current.yourIndex) return {};
    if (area == 12 && !obs.current.has_looking) return {};
    return get_card(obs, area, index, player);
}
inline CardView candidate(const Observation& obs, const OptionView& o) {
    const int actor=obs.current.yourIndex;
    if (o.type==7) return card_at(obs, 2, o.index, actor);
    if (o.type==8 || o.type==9 || o.type==10 || o.type==11) return card_at(obs, o.area, o.index, actor);
    if (o.type>=3 && o.type<=6) return card_at(obs, o.area, o.index, o.playerIndex);
    if (o.type==13 && !obs.current.players[actor].active.empty()) return obs.current.players[actor].active[0];
    return {};
}
inline std::vector<int> options_for(const Observation& obs, const std::vector<int>& prefix) {
    const auto& sel=obs.select;
    const std::set<int> selected(prefix.begin(),prefix.end());
    if (selected.size()!=prefix.size() || count_of(prefix)>sel.maxCount) throw std::runtime_error("Invalid selection prefix");
    for (int i: prefix) if(i<0 || i>=count_of(sel.option)) throw std::runtime_error("Prefix outside legal options");
    if (count_of(prefix)==sel.maxCount) return {};
    std::vector<int> values;
    for (int i=0;i<count_of(sel.option);++i) if(!selected.contains(i)) values.push_back(i);
    if (count_of(prefix)>=sel.minCount) values.push_back(-1);
    return values;
}
inline void card_values(std::vector<double>& row, const CardView& card, const CardTableView& table) {
    if (!card.present) { row.insert(row.end(),15,0.); return; }
    const auto& master=table.at(card.id);
    const int prizes=master.megaEx ? 3 : master.ex ? 2 : 1;
    for (double value : std::array<double,15>{1., double(card.is_pokemon), double(card.hp), double(card.maxHp),
         double(card.maxHp-card.hp), double(card.energies.size()), double(card.tools.size()), double(master.stage1),
         double(master.stage2), double(master.ex), double(master.megaEx), double(master.weakness==iono_schema::ATTACK_TYPE),
         double(master.resistance==iono_schema::ATTACK_TYPE), double(card.appearThisTurn), double(prizes)}) row.push_back(value);
}
} // namespace iono_features
#include "rich_features.h"
namespace iono_features {
inline std::vector<double> row(const Observation& obs, int option_index, const std::vector<int>& prefix, const CardTableView& table, bool rich=true) {
    const auto& cur=obs.current; const auto& sel=obs.select;
    const int actor=cur.yourIndex; const auto& me=cur.players[actor]; const auto& op=cur.players[1-actor];
    const bool stop=option_index==-1;
    const OptionView o=stop ? OptionView{} : sel.option.at(option_index);
    const auto card=stop ? CardView{} : candidate(obs,o);
    CardView target;
    if(o.type==8 || o.type==9) target=card_at(obs,o.inPlayArea,o.inPlayIndex,actor);
    else if(o.type>=3 && o.type<=6) target=card;
    const auto active=me.active.empty() ? CardView{} : me.active[0];
    const auto op_active=op.active.empty() ? CardView{} : op.active[0];
    if(o.type==13) target=op_active;
    const int target_area=o.inPlayArea>=0 ? o.inPlayArea : o.area;
    const int owner=o.playerIndex>=0 ? o.playerIndex : actor;
    std::vector<double> out;
    auto add=[&](auto value) { out.push_back(static_cast<double>(value)); };
    add(cur.turn); add(cur.firstPlayer==actor); add(cur.supporterPlayed); add(cur.energyAttached);
    add(me.handCount); add(me.deckCount); add(me.prize.size()); add(op.handCount); add(op.deckCount); add(op.prize.size());
    add(me.bench.size()); add(op.bench.size()); add(me.asleep); add(me.paralyzed); add(sel.minCount); add(sel.maxCount);
    add(sel.option.size()); add(sel.remainDamageCounter); add(sel.has_deck); add(cur.has_looking); add(prefix.size());
    add(sel.maxCount-count_of(prefix)); add(stop); add(o.type==0 ? o.number : 0);
    add(card.present && owner==actor); add(o.area==4); add(o.area==5); add(target_area==4); add(target_area==5);
    add(target.present && (owner!=actor || o.type==13));
    for (const auto& value : {card,target,active,op_active}) card_values(out,value,table);
    for(int i=0;i<48;++i) add(sel.context==i);
    for(int i=0;i<18;++i) add(o.type==i);
    for(int i=1;i<=12;++i) add(o.area==i);
    for(int i=0;i<8;++i) add(o.attackId>=0 && o.attackId%8==i);
    std::vector<CardView> selected;
    for(int i:prefix) selected.push_back(candidate(obs,sel.option.at(i)));
    const std::vector<CardView> single_card{card}, single_target{target}, effect{sel.effect}, context_card{sel.contextCard};
    for(const auto* zone : std::array<const std::vector<CardView>*,10>{&me.hand,&me.discard,&me.active,&me.bench,&sel.deck,&single_card,&single_target,&selected,&effect,&context_card}) {
        std::map<int,int> counts;
        for(const auto& c:*zone) if(c.present) ++counts[c.id];
        for(int id:iono_schema::IDS) add(counts[id]);
    }
    std::map<int,int> counts,hand;int energy=0,one=0,two=0,discard=0;
    for(const auto& c:concat(me.active,me.bench))if(c.present){++counts[c.id];energy+=count_of(c.energies);}
    for(const auto& c:me.hand)if(c.present)++hand[c.id];
    for(const auto& c:me.bench)if(c.present){one+=c.energies.size()>=1;two+=c.energies.size()>=2;}
    const auto& basic=iono_schema::BASIC_ENERGY_IDS;
    for(const auto& c:me.discard)discard+=c.present && contains(basic,c.id);
    int pokemon=0,evolutions=0,attackers=0,hand_energy=0;
    for(int id:iono_schema::POKEMON_IDS)pokemon+=counts[id];
    for(const auto& pair:iono_schema::EVOLUTIONS)evolutions+=std::min(counts[pair[0]],hand[pair[1]]);
    for(int id:iono_schema::ATTACKERS)attackers+=counts[id];
    for(int id:basic)hand_energy+=hand[id];
    add(pokemon);add(evolutions);add(attackers);add(one);add(two);add(discard);add(energy);add(hand_energy);
    add(0);add(0);add(0);add(0); // Unknown attack outcomes, flagged uncertain by tokens.
    if(!rich)return out;
    rich_append(out,obs,o,prefix,table);
    if(out.size()!=iono_schema::NAMES.size()) throw std::runtime_error("Feature width mismatch");
    return out;
}
} // namespace iono_features
