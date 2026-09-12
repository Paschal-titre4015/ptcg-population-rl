#pragma once
#include <cmath>
#include "../../../../engine/cpp/agent.h"
#include "mega_lucario_schema.h"

namespace lucario_features {
using namespace official;
inline void check_deck(const std::vector<int>& deck) {
    std::map<int, int> counts;
    for (int id : deck) ++counts[id];
    std::map<int, int> expected;
    for (std::size_t i=0; i<lucario_schema::IDS.size(); ++i) expected[lucario_schema::IDS[i]]=lucario_schema::COUNTS[i];
    if (counts != expected) throw std::runtime_error("Mega-Lucario feature/model deck mismatch");
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
         double(master.stage2), double(master.ex), double(master.megaEx), double(master.weakness==6),
         double(master.resistance==6), double(card.appearThisTurn), double(prizes)}) row.push_back(value);
}
} // namespace lucario_features
#include "rich_features.h"
namespace lucario_features {
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
    for(int i=976;i<984;++i) add(o.attackId==i);
    std::vector<CardView> selected;
    for(int i:prefix) selected.push_back(candidate(obs,sel.option.at(i)));
    const std::vector<CardView> single_card{card}, single_target{target}, effect{sel.effect}, context_card{sel.contextCard};
    for(const auto* zone : std::array<const std::vector<CardView>*,10>{&me.hand,&me.discard,&me.active,&me.bench,&sel.deck,&single_card,&single_target,&selected,&effect,&context_card}) {
        std::map<int,int> counts;
        for(const auto& c:*zone) if(c.present) ++counts[c.id];
        for(int id:lucario_schema::IDS) add(counts[id]);
    }
    std::map<int,int> counts; int energy=0,ready=0,after_attach=0,fighting=0;
    for(const auto& c:concat(me.active,me.bench)) if(c.present) { ++counts[c.id]; energy+=count_of(c.energies); }
    for(const auto& c:me.bench) if(c.present) {
        int needed=99;
        if(c.id==673 || c.id==676 || c.id==677 || c.id==678) needed=1;
        if(c.id==674) needed=3;
        if(c.id==675) needed=2;
        bool condition=c.id!=676 || std::any_of(me.bench.begin(),me.bench.end(),[](const auto& b){return b.present && b.id==675;});
        const int fighting_count=std::count(c.energies.begin(),c.energies.end(),6);
        ready+=condition && fighting_count>=needed; after_attach+=condition && fighting_count+1>=needed;
    }
    for(const auto& c:me.discard) fighting+=c.present && c.id==6;
    int damage=0;
    switch(o.attackId) {
        case 976:damage=10;break; case 977:damage=30;break; case 978:damage=210;break; case 979:damage=50;break;
        case 980:damage=70;break; case 981:damage=30;break; case 982:damage=130;break; case 983:damage=270;break;
    }
    if(o.attackId==980 && !std::any_of(me.bench.begin(),me.bench.end(),[](const auto& c){return c.present && c.id==675;}))damage=0;
    if(damage && op_active.present && o.attackId!=980) {
        const auto& master=table.at(op_active.id);
        if(master.weakness==6) damage*=2; else if(master.resistance==6) damage-=30;
    }
    const bool ko=damage>0 && op_active.hp>0 && damage>=op_active.hp;
    int prizes=0;
    if(op_active.present) { const auto& m=table.at(op_active.id); prizes=m.megaEx ? 3 : m.ex ? 2 : 1; }
    add(counts[675]>0 && counts[676]>0); add(counts[677]+counts[678]); add(counts[673]+counts[674]);
    add(ready); add(after_attach); add(fighting); add(energy); add(std::min(3,fighting));
    add(damage); add(damage ? damage-op_active.hp : 0); add(ko); add(ko && count_of(me.prize)<=prizes);
    if(!rich)return out;
    rich_append(out,obs,o,prefix,table);
    if(out.size()!=lucario_schema::NAMES.size()) throw std::runtime_error("Feature width mismatch");
    return out;
}
} // namespace lucario_features
