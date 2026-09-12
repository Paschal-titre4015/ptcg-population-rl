#pragma once
// Independent C++ implementation of rich_features.py; only public CardViews.
namespace mega_abomasnow_features {
inline void hashes(std::vector<double>& out,const std::vector<CardView>& cards,
                   std::initializer_list<int> moduli={67,71}) {
    for(int m:moduli) {
        std::vector<double> counts(m+1,0.);
        for(const auto& c:cards) if(c.present) ++counts[c.id>0 ? 1+c.id%m : 0];
        out.insert(out.end(),counts.begin(),counts.end());
    }
}
inline CardView rich_target(const Observation& obs,const OptionView& o) {
    int actor=obs.current.yourIndex;
    if(o.type==8 || o.type==9) return card_at(obs,o.inPlayArea,o.inPlayIndex,actor);
    if(o.type>=3 && o.type<=6) return candidate(obs,o);
    const auto& active=obs.current.players[1-actor].active;
    if(o.type==13 && !active.empty()) return active[0];
    return {};
}
inline void rich_append(std::vector<double>& out,const Observation& obs,const OptionView& o,
                        const std::vector<int>& prefix,const CardTableView& table) {
    const auto& cur=obs.current; const auto& sel=obs.select;
    const auto& me=cur.players[cur.yourIndex]; const auto& op=cur.players[1-cur.yourIndex];
    const auto board=concat(me.active,me.bench), enemy=concat(op.active,op.bench);
    const CardView active=me.active.empty()?CardView{}:me.active[0];
    auto fields=[&](const CardView& c) {
        std::vector<double> v;card_values(v,c,table);hashes(v,{c});
        for(int id:mega_abomasnow_schema::IDS)v.push_back(c.present && c.id==id);
        for(int i=0;i<12;++i)v.push_back(std::count(c.energies.begin(),c.energies.end(),i));
        hashes(v,c.tools,{17});hashes(v,c.preEvolution,{17});return v;
    };
    auto append=[&](const auto& v){out.insert(out.end(),v.begin(),v.end());};
    const auto width=fields({}).size();
    for(const auto& c:{candidate(obs,o),rich_target(obs,o),active,op.active.empty()?CardView{}:op.active[0]}) append(fields(c));
    for(const auto* ps:{&me,&op}) {
        if(ps->bench.size()>5)throw std::runtime_error("GBDT public board exceeds five bench slots");
        std::vector<std::vector<double>> canonical;
        for(const auto& c:ps->bench)if(c.present)canonical.push_back(fields(c));
        std::sort(canonical.begin(),canonical.end(),std::greater<>());
        for(const auto& v:canonical)append(v);
        out.insert(out.end(),width*(5-canonical.size()),0.);
    }
    std::vector<CardView> candidates,targets,selected;
    for(const auto& opt:sel.option){candidates.push_back(candidate(obs,opt));targets.push_back(rich_target(obs,opt));}
    for(int i:prefix)selected.push_back(candidate(obs,sel.option.at(i)));
    const auto& public_search=sel.has_deck?sel.deck:cur.looking;
    for(const auto* zone:std::array<const std::vector<CardView>*,10>{&me.hand,&me.discard,&board,&enemy,&op.discard,
             &cur.stadium,&public_search,&candidates,&targets,&selected}) hashes(out,*zone);
    for(const auto* zone:std::array<const std::vector<CardView>*,4>{&board,&me.bench,&me.hand,&me.discard}){
        std::map<int,int> counts;for(const auto& c:*zone)if(c.present)++counts[c.id];
        for(int id:mega_abomasnow_schema::IDS)out.push_back(counts[id]);
    }
    const auto target=rich_target(obs,o);
    for(const auto& card:{active,target})for(int i=0;i<12;++i)out.push_back(std::count(card.energies.begin(),card.energies.end(),i));
    for(int i=0;i<8;++i)out.push_back(std::any_of(sel.option.begin(),sel.option.end(),[&](const auto& a){return a.type==13 && a.attackId%8==i;}));
}
} // namespace mega_abomasnow_features
