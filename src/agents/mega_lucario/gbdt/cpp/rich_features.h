#pragma once
// Independent C++ implementation of rich_features.py; only public CardViews.
namespace lucario_features {
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
        for(int id:lucario_schema::IDS)v.push_back(c.present && c.id==id);
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
    constexpr std::array<int,6> ids{673,674,675,676,677,678},needed{1,3,2,1,1,1};
    std::array<std::map<int,int>,4> counts;
    int z=0;
    for(const auto* zone:std::array<const std::vector<CardView>*,4>{&board,&me.bench,&me.hand,&me.discard}) {
        for(const auto& c:*zone)if(c.present)++counts[z][c.id];
        for(int id:ids)out.push_back(counts[z][id]);++z;
    }
    const bool lunar_bench=std::any_of(me.bench.begin(),me.bench.end(),[](const auto& c){return c.present && c.id==675;});
    for(int extra:{0,1})for(int i=0;i<6;++i) {
        int n=0;for(const auto& c:board)n+=c.present && c.id==ids[i] && (c.id!=676 || lunar_bench) && std::count(c.energies.begin(),c.energies.end(),6)+extra>=needed[i];
        out.push_back(n);
    }
    for(int i=0;i<12;++i)out.push_back(std::count(active.energies.begin(),active.energies.end(),i));
    int multi=0,mega=0,damaged=0,stage2=0,op_hp=0,my_hp=0;
    for(const auto& c:enemy)if(c.present){const auto& m=table.at(c.id);multi+=m.ex||m.megaEx;mega+=m.megaEx;stage2+=m.stage2;op_hp+=c.hp;}
    for(const auto& c:board)if(c.present){damaged+=c.id==678 && c.hp<c.maxHp;my_hp+=c.hp;}
    bool gravity=false;for(const auto& c:cur.stadium)gravity|=c.present && c.id==1252;
    auto& b=counts[0];auto& hand=counts[2];auto& discard=counts[3];
    for(double v:std::array<double,16>{double(hand[6]),double(discard[6]),double(!cur.energyAttached),
        double(std::max(0,5-count_of(me.bench))),double(multi),double(mega),double(damaged),double(gravity),
        double(stage2),double(op_hp),double(my_hp),double(o.attackId==980),
        double(o.attackId==982 && !me.bench.empty()?std::min(3,discard[6]):0),
        double(b[675]>0 && b[676]>0 && hand[6]>0),double(std::min(b[677],hand[678])),double(std::min(b[673],hand[674]))})out.push_back(v);
    for(int id=976;id<984;++id)out.push_back(std::any_of(sel.option.begin(),sel.option.end(),[&](const auto& a){return a.type==13 && a.attackId==id;}));
    const auto c=candidate(obs,o),target=rich_target(obs,o);
    const bool attaching=o.type==8 && c.present && c.id==6 && target.present;
    const int target_energy=std::count(target.energies.begin(),target.energies.end(),6);
    for(int n:{1,2,3})out.push_back(attaching && target_energy==n-1);
    int brave=0,bench_fighting=0;
    for(const auto& card:board)brave+=card.present && card.id==678 && std::count(card.energies.begin(),card.energies.end(),6)>=2;
    for(const auto& card:me.bench)if(card.present)bench_fighting+=std::count(card.energies.begin(),card.energies.end(),6);
    const int effect=sel.effect.present?sel.effect.id:0,cid=c.present?c.id:0;
    for(double v:std::array<double,13>{double(brave),double(bench_fighting),double(effect==675 && cid==6),
        double(std::min(me.deckCount,std::max(0,(me.prize.size()==6?8:6)-std::max(0,me.handCount-1)))),
        double(std::min(5,me.deckCount)),double(cid==1192 && cur.turn==1),
        double(effect==1102 && (sel.has_deck || cur.has_looking)),double(cid==1252?stage2:0),
        double(o.type==9 && cid==678),double(o.type==9 && cid==674),double(o.attackId==978?70:0),
        double(o.attackId==978 && active.present && active.hp>0 && active.hp<=70),
        double(o.attackId==982?count_of(me.bench):0)})out.push_back(v);
}
} // namespace lucario_features
