#pragma once
#include "../../gbdt/cpp/features.h"
namespace dragapult_tokens {
using namespace official;
using Matrix=std::vector<std::vector<double>>;
inline int slot(const CardView& c) {
    auto it=std::find(dragapult_schema::IDS.begin(),dragapult_schema::IDS.end(),c.id);
    return !c.present || it==dragapult_schema::IDS.end() ? -1 : int(it-dragapult_schema::IDS.begin());
}
inline std::vector<float> counts(const std::vector<CardView>& cards) {
    std::vector<float> result(32);for(const auto& c:cards){int s=slot(c);if(s>=0)++result[s];}return result;
}
inline std::vector<int> indices(const Observation& obs,const std::vector<int>& options){
    std::vector<int> result;for(int i:options)result.push_back(i<0 ? 19+count_of(obs.select.option) : 19+i);return result;
}
// Prefix affects only the pointer-head masks; the Encoder input is unchanged.
inline void set_prefix(Matrix& rows,const Observation& obs,const std::vector<int>& prefix){
    if(rows.size()!=20+obs.select.option.size())throw std::runtime_error("Token/observation mismatch");
    const auto legal=dragapult_features::options_for(obs,prefix);
    for(int i=0;i<count_of(obs.select.option);++i){rows[19+i][2]=contains(legal,i);rows[19+i][3]=contains(prefix,i);}
    rows.back()[2]=contains(legal,-1);
}
inline Matrix tokenize(const Observation& obs,const std::vector<int>& prefix,const CardTableView& table){
    const auto& cur=obs.current;const auto& sel=obs.select;const int actor=cur.yourIndex,n=count_of(sel.option);
    if(n>100)throw std::runtime_error("Transformer option overflow: maximum 100; no truncation");
    auto legal=dragapult_features::options_for(obs,prefix);Matrix x(20+n,std::vector<double>(198));
    for(auto& r:x)r[4]=r[5]=-1;
    auto generic=[&](int r,const std::vector<double>& v){for(std::size_t j=0;j<v.size();++j){float a=v[j];x[r][6+j]=std::copysign(std::log1p(std::abs(a)),a);}};
    auto append=[](auto& dst,const auto& src,int start,int end){dst.insert(dst.end(),src.begin()+start,src.begin()+end);};
    auto base=dragapult_features::row(obs,-1,{},table,false);std::vector<double> g;
    append(g,base,0,20);append(g,base,90,138);append(g,base,(dragapult_schema::DOMAIN_OFFSET+0),(dragapult_schema::DOMAIN_OFFSET+8));
    x[0][0]=1;x[0][4]=slot(sel.effect);x[0][5]=slot(sel.contextCard);generic(0,g);
    const auto& me=cur.players[actor];const auto& op=cur.players[1-actor];auto known=concat(me.hand,me.discard);
    for(int side=0;side<2;++side){const auto& ps=side==0 ? me : op;
        std::vector<CardView> board{ps.active.empty() ? CardView{} : ps.active[0]};board=concat(board,ps.bench);
        if(board.size()>6)throw std::runtime_error("Board exceeds 6 slots per side");
        for(int j=0;j<6;++j){int r=1+6*side+j;x[r][1]=1;if(j>=count_of(board) || !board[j].present)continue;
            const auto& c=board[j];x[r][0]=1;x[r][4]=slot(c);std::vector<double> v{double(side),double(j==0)};
            dragapult_features::card_values(v,c,table);generic(r,v);
            auto attached=concat(c.energyCards,c.tools);auto a=counts(attached),e=counts(c.preEvolution);
            for(int k=0;k<32;++k){x[r][102+k]=a[k]/60.f;x[r][134+k]=e[k]/60.f;}
            if(side==0){known.push_back(c);known=concat(known,attached);known=concat(known,c.preEvolution);}
        }
    }
    for(const auto& c:cur.stadium)if(c.present && c.playerIndex==actor)known.push_back(c);
    auto used=counts(known);std::vector<float> remaining(32),deck(32),prize(32);
    for(std::size_t k=0;k<dragapult_schema::IDS.size();++k)remaining[k]=dragapult_schema::COUNTS[k];
    for(int k=0;k<32;++k)remaining[k]=std::max(0.f,remaining[k]-used[k]);
    auto exposed=counts(sel.deck);bool exact=sel.has_deck && count_of(sel.deck)==me.deckCount;
    for(const auto& c:sel.deck)exact=exact && c.present;
    for(int k=0;k<32;++k)exact=exact && exposed[k]<=remaining[k];
    int total=std::max(1,me.deckCount+count_of(me.prize));
    for(int k=0;k<32;++k){deck[k]=exact ? exposed[k] : remaining[k]*float(double(me.deckCount)/total);
        prize[k]=exact ? remaining[k]-deck[k] : remaining[k]*float(double(me.prize.size())/total);}
    std::vector<std::vector<float>> zones{counts(me.hand),counts(me.discard),deck,prize,exposed,counts(op.discard)};
    for(int z=0;z<6;++z){int r=13+z;x[r][0]=z!=4 || sel.has_deck;x[r][1]=2;
        std::vector<double> v(6);v[z]=1;float sum=0;for(float a:zones[z])sum+=a;
        v.push_back(sum);v.push_back(z==2 || z==3);v.push_back(exact && (z==2 || z==3));generic(r,v);
        for(int k=0;k<32;++k)x[r][102+k]=zones[z][k]/60.f;
    }
    for(int i=0;i<n;++i){int r=19+i;const auto& o=sel.option[i];x[r][0]=1;x[r][1]=3;x[r][2]=contains(legal,i);x[r][3]=contains(prefix,i);
        x[r][4]=slot(dragapult_features::candidate(obs,o));auto f=dragapult_features::row(obs,i,{},table,false);std::vector<double> v;
        append(v,f,22,60);append(v,f,138,176);append(v,f,(dragapult_schema::DOMAIN_OFFSET+0),(dragapult_schema::DOMAIN_OFFSET+12));generic(r,v);
        if(o.type==13){x[r][166]=1;x[r][169]=1;x[r][195]=float(f[(dragapult_schema::DOMAIN_OFFSET+8)]/600);x[r][196]=f[(dragapult_schema::DOMAIN_OFFSET+10)];x[r][197]=f[(dragapult_schema::DOMAIN_OFFSET+11)];}
    }
    x.back()[0]=1;x.back()[1]=4;x.back()[2]=contains(legal,-1);return x;
}
}
