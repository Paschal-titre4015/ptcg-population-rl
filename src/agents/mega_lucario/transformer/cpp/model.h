#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include "numeric.h"

namespace transformer {
// Structured public-token Transformer; independent native inference.
template<class Real> class BasicModel {
    using Matrix=std::vector<std::vector<Real>>;
    struct Tensor { std::vector<unsigned> shape; std::vector<Real> data; };
    std::map<std::string,Tensor> tensors;
    static constexpr int D=128,H=4,L=6,F=256;
    int width=0;
    const Tensor& tensor(const std::string& name) const {return tensors.at(name);}
    Matrix linear(const Matrix& x,const std::string& name) const {
        const auto& w=tensor(name+".weight"); const auto& b=tensor(name+".bias");
        const int out=w.shape[0],in=w.shape[1];
        Matrix y(x.size(),std::vector<Real>(out));
        for(std::size_t r=0;r<x.size();++r) for(int i=0;i<out;++i) {
            double value=b.data[i];
            for(int j=0;j<in;++j) value+=x[r][j]*w.data[i*in+j];
            y[r][i]=value;
        }
        return y;
    }
    Matrix norm(const Matrix& x,const std::string& name) const {
        Matrix y=x; const auto& w=tensor(name+".weight").data;const auto& b=tensor(name+".bias").data;
        for(auto& row:y) {
            double mean=0;for(double v:row)mean+=v;mean/=D;
            double variance=0;for(double v:row)variance+=(v-mean)*(v-mean);variance/=D;
            const double scale=1/std::sqrt(variance+1e-5);
            for(int j=0;j<D;++j)row[j]=(row[j]-mean)*scale*w[j]+b[j];
        }
        return y;
    }
    static void add(Matrix& a,const Matrix& b) {
        for(std::size_t r=0;r<a.size();++r)for(std::size_t c=0;c<a[r].size();++c)a[r][c]+=b[r][c];
    }
public:
    const std::vector<Real>& parameter(const std::string& name) const {return tensor(name).data;}
    std::string id;
    struct Output {std::vector<double> scores;double value=0;};
    BasicModel(const std::string& path,const std::string& contract,const std::string& schema,int expected_width,const std::map<int,int>& deck) {
        std::ifstream in(path,std::ios::binary);
        auto require=[&](bool ok){if(!ok || !in)throw std::runtime_error("Invalid/incompatible Transformer model: "+path);};
        auto bytes=[&](std::size_t n){std::string s(n,'\0');in.read(s.data(),n);require(bool(in));return s;};
        auto u32=[&](){auto s=bytes(4);std::uint32_t v=0;for(int i=0;i<4;++i)v|=std::uint32_t(static_cast<unsigned char>(s[i]))<<(8*i);return v;};
        const auto magic=bytes(16);
        require(magic==std::string("PTCG_TFM_V3\0\0\0\0\0",16));
        require(bytes(64)==contract);require(bytes(64)==schema);id=bytes(64);
        require(id.find_first_not_of("0123456789abcdef")==std::string::npos);
        width=u32();require(width==expected_width);require(u32()==D);require(u32()==H);require(u32()==L);require(u32()==F);
        require(u32()==deck.size());for(const auto& [card,count]:deck){require(u32()==static_cast<unsigned>(card));require(u32()==static_cast<unsigned>(count));}
        const auto count=u32();require(count==114);
        for(unsigned i=0;i<count;++i) {
            auto length=u32(),ndim=u32();require(length>0 && length<=128 && ndim>0 && ndim<=2);
            auto name=bytes(length);Tensor t;std::size_t elements=1;
            for(unsigned j=0;j<ndim;++j){auto d=u32();require(d>0 && d<=4096);t.shape.push_back(d);elements*=d;}
            require(elements<=1000000);t.data.reserve(elements);
            for(std::size_t j=0;j<elements;++j){auto bits=u32();float v;std::memcpy(&v,&bits,4);require(std::isfinite(v));t.data.push_back(v);}
            require(tensors.emplace(name,std::move(t)).second);
        }
        require(in.peek()==std::char_traits<char>::eof());
        auto shape=[&](const std::string& name,std::vector<unsigned> expected){
            auto it=tensors.find(name);if(it==tensors.end() || it->second.shape!=expected)throw std::runtime_error("Transformer tensor shape: "+name);
        };
        auto dense=[&](const std::string& name,unsigned out,unsigned input){shape(name+".weight",{out,input});shape(name+".bias",{out});};
        auto ln=[&](const std::string& name){shape(name+".weight",{D});shape(name+".bias",{D});};
        shape("deck_embedding.weight",{32,16});shape("token_type.weight",{5,D});shape("stop",{D});shape("step_embedding.weight",{6,D});
        for(const auto& entry:std::vector<std::pair<std::string,unsigned>>{{"global_stem",128},{"board_stem",144},{"zone_stem",64},{"option_stem",144}}){
            dense(entry.first+".linear1",D,entry.second);dense(entry.first+".linear2",D,D);ln(entry.first+".norm");}
        for(int layer=0;layer<L;++layer){auto p="blocks."+std::to_string(layer)+".";
            ln(p+"norm1");dense(p+"qkv",3*D,D);dense(p+"out",D,D);ln(p+"norm2");dense(p+"ff1",F,D);dense(p+"ff2",D,F);}
        ln("norm");dense("policy_query",D,D);dense("policy_key",D,D);dense("policy_bias",1,D);
        dense("result_delta",D,32);dense("value1",96,D);dense("value2",51,96);
        std::size_t parameters=0;for(const auto& [name,t]:tensors)parameters+=t.data.size();require(parameters==980916);
    }
    static void gelu(Matrix& x){for(auto& row:x)for(auto& v:row)v=.5*v*(1+std::erf(v/std::sqrt(2.)));}
    Output evaluate(const Matrix& raw) const {
        if(raw.size()<20 || raw.size()>120)throw std::runtime_error("Invalid structured token count");
        const auto& emb=tensor("deck_embedding.weight").data;
        Matrix x;std::vector<int> positions;
        for(std::size_t i=0;i<raw.size();++i){const auto& r=raw[i];
            if(r.size()!=198)throw std::runtime_error("Transformer input width mismatch");
            for(double v:r)if(!std::isfinite(v))throw std::runtime_error("Non-finite token");
            if(!r[0])continue;
            if(r[1]<0 || r[1]>4 || r[1]!=std::floor(r[1]))throw std::runtime_error("Invalid token type");
            int kind=int(r[1]);
            positions.push_back(i);Matrix h;
            if(kind==4)h={tensor("stop").data};
            else {
                std::vector<Real> v(r.begin()+6,r.begin()+(kind==2 ? 54 : 102));
                auto identity=[&](int column){if(r[column]<-1 || r[column]>=32 || r[column]!=std::floor(r[column]))throw std::runtime_error("Invalid deck slot");int slot=int(r[column]);
                    for(int k=0;k<16;++k)v.push_back(slot<0 ? 0. : emb[slot*16+k]);};
                auto counts=[&](int column){for(int k=0;k<16;++k){double sum=0;for(int j=0;j<32;++j)sum+=r[column+j]*emb[j*16+k];v.push_back(sum);}};
                if(kind==0){identity(4);identity(5);}else if(kind==1){identity(4);counts(102);counts(134);}
                else if(kind==2)counts(102);else {identity(4);v.insert(v.end(),r.begin()+166,r.end());}
                const std::string name=std::array<std::string,4>{"global_stem","board_stem","zone_stem","option_stem"}[kind];
                h=linear({v},name+".linear1");gelu(h);h=norm(linear(h,name+".linear2"),name+".norm");
                if(kind==3)add(h,linear({std::vector<Real>(r.begin()+166,r.end())},"result_delta"));
            }
            for(int k=0;k<D;++k)h[0][k]+=tensor("token_type.weight").data[kind*D+k];x.push_back(h[0]);
        }
        if(positions.empty() || positions[0]!=0)throw std::runtime_error("Missing Global token");
        const auto n=x.size();
        for(int layer=0;layer<L;++layer){
            const auto p="blocks."+std::to_string(layer)+".";auto qkv=linear(norm(x,p+"norm1"),p+"qkv");
            Matrix attended(n,std::vector<Real>(D));
            for(int h=0;h<H;++h)for(std::size_t i=0;i<n;++i){
                std::vector<double> scores(n);double maximum=-INFINITY;
                for(std::size_t j=0;j<n;++j){double score=0;
                    for(int k=0;k<D/H;++k)score+=qkv[i][h*(D/H)+k]*qkv[j][D+h*(D/H)+k];
                    scores[j]=score/std::sqrt(double(D/H));maximum=std::max(maximum,scores[j]);}
                double sum=0;for(auto& score:scores){score=std::exp(score-maximum);sum+=score;}
                for(std::size_t j=0;j<n;++j) {
                    for(int k=0;k<D/H;++k)attended[i][h*(D/H)+k]+=(scores[j]/sum)*qkv[j][2*D+h*(D/H)+k];
                }
            }
            add(x,linear(attended,p+"out"));auto ff=linear(norm(x,p+"norm2"),p+"ff1");gelu(ff);add(x,linear(ff,p+"ff2"));
        }
        auto hidden=norm(x,"norm");Matrix query{hidden[0]};int selected=0;
        for(std::size_t i=0;i<n;++i)if(raw[positions[i]][3]){++selected;for(int k=0;k<D;++k)query[0][k]+=hidden[i][k];}
        for(int k=0;k<D;++k)query[0][k]+=tensor("step_embedding.weight").data[std::min(selected,5)*D+k];
        query=linear(query,"policy_query");auto keys=linear(hidden,"policy_key"),bias=linear(hidden,"policy_bias");
        Output result;result.scores.resize(raw.size(),-INFINITY);
        for(std::size_t i=0;i<n;++i)if(raw[positions[i]][2]){double score=0;for(int k=0;k<D;++k)score+=keys[i][k]*query[0][k];result.scores[positions[i]]=score/std::sqrt(double(D))+bias[i][0];if(!std::isfinite(result.scores[positions[i]]))throw std::runtime_error("Non-finite policy output");}
        auto value=linear({hidden[0]},"value1");gelu(value);value=linear(value,"value2");
        double maximum=*std::max_element(value[0].begin(),value[0].end()),sum=0;
        for(int j=0;j<51;++j){double p=std::exp(value[0][j]-maximum);sum+=p;result.value+=p*double(float(-1.+2.*j/50.));}
        result.value/=sum;if(!std::isfinite(result.value))throw std::runtime_error("Non-finite Value output");return result;
    }
};
} // namespace transformer
#include "fast_model.h"
namespace transformer {
// Public raw-feature interface stays double; model storage and computation use
// float32 by default. The independent double path is retained for parity audits.
class Model {
    using Matrix=std::vector<std::vector<double>>;
    std::unique_ptr<BasicModel<float>> fast;
    std::unique_ptr<BasicModel<double>> reference;
    std::unique_ptr<FastModel> optimized;
public:
    std::string id;
    using Output=BasicModel<double>::Output;
    Model(const std::string& path,const std::string& contract,const std::string& schema,int width,const std::map<int,int>& deck) {
        if(numeric::requested_backend()=="reference") {
            reference=std::make_unique<BasicModel<double>>(path,contract,schema,width,deck);id=reference->id;
        } else {
            (void)numeric::backend();fast=std::make_unique<BasicModel<float>>(path,contract,schema,width,deck);id=fast->id;
            optimized=std::make_unique<FastModel>(*fast);
        }
    }
    bool has_value() const {return true;}
    Output evaluate(const Matrix& raw) const {
        if(optimized){FastModel::State state;optimized->encode(raw,state);return optimized->decode(state,raw);}
        return reference->evaluate(raw);
    }
    struct Prepared {std::unique_ptr<FastModel::State> state;const Model* owner=nullptr;};
    // Callers may reuse the bounded state across decisions. Within a decision,
    // decode accepts only updates to the legal/selected pointer-head masks.
    void prepare(const Matrix& raw,Prepared& p) const {
        if(p.owner && p.owner!=this)throw std::runtime_error("Prepared model mismatch");
        p.owner=this;
        if(optimized){if(!p.state)p.state=std::make_unique<FastModel::State>();optimized->encode(raw,*p.state);}
    }
    Prepared prepare(const Matrix& raw) const {Prepared p;prepare(raw,p);return p;}
    Output evaluate(const Prepared& p,const Matrix& raw) const {
        if(p.owner!=this)throw std::runtime_error("Prepared model mismatch");
        return p.state ? optimized->decode(*p.state,raw) : evaluate(raw);
    }
};
} // namespace transformer
