#pragma once
#include "kernels.h"
namespace transformer {
// Immutable weights, direct tensor pointers and thread-local bounded scratch.
// The same encoder/head equations and V3 weights as BasicModel are used.
class FastModel {
    static constexpr int N=120,D=128;
    struct Dense {const float* weight;const float* bias;int input,output;};
    struct Norm {const float* weight;const float* bias;};
    struct Stem {Dense first,second;Norm norm;};
    struct Block {Norm first,second;Dense qkv,out,ff1,ff2;};
    std::array<Stem,4> stems;std::array<Block,6> blocks;
    Norm final_norm;Dense query,key,bias,value1,value2,delta;
    const float *embedding,*types,*stop,*steps;
    struct Workspace {
        alignas(64) float x[N*D],normalized[N*D],qkv[N*3*D],attention[N*D],temp[N*D],ff[N*256];
        alignas(64) float stem_input[N*144],stem_hidden[N*D],stem_output[N*D],results[N*32];
        alignas(64) float probabilities[N],q[D],v[D];
    };
    static Workspace& workspace(){static thread_local Workspace value;return value;}
    static void linear(const float* input,float* output,int n,const Dense& layer,bool gelu=false){
        if(const auto fn=numeric::blas().sgemm;fn && n>1)
            fn(101,111,112,n,layer.output,layer.input,1.f,input,layer.input,layer.weight,layer.input,0.f,output,layer.output);
#if defined(__x86_64__) || defined(__i386__)
        else if(numeric::use_avx2() && n>=8)
            kernels::batch_avx(input,output,n,layer.input,layer.weight,layer.output);
#endif
        else for(int i=0;i<n;++i)for(int j=0;j<layer.output;++j)
            output[i*layer.output+j]=numeric::dot(input+i*layer.input,layer.weight+j*layer.input,layer.input);
        for(int i=0;i<n;++i)kernels::post(output+i*layer.output,layer.bias,layer.output,gelu);
    }
    static void normalize(const float* x,float* y,int n,Norm norm){
        for(int i=0;i<n;++i)kernels::norm(x+i*D,y+i*D,norm.weight,norm.bias);
    }
public:
    struct State {
        int size=0,raw_size=0;
        std::array<int,N> positions{};
        alignas(64) float hidden[N*D],keys[N*D],biases[N];
        double value=0;
    };
    template<class Source> explicit FastModel(const Source& source){
        auto param=[&](const std::string& name){return source.parameter(name).data();};
        auto dense=[&](const std::string& name,int input,int output){return Dense{param(name+".weight"),param(name+".bias"),input,output};};
        auto norm=[&](const std::string& name){return Norm{param(name+".weight"),param(name+".bias")};};
        std::array<std::string,4> names{"global_stem","board_stem","zone_stem","option_stem"};
        std::array<int,4> widths{128,144,64,144};
        for(int i=0;i<4;++i)stems[i]={dense(names[i]+".linear1",widths[i],D),dense(names[i]+".linear2",D,D),norm(names[i]+".norm")};
        for(int i=0;i<6;++i){std::string p="blocks."+std::to_string(i)+".";blocks[i]={norm(p+"norm1"),norm(p+"norm2"),dense(p+"qkv",D,3*D),dense(p+"out",D,D),dense(p+"ff1",D,256),dense(p+"ff2",256,D)};}
        final_norm=norm("norm");query=dense("policy_query",D,D);key=dense("policy_key",D,D);bias=dense("policy_bias",D,1);
        value1=dense("value1",D,96);value2=dense("value2",96,51);delta=dense("result_delta",32,D);
        embedding=param("deck_embedding.weight");types=param("token_type.weight");stop=param("stop");steps=param("step_embedding.weight");
    }
    void encode(const std::vector<std::vector<double>>& raw,State& state) const {
        if(raw.size()<20 || raw.size()>N)throw std::runtime_error("Invalid token count");
        auto& w=workspace();state.size=0;state.raw_size=raw.size();
        std::array<int,N> kinds;
        for(int r=0;r<int(raw.size());++r){const auto& row=raw[r];
            if(row.size()!=198)throw std::runtime_error("Input width mismatch");
            for(double x:row)if(!std::isfinite(x))throw std::runtime_error("Non-finite token");
            if(!row[0])continue;
            if(row[1]<0 || row[1]>4 || row[1]!=std::floor(row[1]))throw std::runtime_error("Invalid token kind");
            for(int c:{4,5})if(row[c]<-1 || row[c]>=32 || row[c]!=std::floor(row[c]))throw std::runtime_error("Invalid deck slot");
            int i=state.size++;state.positions[i]=r;kinds[i]=int(row[1]);
        }
        const int n=state.size;if(!n || state.positions[0]!=0)throw std::runtime_error("Missing Global token");
        // Group stems by kind and execute each as a compact token batch.
        for(int kind=0;kind<4;++kind){std::array<int,N> indices;int count=0;const auto& stem=stems[kind];
            for(int i=0;i<n;++i)if(kinds[i]==kind){const auto& row=raw[state.positions[i]];
                indices[count]=i;float* input=w.stem_input+count*stem.first.input;int cursor=kind==2 ? 48 : 96;
                for(int k=0;k<cursor;++k)input[k]=row[6+k];
                auto identity=[&](int column){int slot=int(row[column]);for(int k=0;k<16;++k)input[cursor++]=slot<0 ? 0.f : embedding[slot*16+k];};
                auto counts=[&](int column){for(int k=0;k<16;++k){double sum=0;for(int j=0;j<32;++j)sum+=float(row[column+j])*embedding[j*16+k];input[cursor++]=sum;}};
                if(kind==0){identity(4);identity(5);}else if(kind==1){identity(4);counts(102);counts(134);}else if(kind==2)counts(102);
                else {identity(4);for(int k=0;k<32;++k){input[cursor++]=row[166+k];w.results[count*32+k]=row[166+k];}}
                ++count;
            }
            if(!count)continue;
            linear(w.stem_input,w.stem_hidden,count,stem.first,true);linear(w.stem_hidden,w.stem_output,count,stem.second);
            normalize(w.stem_output,w.stem_hidden,count,stem.norm);
            if(kind==3)linear(w.results,w.stem_output,count,delta);
            for(int j=0;j<count;++j)for(int k=0;k<D;++k){float v=w.stem_hidden[j*D+k];if(kind==3)v+=w.stem_output[j*D+k];w.x[indices[j]*D+k]=v+types[kind*D+k];}
        }
        for(int i=0;i<n;++i)if(kinds[i]==4)for(int k=0;k<D;++k)w.x[i*D+k]=stop[k]+types[4*D+k];
        for(const auto& block:blocks){
            normalize(w.x,w.normalized,n,block.first);linear(w.normalized,w.qkv,n,block.qkv);
            std::fill_n(w.attention,n*D,0.f);
#if defined(__x86_64__) || defined(__i386__)
            if(numeric::use_avx2())kernels::attention_avx(w.qkv,w.attention,w.probabilities,n);
            else
#endif
            for(int head=0;head<4;++head)for(int i=0;i<n;++i){
                for(int j=0;j<n;++j)w.probabilities[j]=numeric::dot(w.qkv+i*3*D+head*32,w.qkv+j*3*D+D+head*32,32)/std::sqrt(32.f);
                kernels::softmax(w.probabilities,n);
                for(int j=0;j<n;++j)numeric::axpy(w.probabilities[j],w.qkv+j*3*D+2*D+head*32,w.attention+i*D+head*32,32);
            }
            linear(w.attention,w.temp,n,block.out);for(int j=0;j<n*D;++j)w.x[j]+=w.temp[j];
            normalize(w.x,w.normalized,n,block.second);linear(w.normalized,w.ff,n,block.ff1,true);linear(w.ff,w.temp,n,block.ff2);
            for(int j=0;j<n*D;++j)w.x[j]+=w.temp[j];
        }
        normalize(w.x,state.hidden,n,final_norm);linear(state.hidden,state.keys,n,key);linear(state.hidden,state.biases,n,bias);
        linear(state.hidden,w.q,1,value1,true);linear(w.q,w.v,1,value2);
        float maximum=*std::max_element(w.v,w.v+51);double sum=0;state.value=0;
        for(int j=0;j<51;++j){double p=std::exp(double(w.v[j])-maximum);sum+=p;state.value+=p*double(float(-1.+2.*j/50.));}
        state.value/=sum;
        if(!std::isfinite(state.value))throw std::runtime_error("Non-finite Value output");
    }
    BasicModel<double>::Output decode(const State& state,const std::vector<std::vector<double>>& raw) const {
        if(int(raw.size())!=state.raw_size)throw std::runtime_error("Prepared observation mismatch");
        auto& w=workspace();int selected=0;std::fill_n(w.q,D,0.f);
        for(int i=0;i<state.size;++i)if(raw[state.positions[i]][3]){++selected;for(int k=0;k<D;++k)w.q[k]+=state.hidden[i*D+k];}
        for(int k=0;k<D;++k)w.q[k]=(state.hidden[k]+w.q[k])+steps[std::min(selected,5)*D+k];
        linear(w.q,w.v,1,query);
        BasicModel<double>::Output output;output.value=state.value;output.scores.resize(state.raw_size,-INFINITY);
        for(int i=0;i<state.size;++i)if(raw[state.positions[i]][2]){
            double score=0;for(int k=0;k<D;++k)score+=double(state.keys[i*D+k])*w.v[k];
            output.scores[state.positions[i]]=score/std::sqrt(double(D))+state.biases[i];
            if(!std::isfinite(output.scores[state.positions[i]]))throw std::runtime_error("Non-finite policy output");
        }
        return output;
    }
};
}
