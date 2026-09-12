#pragma once
#include "numeric.h"
#include <cmath>
#include <limits>
namespace transformer::kernels {
#if defined(__x86_64__) || defined(__i386__)
struct VectorMath {
    void* erf=nullptr;void* exp=nullptr;
    VectorMath(){void* library=dlopen("libmvec.so.1",RTLD_NOW|RTLD_LOCAL);
        if(library){erf=dlsym(library,"_ZGVdN8v_erff");exp=dlsym(library,"_ZGVdN8v_expf");}}
};
inline const VectorMath& vector_math(){static VectorMath m;return m;}
// OpenBLAS-free fallback: eight tokens per SIMD lane group, with a bounded
// transpose buffer. Weight rows are reused across the active token batch.
__attribute__((target("avx2,fma"))) inline void batch_avx(
    const float* input,float* output,int n,int width,const float* weight,int outputs){
    alignas(32) static thread_local float transposed[256*120];
    const int padded=(n+7)&~7;
    for(int k=0;k<width;++k){
        for(int i=0;i<n;++i)transposed[k*padded+i]=input[i*width+k];
        for(int i=n;i<padded;++i)transposed[k*padded+i]=0;
    }
    for(int first=0;first<padded;first+=32){
        const int groups=std::min(4,(padded-first)/8);
        for(int j=0;j<outputs;++j){
            __m256 sums[4];for(int g=0;g<groups;++g)sums[g]=_mm256_setzero_ps();
            for(int k=0;k<width;++k){auto coefficient=_mm256_set1_ps(weight[j*width+k]);
                for(int g=0;g<groups;++g)sums[g]=_mm256_fmadd_ps(coefficient,_mm256_load_ps(transposed+k*padded+first+g*8),sums[g]);
            }
            alignas(32) float lanes[32];for(int g=0;g<groups;++g)_mm256_store_ps(lanes+g*8,sums[g]);
            for(int i=first;i<std::min(n,first+32);++i)output[i*outputs+j]=lanes[i-first];
        }
    }
}
__attribute__((target("avx2,fma"))) inline void post_avx(float* x,const float* bias,int width,bool gelu){
    using VecFunction=__m256(*)(__m256);
    auto erf=reinterpret_cast<VecFunction>(vector_math().erf);
    const __m256 half=_mm256_set1_ps(.5f),one=_mm256_set1_ps(1.f),scale=_mm256_set1_ps(.7071067811865475f);
    int j=0;
    for(;j+8<=width;j+=8){__m256 v=_mm256_add_ps(_mm256_loadu_ps(x+j),_mm256_loadu_ps(bias+j));
        if(gelu && erf)v=_mm256_mul_ps(_mm256_mul_ps(half,v),_mm256_add_ps(one,erf(_mm256_mul_ps(v,scale))));
        _mm256_storeu_ps(x+j,v);
        if(gelu && !erf)for(int k=0;k<8;++k)x[j+k]=.5f*x[j+k]*(1.f+std::erf(x[j+k]*.7071067811865475f));
    }
    for(;j<width;++j){x[j]+=bias[j];if(gelu)x[j]=.5f*x[j]*(1.f+std::erf(x[j]*.7071067811865475f));}
}
__attribute__((target("avx2,fma"))) inline void norm_avx(const float* x,float* out,const float* weight,const float* bias){
    __m256 sum=_mm256_setzero_ps();alignas(32) float lanes[8];
    for(int j=0;j<128;j+=8)sum=_mm256_add_ps(sum,_mm256_loadu_ps(x+j));
    _mm256_store_ps(lanes,sum);double mean=0;for(float v:lanes)mean+=v;mean/=128;
    __m256 m=_mm256_set1_ps(mean),squared=_mm256_setzero_ps();
    for(int j=0;j<128;j+=8){auto d=_mm256_sub_ps(_mm256_loadu_ps(x+j),m);squared=_mm256_fmadd_ps(d,d,squared);}
    _mm256_store_ps(lanes,squared);double variance=0;for(float v:lanes)variance+=v;
    auto scale=_mm256_set1_ps(1/std::sqrt(variance/128+1e-5));
    for(int j=0;j<128;j+=8)_mm256_storeu_ps(out+j,_mm256_fmadd_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x+j),m),scale),_mm256_loadu_ps(weight+j),_mm256_loadu_ps(bias+j)));
}
__attribute__((target("avx2,fma"))) inline void softmax_avx(float* x,int n){
    using VecFunction=__m256(*)(__m256);auto exp=reinterpret_cast<VecFunction>(vector_math().exp);
    float maximum=*std::max_element(x,x+n);int padded=(n+7)&~7;
    for(int i=n;i<padded;++i)x[i]=-std::numeric_limits<float>::infinity();
    __m256 shift=_mm256_set1_ps(maximum);
    if(exp)for(int i=0;i<padded;i+=8)_mm256_storeu_ps(x+i,exp(_mm256_sub_ps(_mm256_loadu_ps(x+i),shift)));
    else for(int i=0;i<n;++i)x[i]=std::exp(x[i]-maximum);
    double sum=0;for(int i=0;i<n;++i)sum+=x[i];
    auto inverse=_mm256_set1_ps(1/sum);
    for(int i=0;i<padded;i+=8)_mm256_storeu_ps(x+i,_mm256_mul_ps(_mm256_loadu_ps(x+i),inverse));
}
__attribute__((target("avx2,fma"))) inline void attention_avx(const float* qkv,float* output,float* probabilities,int n){
    constexpr int D=128;
    for(int head=0;head<4;++head)for(int i=0;i<n;++i){
        const float* q=qkv+i*3*D+head*32;
        __m256 q0=_mm256_loadu_ps(q),q1=_mm256_loadu_ps(q+8),q2=_mm256_loadu_ps(q+16),q3=_mm256_loadu_ps(q+24);
        for(int j=0;j<n;++j){const float* k=qkv+j*3*D+D+head*32;
            auto a=_mm256_fmadd_ps(q1,_mm256_loadu_ps(k+8),_mm256_mul_ps(q0,_mm256_loadu_ps(k)));
            auto b=_mm256_fmadd_ps(q3,_mm256_loadu_ps(k+24),_mm256_mul_ps(q2,_mm256_loadu_ps(k+16)));
            auto sum=_mm256_add_ps(a,b);
            auto halves=_mm_add_ps(_mm256_castps256_ps128(sum),_mm256_extractf128_ps(sum,1));
            halves=_mm_hadd_ps(halves,halves);halves=_mm_hadd_ps(halves,halves);
            probabilities[j]=_mm_cvtss_f32(halves)*.1767766952966369f;
        }
        softmax_avx(probabilities,n);
        __m256 a=_mm256_setzero_ps(),b=a,c=a,d=a;
        for(int j=0;j<n;++j){const float* v=qkv+j*3*D+2*D+head*32;auto p=_mm256_set1_ps(probabilities[j]);
            a=_mm256_fmadd_ps(p,_mm256_loadu_ps(v),a);b=_mm256_fmadd_ps(p,_mm256_loadu_ps(v+8),b);
            c=_mm256_fmadd_ps(p,_mm256_loadu_ps(v+16),c);d=_mm256_fmadd_ps(p,_mm256_loadu_ps(v+24),d);
        }
        float* out=output+i*D+head*32;_mm256_storeu_ps(out,a);_mm256_storeu_ps(out+8,b);_mm256_storeu_ps(out+16,c);_mm256_storeu_ps(out+24,d);
    }
}
#endif
inline void post(float* x,const float* bias,int width,bool gelu){
#if defined(__x86_64__) || defined(__i386__)
    if(numeric::use_avx2()){post_avx(x,bias,width,gelu);return;}
#endif
    for(int j=0;j<width;++j){x[j]+=bias[j];if(gelu)x[j]=.5f*x[j]*(1.f+std::erf(x[j]*.7071067811865475f));}
}
inline void norm(const float* x,float* out,const float* weight,const float* bias){
#if defined(__x86_64__) || defined(__i386__)
    if(numeric::use_avx2()){norm_avx(x,out,weight,bias);return;}
#endif
    double mean=0,variance=0;for(int j=0;j<128;++j)mean+=x[j];mean/=128;
    for(int j=0;j<128;++j)variance+=(x[j]-mean)*(x[j]-mean);
    double inverse=1/std::sqrt(variance/128+1e-5);
    for(int j=0;j<128;++j)out[j]=(x[j]-mean)*inverse*weight[j]+bias[j];
}
// x has capacity rounded up to 8, as provided by the fixed workspace.
inline void softmax(float* x,int n){
#if defined(__x86_64__) || defined(__i386__)
    if(numeric::use_avx2()){softmax_avx(x,n);return;}
#endif
    float maximum=*std::max_element(x,x+n);double sum=0;
    for(int i=0;i<n;++i){x[i]=std::exp(x[i]-maximum);sum+=x[i];}
    for(int i=0;i<n;++i)x[i]/=sum;
}
}
