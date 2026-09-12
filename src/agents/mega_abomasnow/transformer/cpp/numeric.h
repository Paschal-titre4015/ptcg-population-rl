#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <dlfcn.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#ifndef PTCG_OPENBLAS_DEFAULT
#define PTCG_OPENBLAS_DEFAULT ""
#endif

namespace mega_abomasnow_transformer::numeric {
inline std::string requested_backend() {
    const char* p=std::getenv("PTCG_TRANSFORMER_BACKEND");
    const std::string name=p ? p : "auto";
    if(name!="auto" && name!="reference" && name!="scalar" && name!="avx2" && name!="blas")
        throw std::runtime_error("Unknown PTCG_TRANSFORMER_BACKEND");
    return name;
}
inline bool has_avx2() {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}
inline bool use_avx2() {
    static const bool enabled=[] {auto name=requested_backend();
        if(name=="avx2" && !has_avx2())throw std::runtime_error("AVX2/FMA unavailable");
        return name!="reference" && name!="scalar" && has_avx2();}();
    return enabled;
}
#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2,fma"))) inline float avx_dot(const float* a,const float* b,int n) {
    __m256 sum=_mm256_setzero_ps();int i=0;
    for(;i+8<=n;i+=8)sum=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),sum);
    alignas(32) float lanes[8];_mm256_store_ps(lanes,sum);
    float result=0;for(float v:lanes)result+=v;for(;i<n;++i)result+=a[i]*b[i];return result;
}
__attribute__((target("avx2,fma"))) inline void avx_axpy(float c,const float* x,float* y,int n) {
    __m256 coefficient=_mm256_set1_ps(c);int i=0;
    for(;i+8<=n;i+=8)_mm256_storeu_ps(y+i,_mm256_fmadd_ps(coefficient,_mm256_loadu_ps(x+i),_mm256_loadu_ps(y+i)));
    for(;i<n;++i)y[i]+=c*x[i];
}
#endif
inline float dot(const float* a,const float* b,int n) {
#if defined(__x86_64__) || defined(__i386__)
    if(use_avx2())return avx_dot(a,b,n);
#endif
    float sum=0;for(int i=0;i<n;++i)sum+=a[i]*b[i];return sum;
}
inline void axpy(float c,const float* x,float* y,int n) {
#if defined(__x86_64__) || defined(__i386__)
    if(use_avx2()){avx_axpy(c,x,y,n);return;}
#endif
    for(int i=0;i<n;++i)y[i]+=c*x[i];
}
struct Blas {
    using Sgemm=void (*)(int,int,int,int,int,int,float,const float*,int,const float*,int,float,float*,int);
    void* handle=nullptr;Sgemm sgemm=nullptr;
    Blas() {
        auto backend=requested_backend();if(backend=="scalar" || backend=="reference" || backend=="avx2")return;
        const char* configured=std::getenv("PTCG_OPENBLAS_LIBRARY");
        const char* path=configured ? configured : PTCG_OPENBLAS_DEFAULT;
        handle=dlopen(*path ? path : "libopenblas.so.0",RTLD_NOW|RTLD_LOCAL);
        if(!handle){if(configured || backend=="blas")throw std::runtime_error("Cannot load requested OpenBLAS");return;}
        sgemm=reinterpret_cast<Sgemm>(dlsym(handle,"scipy_cblas_sgemm"));
        if(!sgemm)sgemm=reinterpret_cast<Sgemm>(dlsym(handle,"cblas_sgemm"));
        using SetThreads=void (*)(int);
        auto threads=reinterpret_cast<SetThreads>(dlsym(handle,"scipy_openblas_set_num_threads"));
        if(!threads)threads=reinterpret_cast<SetThreads>(dlsym(handle,"openblas_set_num_threads"));
        if(!sgemm || !threads)throw std::runtime_error("OpenBLAS SGEMM/thread symbols missing");
        threads(1);
    }
};
inline const Blas& blas(){static Blas instance;return instance;}
inline std::string backend() {
    auto name=requested_backend();if(name=="reference")return "reference-double";
    if(blas().sgemm)return use_avx2() ? "openblas-avx2-float32" : "openblas-float32";
    return use_avx2() ? "avx2-float32" : "scalar-float32";
}
}
