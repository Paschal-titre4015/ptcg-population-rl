#pragma once
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace iono_json {
// Locale-independent, round-trippable JSON numbers without iostream formatting
// per feature. Keep the replay's double precision and existing array schema.
template<class T> inline void number(std::string& out,T value){
    char buffer[64];std::to_chars_result result;
    if constexpr(std::is_floating_point_v<T>){
        if(!std::isfinite(value))throw std::runtime_error("Non-finite replay number");
        result=std::to_chars(buffer,buffer+sizeof(buffer),value,std::chars_format::general,17);
    }else result=std::to_chars(buffer,buffer+sizeof(buffer),value);
    if(result.ec!=std::errc{})throw std::runtime_error("Cannot format replay number");
    out.append(buffer,result.ptr);
}
template<class Values> inline void numbers(std::string& out,const Values& values){
    out+='[';bool first=true;
    for(auto value:values){if(!first)out+=',';first=false;number(out,value);}
    out+=']';
}
}
