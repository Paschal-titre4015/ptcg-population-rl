#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace dragapult_ppo {
struct Sampling { bool enabled=false;std::uint32_t seed=0;double temperature=1.; };
struct Statistics { double logp=0,value=0,entropy=0;int queries=0; };
inline int sample(const std::vector<double>& scores,double temperature,std::mt19937& rng,Statistics& statistics) {
    if(scores.empty() || !std::isfinite(temperature) || temperature<=0)throw std::runtime_error("Invalid sampling input");
    const double maximum=*std::max_element(scores.begin(),scores.end());
    std::vector<double> weights;double sum=0;
    for(double score:scores){double w=std::exp((score-maximum)/temperature);weights.push_back(w);sum+=w;}
    double draw=scores.size()==1 ? 0 : std::generate_canonical<double,53>(rng);
    int chosen=std::max_element(scores.begin(),scores.end())-scores.begin();double cumulative=0;
    for(std::size_t i=0;i<scores.size();++i){cumulative+=weights[i]/sum;if(draw<cumulative){chosen=i;break;}}
    const double log_sum=std::log(sum);
    statistics.logp+=(scores[chosen]-maximum)/temperature-log_sum;
    for(std::size_t i=0;i<scores.size();++i)if(weights[i]>0)statistics.entropy-=(weights[i]/sum)*((scores[i]-maximum)/temperature-log_sum);
    ++statistics.queries;
    return chosen;
}
} // namespace dragapult_ppo
