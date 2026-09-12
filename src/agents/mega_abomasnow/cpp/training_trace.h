#pragma once
#include <iomanip>
#include "../gbdt/cpp/features.h"
#include "../gbdt/cpp/model.h"

namespace mega_abomasnow_trace {
inline void gbdt_queries(std::ostream& out, const official::Observation& obs, const std::vector<int>& action,
                         const official::CardTableView& table, const mega_abomasnow_gbdt::Model* model) {
    out << '[' << std::setprecision(17);
    std::vector<int> prefix;
    auto targets=action;
    if(official::count_of(action)<obs.select.maxCount) targets.push_back(-1);
    for(std::size_t q=0;q<targets.size();++q) {
        if(q) out<<',';
        const auto options=mega_abomasnow_features::options_for(obs,prefix);
        auto numbers=[&](const auto& values) { out<<'['; for(std::size_t i=0;i<values.size();++i) { if(i) out<<',';out<<values[i];}out<<']'; };
        out<<"{\"prefix\":"; numbers(prefix); out<<",\"options\":"; numbers(options);
        out<<",\"target\":"<<targets[q]<<",\"features\":[";
        std::vector<double> scores;
        for(std::size_t i=0;i<options.size();++i) {
            if(i)out<<',';const auto features=mega_abomasnow_features::row(obs,options[i],prefix,table);numbers(features);
            if(model)scores.push_back(model->predict(features));
        }
        out<<"],\"scores\":";
        if(model) numbers(scores); else out<<"null";
        out<<'}';
        if(targets[q]!=-1) prefix.push_back(targets[q]);
    }
    out<<']';
}
} // namespace mega_abomasnow_trace
