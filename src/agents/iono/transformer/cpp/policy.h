#pragma once
#include <memory>
#include "features.h"
#include "model.h"
#include "../../cpp/json_numbers.h"
#include "../../ppo/cpp/sampling.h"

namespace official::iono {
class TransformerPolicy final : public Agent {
    const CardTableView& table;
    std::shared_ptr<const iono_transformer::Model> model;
    iono_ppo::Sampling sampling;
    std::mt19937 rng;
    struct Query {std::vector<int> prefix,options;std::vector<double> scores;int target;double value;};
    iono_tokens::Matrix last_tokens;
    std::vector<Query> last_queries;
    std::vector<int> last_action;
    bool recorded=false;
    mutable bool has_encoding=false;
    mutable iono_transformer::Model::Prepared prepared;
public:
    iono_ppo::Statistics last;
    TransformerPolicy(const CardTableView& table,const std::vector<int>& deck,std::shared_ptr<const iono_transformer::Model> model, iono_ppo::Sampling sampling={})
        :table(table),model(std::move(model)),sampling(sampling),rng(sampling.seed) {
        iono_features::check_deck(deck);
        if(!this->model)throw std::runtime_error("iono_transformer requires a model");
        if(sampling.enabled && !this->model->has_value())throw std::runtime_error("PPO sampling requires a Value head");
    }
    std::vector<int> decide(const Observation& obs) override {
        last={};last_queries.clear();recorded=false;
        last_tokens.clear();
        bool encoded=false;
        std::vector<int> action;
        while(count_of(action)<obs.select.maxCount){
            const auto options=iono_features::options_for(obs,action);
            if(options.empty())throw std::runtime_error("No legal Transformer candidate");
            std::vector<double> scores;
            std::size_t chosen=0;double value=0;
            // A unique legal action needs no Policy evaluation. If a replay is
            // requested, its scores/Value are computed by the record hook.
            // Stochastic rollouts still evaluate Value and consume the same RNG.
            if(options.size()>1 || sampling.enabled){
                if(last_tokens.empty()){
                    last_tokens=iono_tokens::tokenize(obs,{},table);
                    model->prepare(last_tokens,prepared);encoded=true;
                }
                iono_tokens::set_prefix(last_tokens,obs,action);
                auto output=model->evaluate(prepared,last_tokens);value=output.value;
                for(int index:iono_tokens::indices(obs,options))scores.push_back(output.scores[index]);
                last.value=value;
                const double maximum=*std::max_element(scores.begin(),scores.end());
                while(scores[chosen]<maximum-iono_schema::TRANSFORMER_TIE)++chosen;
            }
            if(sampling.enabled)chosen=iono_ppo::sample(scores,sampling.temperature,rng,last);
            else ++last.queries;
            int best=options[chosen];
            last_queries.push_back({action,options,std::move(scores),best,value});
            if(best==-1)break;
            action.push_back(best);
        }
        last_action=action;
        recorded=true;
        has_encoding=encoded;
        return action;
    }
    void write_training_record(std::ostream& out,const Observation& obs,const std::vector<int>& action) const override {
        if(!recorded || action!=last_action)throw std::runtime_error("Missing Transformer decision record");
        auto rows=last_tokens.empty() ? iono_tokens::tokenize(obs,{},table) : last_tokens;
        std::string buffer;buffer.reserve(rows.size()*198*8);
        buffer+=",\"transformer_queries\":[";
        for(std::size_t q=0;q<last_queries.size();++q){const auto& query=last_queries[q];
            if(q)buffer+=',';
            iono_tokens::set_prefix(rows,obs,query.prefix);
            buffer+="{\"prefix\":";iono_json::numbers(buffer,query.prefix);
            buffer+=",\"options\":";iono_json::numbers(buffer,query.options);
            buffer+=",\"target\":";iono_json::number(buffer,query.target);
            buffer+=",\"features\":[";
            for(std::size_t i=0;i<rows.size();++i){if(i)buffer+=',';iono_json::numbers(buffer,rows[i]);}
            auto scores=query.scores;double value=query.value;
            if(scores.empty()){
                if(!has_encoding){model->prepare(rows,prepared);has_encoding=true;}
                auto output=model->evaluate(prepared,rows);value=output.value;
                for(int index:iono_tokens::indices(obs,query.options))scores.push_back(output.scores[index]);
            }
            buffer+="],\"scores\":";iono_json::numbers(buffer,scores);
            buffer+=",\"value\":";iono_json::number(buffer,value);buffer+='}';
        }
        buffer+=']';
        if(sampling.enabled){
            buffer+=",\"ppo\":{\"old_logp\":";iono_json::number(buffer,last.logp);
            buffer+=",\"old_value\":";iono_json::number(buffer,last.value);
            buffer+=",\"old_entropy\":";iono_json::number(buffer,last.entropy);
            buffer+=",\"queries\":";iono_json::number(buffer,last.queries);buffer+='}';
        }
        out.write(buffer.data(),buffer.size());
    }
};
} // namespace official::iono
