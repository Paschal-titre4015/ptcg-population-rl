#pragma once
#include "policy.h"
#include "../gbdt/cpp/policy.h"
#include "../transformer/cpp/policy.h"
#include "training_trace.h"

namespace official::mega_abomasnow {
// All Mega Abomasnow model assets and training-record details stay with the agent.
class RecordedPolicy final : public Agent {
    std::unique_ptr<Agent> inner;
    const CardTableView& table;
    std::shared_ptr<const mega_abomasnow_gbdt::Model> trees;
    std::shared_ptr<const mega_abomasnow_transformer::Model> neural;
public:
    RecordedPolicy(std::unique_ptr<Agent> inner, const CardTableView& table,
                   std::shared_ptr<const mega_abomasnow_gbdt::Model> trees,
                   std::shared_ptr<const mega_abomasnow_transformer::Model> neural)
        : inner(std::move(inner)),table(table),trees(std::move(trees)),neural(std::move(neural)) {}
    std::vector<int> decide(const Observation& obs) override { return inner->decide(obs); }
    void write_training_record(std::ostream& out,const Observation& obs,const std::vector<int>& action) const override {
        if(neural){inner->write_training_record(out,obs,action);return;}
        out << ",\"gbdt_queries\":";
        mega_abomasnow_trace::gbdt_queries(out,obs,action,table,trees.get());
    }
};
inline AgentFactory factory(const std::string& name,const CardTableView& table,
                            const std::vector<int>& deck,const std::string& path) {
    mega_abomasnow_features::check_deck(deck);
    std::map<int,int> counts;
    for(int id:deck)++counts[id];
    std::shared_ptr<const mega_abomasnow_gbdt::Model> trees;
    std::shared_ptr<const mega_abomasnow_transformer::Model> neural;
    if(name=="mega_abomasnow_gbdt")
        trees=std::make_shared<mega_abomasnow_gbdt::Model>(path,mega_abomasnow_schema::ID,mega_abomasnow_schema::NAMES,counts);
    else if(name=="mega_abomasnow_transformer")
        neural=std::make_shared<mega_abomasnow_transformer::Model>(path,mega_abomasnow_schema::TRANSFORMER_ID,mega_abomasnow_schema::ID,198,counts);
    else if(name!="mega_abomasnow" || !path.empty())throw std::runtime_error("Invalid Mega Abomasnow policy/model");
    AgentFactory result;
    result.schema_id=mega_abomasnow_schema::ID;
    result.model_id=trees?trees->id:neural?neural->id:"";
    result.supports_sampling=neural && neural->has_value();
    result.create=[name,&table,deck,trees,neural](SamplingOptions sampling)->std::unique_ptr<Agent> {
        if(sampling.enabled && (!neural || !neural->has_value()))throw std::runtime_error("Sampling requires Transformer with Value");
        std::unique_ptr<Agent> inner;
        if(trees)inner=std::make_unique<GbdtPolicy>(table,deck,trees);
        else if(neural)inner=std::make_unique<TransformerPolicy>(table,deck,neural,mega_abomasnow_ppo::Sampling{sampling.enabled,sampling.seed,sampling.temperature});
        else inner=std::make_unique<Policy>(table,deck);
        return std::make_unique<RecordedPolicy>(std::move(inner),table,trees,neural);
    };
    return result;
}
}
