#pragma once
#include "policy.h"
#include "../gbdt/cpp/policy.h"
#include "../transformer/cpp/policy.h"
#include "training_trace.h"

namespace official::iono {
// All Iono model assets and training-record details stay with the agent.
class RecordedPolicy final : public Agent {
    std::unique_ptr<Agent> inner;
    const CardTableView& table;
    std::shared_ptr<const iono_gbdt::Model> trees;
    std::shared_ptr<const iono_transformer::Model> neural;
public:
    RecordedPolicy(std::unique_ptr<Agent> inner, const CardTableView& table,
                   std::shared_ptr<const iono_gbdt::Model> trees,
                   std::shared_ptr<const iono_transformer::Model> neural)
        : inner(std::move(inner)),table(table),trees(std::move(trees)),neural(std::move(neural)) {}
    std::vector<int> decide(const Observation& obs) override { return inner->decide(obs); }
    void write_training_record(std::ostream& out,const Observation& obs,const std::vector<int>& action) const override {
        if(neural){inner->write_training_record(out,obs,action);return;}
        out << ",\"gbdt_queries\":";
        iono_trace::gbdt_queries(out,obs,action,table,trees.get());
    }
};
inline AgentFactory factory(const std::string& name,const CardTableView& table,
                            const std::vector<int>& deck,const std::string& path) {
    iono_features::check_deck(deck);
    std::map<int,int> counts;
    for(int id:deck)++counts[id];
    std::shared_ptr<const iono_gbdt::Model> trees;
    std::shared_ptr<const iono_transformer::Model> neural;
    if(name=="iono_gbdt")
        trees=std::make_shared<iono_gbdt::Model>(path,iono_schema::ID,iono_schema::NAMES,counts);
    else if(name=="iono_transformer")
        neural=std::make_shared<iono_transformer::Model>(path,iono_schema::TRANSFORMER_ID,iono_schema::ID,198,counts);
    else if(name!="iono" || !path.empty())throw std::runtime_error("Invalid Iono policy/model");
    AgentFactory result;
    result.schema_id=iono_schema::ID;
    result.model_id=trees?trees->id:neural?neural->id:"";
    result.supports_sampling=neural && neural->has_value();
    result.create=[name,&table,deck,trees,neural](SamplingOptions sampling)->std::unique_ptr<Agent> {
        if(sampling.enabled && (!neural || !neural->has_value()))throw std::runtime_error("Sampling requires Transformer with Value");
        std::unique_ptr<Agent> inner;
        if(trees)inner=std::make_unique<GbdtPolicy>(table,deck,trees);
        else if(neural)inner=std::make_unique<TransformerPolicy>(table,deck,neural,iono_ppo::Sampling{sampling.enabled,sampling.seed,sampling.temperature});
        else inner=std::make_unique<Policy>(table,deck);
        return std::make_unique<RecordedPolicy>(std::move(inner),table,trees,neural);
    };
    return result;
}
}
