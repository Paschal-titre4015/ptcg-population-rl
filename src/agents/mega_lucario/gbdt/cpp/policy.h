#pragma once
#include <memory>
#include <limits>
#include "features.h"
#include "model.h"

namespace official::mega_lucario {
class GbdtPolicy final : public Agent {
    const CardTableView& table;
    std::shared_ptr<const gbdt::Model> model;
public:
    GbdtPolicy(const CardTableView& table, const std::vector<int>& deck, std::shared_ptr<const gbdt::Model> model)
        : table(table), model(std::move(model)) {
        lucario_features::check_deck(deck);
        if(!this->model) throw std::runtime_error("mega_lucario_gbdt requires a model");
    }
    std::vector<int> decide(const Observation& obs) override {
        std::vector<int> action;
        while(count_of(action)<obs.select.maxCount) {
            const auto options=lucario_features::options_for(obs,action);
            if(options.empty()) throw std::runtime_error("No legal GBDT candidate");
            int best=options.front(); double best_score=-std::numeric_limits<double>::infinity();
            for(int option:options) {
                const double score=model->predict(lucario_features::row(obs,option,action,table));
                if(score>best_score) {best_score=score;best=option;}
            }
            if(best==-1) break;
            action.push_back(best);
        }
        return action;
    }
};
} // namespace official::mega_lucario
