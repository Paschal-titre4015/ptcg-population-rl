#pragma once
#include <memory>
#include "../mega_lucario/cpp/factory.h"
#include "../dragapult/cpp/factory.h"
#include "../iono/cpp/factory.h"
#include "../mega_abomasnow/cpp/factory.h"
namespace official {
inline AgentFactory make_factory(const std::string& name,const CardTableView& table,const std::vector<int>& deck,const std::string& model) {
    if(name=="mega_lucario" || name=="mega_lucario_gbdt" || name=="mega_lucario_transformer")
        return mega_lucario::factory(name,table,deck,model);
    if(name=="dragapult" || name=="dragapult_gbdt" || name=="dragapult_transformer")return dragapult::factory(name,table,deck,model);
    if(name=="iono" || name=="iono_gbdt" || name=="iono_transformer")return iono::factory(name,table,deck,model);
    if(name=="mega_abomasnow" || name=="mega_abomasnow_gbdt" || name=="mega_abomasnow_transformer")return mega_abomasnow::factory(name,table,deck,model);
    throw std::runtime_error("Unknown agent: " + name);
}
} // namespace official
