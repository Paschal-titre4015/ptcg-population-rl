#pragma once
#include <cmath>
#include <array>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace gbdt {
// Exported numeric LightGBM trees. Leaves already contain shrinkage.
// No LightGBM/Python runtime dependency is required for inference.
class Model {
    struct Node { int feature=-1, left=-1, right=-1; double threshold=0, value=0; };
    std::array<std::vector<std::vector<Node>>,2> banks;
    int route_feature=-1,stop_feature=-1;
    std::map<int,double> thresholds;
    std::size_t width=0;
public:
    std::string id;
    Model(const std::string& path, const std::string& schema,
          const std::vector<std::string>& names, const std::map<int,int>& deck) {
        std::ifstream in(path);
        auto require=[&](bool condition) { if(!condition || !in) throw std::runtime_error("Invalid/incompatible GBDT model: "+path); };
        auto tag=[&](const std::string& expected) { std::string value; in>>value; require(value==expected); };
        std::string format; in>>format;
        require(format=="PTCG_GBDT_V2" || format=="PTCG_GBDT_V3"); tag(schema); tag("MODEL_ID"); in>>id;
        require(id.size()==64 && id.find_first_not_of("0123456789abcdef")==std::string::npos);
        tag("FEATURES"); int count=0; in>>count; require(count==static_cast<int>(names.size())); width=names.size();
        for(const auto& name:names) tag(name);
        tag("DECK"); in>>count; require(count==static_cast<int>(deck.size()));
        for(const auto& [expected_id,expected_count]:deck) { int card=0,n=0; in>>card>>n; require(card==expected_id && n==expected_count); }
        tag("ROUTE_FEATURE"); in>>route_feature;
        require(route_feature>=0 && route_feature<count_of_features() && names[route_feature]=="context_0");
        tag("BANKS"); tag("2");
        for(int bank=0;bank<2;++bank) {
        tag("BANK"); tag(std::to_string(bank));
        tag("TREES"); int tree_count=0; in>>tree_count; require(tree_count>0 && tree_count<=10000);
        std::size_t total_nodes=0;
        for(int t=0;t<tree_count;++t) {
            tag("NODES"); int n=0; in>>n; require(n>0 && n<=1000000);
            total_nodes+=n; require(total_nodes<=3000000);
            std::vector<Node> nodes(n);
            std::vector<int> incoming(n,0);
            for(int i=0;i<n;++i) {
                auto& node=nodes[i]; in>>node.feature>>node.threshold>>node.left>>node.right>>node.value;
                require(std::isfinite(node.threshold) && std::isfinite(node.value));
                if(node.feature==-1) require(node.left==-1 && node.right==-1);
                else {
                    require(node.feature>=0 && node.feature<count_of_features() && node.left>i && node.right>i && node.left<n && node.right<n && node.left!=node.right);
                    ++incoming[node.left]; ++incoming[node.right];
                }
            }
            require(incoming[0]==0);
            for(int i=1;i<n;++i) require(incoming[i]==1);
            banks[bank].push_back(std::move(nodes));
        }
        }
        if(format=="PTCG_GBDT_V3") {
            tag("STOP_FEATURE");in>>stop_feature;
            require(stop_feature>=0 && stop_feature<count_of_features() && names[stop_feature]=="is_stop");
            tag("THRESHOLDS");int n=0;in>>n;require(n>0 && n<=1000);
            for(int i=0;i<n;++i){int feature=-1;double value=0;in>>feature>>value;
                require(feature>=0 && feature<count_of_features() && names[feature].rfind("context_",0)==0 && std::isfinite(value));
                require(thresholds.emplace(feature,value).second);}
        }
        {
            tag("ARCHIVE_BYTES"); long long bytes=0; in>>bytes;
            require(bytes>0 && bytes<=67108864);
            require(in.get()=='\n'); in.ignore(bytes); require(in.gcount()==bytes);
            require(in.get()=='\n');
        }
        tag("END"); in>>std::ws; require(in.eof());
    }
    int count_of_features() const { return static_cast<int>(width); }
    double predict(const std::vector<double>& features) const {
        if(features.size()!=width) throw std::runtime_error("GBDT input width mismatch");
        for(double value:features) if(!std::isfinite(value)) throw std::runtime_error("GBDT input must be finite");
        if(stop_feature>=0 && features[stop_feature]==1.)
            for(const auto& [context,threshold]:thresholds)if(features[context]==1.)return threshold;
        double score=0;
        if(features[route_feature]!=0. && features[route_feature]!=1.) throw std::runtime_error("Invalid MAIN routing feature");
        for(const auto& nodes:banks[features[route_feature]==1. ? 1 : 0]) {
            int i=0;
            while(nodes[i].feature>=0) {
                const auto& node=nodes[i];
                i=features[node.feature]<=node.threshold ? node.left : node.right;
            }
            score+=nodes[i].value;
        }
        if(!std::isfinite(score)) throw std::runtime_error("Non-finite GBDT prediction");
        return score;
    }
};
} // namespace gbdt
