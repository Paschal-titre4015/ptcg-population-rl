#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include "mega_lucario_schema.h"
#include "model.h"
int main(int argc,char** argv) {
    try {
        if(argc!=3)throw std::runtime_error("Usage: benchmark model.bin repetitions < tokens");
        int repetitions=std::stoi(argv[2]);if(repetitions<1)throw std::runtime_error("Positive repetitions required");
        std::map<int,int> deck;for(std::size_t i=0;i<lucario_schema::IDS.size();++i)deck[lucario_schema::IDS[i]]=lucario_schema::COUNTS[i];
        transformer::Model model(argv[1],lucario_schema::TRANSFORMER_ID,lucario_schema::ID,198,deck);
        int n,width;if(!(std::cin>>n>>width) || n<20 || n>120 || width!=198)throw std::runtime_error("Bad benchmark shape");
        std::vector<std::vector<double>> rows(n,std::vector<double>(width));
        for(auto& row:rows)for(auto& x:row)if(!(std::cin>>x))throw std::runtime_error("Incomplete input");
        transformer::Model::Output result;
        for(int i=0;i<3;++i)result=model.evaluate(rows);
        auto start=std::chrono::steady_clock::now();auto cpu_start=std::clock();
        for(int i=0;i<repetitions;++i)result=model.evaluate(rows);
        double cpu_ms=1000.*(std::clock()-cpu_start)/CLOCKS_PER_SEC/repetitions;
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repetitions;
        std::cout<<std::setprecision(17)<<"{\"backend\":\""<<transformer::numeric::backend()<<"\",\"milliseconds\":"<<ms<<",\"cpu_milliseconds\":"<<cpu_ms<<",\"value\":"<<result.value<<",\"scores\":[";
        for(std::size_t i=0;i<result.scores.size();++i){if(i)std::cout<<',';if(std::isfinite(result.scores[i]))std::cout<<result.scores[i];else std::cout<<"null";}
        std::cout<<"]}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
