#include <iomanip>
#include <iostream>
#include "mega_abomasnow_schema.h"
#include "model.h"
int main(int argc,char** argv){
    try {
        if(argc!=2 && !(argc==3 && (std::string(argv[2])=="--value" || std::string(argv[2])=="--prepared")))throw std::runtime_error("Usage: transformer_probe model.bin [--value | --prepared]");
        std::map<int,int> deck;for(std::size_t i=0;i<mega_abomasnow_schema::IDS.size();++i)deck[mega_abomasnow_schema::IDS[i]]=mega_abomasnow_schema::COUNTS[i];
        mega_abomasnow_transformer::Model model(argv[1],mega_abomasnow_schema::TRANSFORMER_ID,mega_abomasnow_schema::ID,198,deck);
        mega_abomasnow_transformer::Model::Prepared prepared;
        int count,width;
        while(std::cin>>count>>width){
            if(count<1 || count>120 || width!=198)throw std::runtime_error("Invalid probe shape");
            std::vector<std::vector<double>> rows(count,std::vector<double>(width));
            for(auto& row:rows)for(auto& v:row)if(!(std::cin>>v))throw std::runtime_error("Incomplete probe input");
            const bool cached=argc==3 && std::string(argv[2])=="--prepared";
            if(cached && !prepared.owner)prepared=model.prepare(rows);
            auto output=cached ? model.evaluate(prepared,rows) : model.evaluate(rows);
            for(double v:output.scores)std::cout<<std::setprecision(17)<<v<<' ';
            if(argc==3 && (std::string(argv[2])=="--value" || cached))std::cout<<std::setprecision(17)<<output.value;
            std::cout<<'\n';
        }
        if(!std::cin.eof())throw std::runtime_error("Invalid probe input");
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
