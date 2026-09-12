// Numeric prediction probe for independent Python/LightGBM parity checks.
#include <iomanip>
#include <iostream>
#include "model.h"
#include "iono_schema.h"
int main(int argc,char** argv) {
    try {
        if(argc!=2) throw std::runtime_error("Usage: gbdt_probe MODEL < feature_rows.txt");
        std::map<int,int> deck;
        for(std::size_t i=0;i<iono_schema::IDS.size();++i) deck[iono_schema::IDS[i]]=iono_schema::COUNTS[i];
        iono_gbdt::Model model(argv[1],iono_schema::ID,iono_schema::NAMES,deck);
        int rows=0,width=0;
        if(!(std::cin>>rows>>width) || rows<1 || width!=model.count_of_features()) throw std::runtime_error("Invalid probe shape");
        std::cout<<std::setprecision(17);
        std::vector<double> row(width);
        for(int i=0;i<rows;++i) {
            for(auto& value:row) if(!(std::cin>>value)) throw std::runtime_error("Invalid probe value");
            std::cout<<model.predict(row)<<'\n';
        }
        std::cin>>std::ws;
        if(!std::cin.eof()) throw std::runtime_error("Extra probe input");
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
