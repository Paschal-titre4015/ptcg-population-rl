#include <fstream>
#include <filesystem>
#include <optional>
#include <cstdint>
#include <memory>
#include <string>

#include "All.h"  // Unmodified competition distribution, provided via -I.
#include "official_observation.h"
#include "observation_trace.h"
#include "../../agents/cpp/registry.h"

namespace {
std::array<int, DECK_SIZE> read_deck(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open deck: " + path);
    std::array<int, DECK_SIZE> cards{};
    for (int& card : cards) {
        if (!(input >> card)) throw std::runtime_error("Expected 60 card IDs: " + path);
    }
    input >> std::ws;
    if (!input.eof()) throw std::runtime_error("Extra data after 60 card IDs: " + path);
    return cards;
}

struct Result {
    int winner = -1;
    int steps = 0;
    int agent_errors = 0;
    int engine_errors = 0;
    int error_seat = -1;
};

Result play(const std::array<int, DECK_SIZE>& a, const std::array<int, DECK_SIZE>& b,
            const std::array<std::string, 2>& names, const official::CardTableView& table,
            int game, std::ostream* trace, std::ostream* replay,
            const std::array<official::AgentFactory,2>& factories,
            const std::array<official::SamplingOptions,2>& sampling,
            std::optional<std::uint32_t> seed) {
    Result result;
    try {
        std::array<std::unique_ptr<official::Agent>, 2> agents{
            factories[0].create(sampling[0]),
            factories[1].create(sampling[1])};
        std::array<int, 2 * DECK_SIZE> cards{};
        std::copy(a.begin(), a.end(), cards.begin());
        std::copy(b.begin(), b.end(), cards.begin() + DECK_SIZE);
        const auto start = ApiBattleStart(cards.data());
        if (!start.battlePtr) {
            std::cerr << "BattleStart error: player=" << start.errorPlayer
                      << " type=" << start.errorType << '\n';
            ++result.engine_errors;
            return result;
        }
        std::unique_ptr<ApiData, decltype(&ApiBattleFinish)> battle(start.battlePtr, ApiBattleFinish);
        if (seed) {
            // Reuse the official API's deck validation, then initialize a fresh
            // battle through the same engine lifecycle with reproducible RNG.
            auto config = battle->game.config;
            config.seed = *seed;
            config.deviceRand = false;
            battle.reset(new ApiData());
            battle->apiDataType = 1;
            battle->init(config);
            battle->game.rng = std::mt19937(*seed);
            battle->start();
            battle->next();
        }
        constexpr int max_steps = 10000;
        while (!battle->state.isFinish()) {
            if (result.steps >= max_steps) throw std::runtime_error("Arena step limit reached");
            const int seat = battle->state.selectPlayer;
            const int start_log = battle->state.nextLogStart();
            const auto observation = arena::observe(battle->state, start_log);
            std::vector<int> action;
            try {
                action = agents[seat]->decide(observation);
            } catch (const std::exception& error) {
                std::cerr << "Agent " << names[seat] << ": " << error.what() << '\n';
                result.error_seat=seat;
                ++result.agent_errors;
                return result;
            }
            if (trace || replay) {
                JsonBuilder json, view;
                ToJsonApi(battle->state, json, start_log);
                arena::view_json(view, observation);
                auto write_decision = [&](std::ostream& out, bool training_replay) {
                    out << "{\"type\":\"decision\",\"game\":" << game << ",\"step\":" << result.steps
                        << ",\"seat\":" << seat << ",\"agent\":\"" << names[seat]
                        << "\",\"observation\":" << reinterpret_cast<const char*>(json.buf.c_str())
                        << ",\"view\":" << reinterpret_cast<const char*>(view.buf.c_str()) << ",\"action\":[";
                    for (int i = 0; i < static_cast<int>(action.size()); ++i) {
                        if (i) out << ',';
                        out << action[i];
                    }
                    out << ']';
                    if (training_replay) agents[seat]->write_training_record(out, observation, action);
                    out << "}\n";
                    if (!out) throw std::runtime_error("Cannot write decision record");
                };
                if (trace) write_decision(*trace, false);
                if (replay) write_decision(*replay, true);
            }
            const int error = ApiSelect(battle.get(), action.data(), static_cast<int>(action.size()));
            if (error != 0) {
                std::cerr << "Agent selection rejected: code=" << error << '\n';
                result.error_seat=seat;
                ++result.agent_errors;
                return result;
            }
            ++result.steps;
        }
        result.winner = static_cast<int>(battle->state.gameResult) - 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        ++result.engine_errors;
    }
    return result;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::string deck_a, deck_b, trace_path, metadata_path, replay_dir;
        std::array<std::string, 2> model_paths;
        std::array<official::SamplingOptions,2> sampling;
        std::optional<std::uint32_t> seed;
        std::array<std::string, 2> names{};
        int games = 1;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--help") {
                std::cout << "arena_cpp --deck-a PATH --deck-b PATH [--games N]\n"
                             "--agent-a NAME --agent-b NAME [--trace PATH] [--metadata PATH]\n"
                             "[--sample-seed-a N --sample-seed-b N] [--temperature X]\n"
                             "[--replay-dir PATH] [--seed N] [--model-a PATH --model-b PATH]\n"
                             "Agents: mega_lucario, dragapult, iono, mega_abomasnow (also _gbdt and _transformer)\n";
                return 0;
            }
            if (i + 1 >= argc) throw std::runtime_error("Missing argument value: " + flag);
            const std::string value = argv[++i];
            if (flag == "--deck-a") deck_a = value;
            else if (flag == "--deck-b") deck_b = value;
            else if (flag == "--agent-a") names[0] = value;
            else if (flag == "--agent-b") names[1] = value;
            else if (flag == "--trace") trace_path = value;
            else if (flag == "--metadata") metadata_path = value;
            else if (flag == "--replay-dir") replay_dir = value;
            else if (flag == "--model-a") model_paths[0] = value;
            else if (flag == "--model-b") model_paths[1] = value;
            else if(flag=="--temperature"){
                std::size_t consumed=0;double temperature=std::stod(value,&consumed);
                if(consumed!=value.size() || !std::isfinite(temperature) || temperature<=0)throw std::runtime_error("Invalid temperature");
                for(auto& config:sampling)config.temperature=temperature;
            }
            else if (flag == "--seed" || flag == "--sample-seed-a" || flag == "--sample-seed-b") {
                std::size_t consumed=0;
                const auto parsed=std::stoull(value,&consumed);
                if(consumed!=value.size() || parsed>UINT32_MAX) throw std::runtime_error("Invalid --seed");
                if(flag=="--seed")seed=static_cast<std::uint32_t>(parsed);
                else {auto& config=sampling[flag=="--sample-seed-a" ? 0 : 1];config.enabled=true;config.seed=static_cast<std::uint32_t>(parsed);}
            }
            else if (flag == "--games") {
                std::size_t consumed = 0;
                games = std::stoi(value, &consumed);
                if (consumed != value.size() || games < 1) throw std::runtime_error("Invalid --games");
            } else throw std::runtime_error("Unknown argument: " + flag);
        }
        if (deck_a.empty() || deck_b.empty()) throw std::runtime_error("Both --deck-a and --deck-b are required");
        if (names[0].empty() || names[1].empty()) throw std::runtime_error("Both --agent-a and --agent-b are required");
        const auto a = read_deck(deck_a);
        const auto b = read_deck(deck_b);
        InitializeAll();
        const auto table = arena::master_view();
        if (seed && static_cast<std::uint64_t>(*seed)+games-1>UINT32_MAX) throw std::runtime_error("Seed range overflow");
        const std::array<official::AgentFactory,2> factories{
            official::make_factory(names[0],table,std::vector<int>(a.begin(),a.end()),model_paths[0]),
            official::make_factory(names[1],table,std::vector<int>(b.begin(),b.end()),model_paths[1])};
        // Each acting player owns its feature contract. Different decks/models
        // may share an arena without sharing a learning schema.
        const std::string schema_id=factories[0].schema_id==factories[1].schema_id ? factories[0].schema_id : "";
        for(int seat=0;seat<2;++seat)if(sampling[seat].enabled){
            if(!factories[seat].supports_sampling)throw std::runtime_error("Agent does not support sampling");
            if(!seed || replay_dir.empty())throw std::runtime_error("PPO sampling requires --seed and --replay-dir");
            if(static_cast<std::uint64_t>(sampling[seat].seed)+games-1>UINT32_MAX)throw std::runtime_error("Sampling seed overflow");
        }
        // Validate names and deck contracts before starting any games.
        factories[0].create(sampling[0]);
        factories[1].create(sampling[1]);
        if(!replay_dir.empty()) {
            std::filesystem::create_directories(replay_dir);
            if(!std::filesystem::is_empty(replay_dir)) throw std::runtime_error("Replay directory must be empty");
            JsonBuilder json; ApiAllCard(json);
            std::ofstream metadata(std::filesystem::path(replay_dir)/"card_metadata.json");
            metadata<<reinterpret_cast<const char*>(json.buf.c_str())<<'\n';
            if(!metadata) throw std::runtime_error("Cannot write replay metadata");
        }
        std::ofstream trace;
        if (!trace_path.empty()) {
            trace.open(trace_path);
            if (!trace) throw std::runtime_error("Cannot open trace: " + trace_path);
        }
        if (!metadata_path.empty()) {
            JsonBuilder json;
            ApiAllCard(json);
            std::ofstream metadata(metadata_path);
            metadata << reinterpret_cast<const char*>(json.buf.c_str()) << '\n';
            if (!metadata) throw std::runtime_error("Cannot write metadata: " + metadata_path);
        }
        int errors = 0;
        for (int game = 0; game < games; ++game) {
            auto game_sampling=sampling;
            for(auto& config:game_sampling)if(config.enabled)config.seed+=game;
            std::ofstream replay;
            if(!replay_dir.empty()) {
                replay.open(std::filesystem::path(replay_dir)/("game-"+std::to_string(game)+".jsonl"));
                if(!replay) throw std::runtime_error("Cannot open replay");
                replay<<"{\"type\":\"header\",\"format\":\"ptcg-replay-v1\",\"game\":"<<game
                      <<",\"schema_id\":\""<<schema_id<<"\",\"seed\":";
                if(seed) replay<<static_cast<std::uint64_t>(*seed)+game; else replay<<"null";
                replay<<",\"schemas\":[\""<<factories[0].schema_id<<"\",\""<<factories[1].schema_id<<"\"]";
                replay<<",\"agents\":[\""<<names[0]<<"\",\""<<names[1]<<"\"],\"models\":[";
                for(int seat=0;seat<2;++seat) {if(seat) replay<<',';if(!factories[seat].model_id.empty()) replay<<'"'<<factories[seat].model_id<<'"';else replay<<"null";}
                replay<<"],\"decks\":[";
                for(int seat=0;seat<2;++seat) {if(seat) replay<<',';replay<<'[';const auto& deck=seat==0?a:b;
                    for(int i=0;i<DECK_SIZE;++i) {if(i) replay<<',';replay<<deck[i];}replay<<']';}
                replay<<']';
                if(sampling[0].enabled || sampling[1].enabled){
                    replay<<",\"sampling\":[";
                    for(int seat=0;seat<2;++seat){if(seat)replay<<',';
                        const auto& config=game_sampling[seat];
                        if(config.enabled)replay<<"{\"seed\":"<<config.seed<<",\"temperature\":"<<std::setprecision(17)<<config.temperature<<'}';
                        else replay<<"null";
                    }
                    replay<<']';
                }
                replay<<"}\n";
            }
            const auto game_seed=seed ? std::optional<std::uint32_t>(*seed+game) : std::nullopt;
            const auto result = play(a, b, names, table, game, trace_path.empty() ? nullptr : &trace,
                                     replay_dir.empty() ? nullptr : &replay, factories, game_sampling, game_seed);
            if(replay.is_open()) {
                replay<<"{\"type\":\"terminal\",\"game\":"<<game<<",\"result\":"<<result.winner
                      <<",\"steps\":"<<result.steps<<",\"agent_errors\":"<<result.agent_errors
                      <<",\"engine_errors\":"<<result.engine_errors<<"}\n";
                replay.flush();
                if(!replay) throw std::runtime_error("Cannot finish replay");
            }
            errors += result.agent_errors + result.engine_errors;
            std::cout << "{\"game\":" << game << ",\"agent_a\":\"" << names[0] << "\",\"agent_b\":\"" << names[1] << "\",\"result\":"
                      << result.winner << ",\"steps\":" << result.steps
                      << ",\"agent_errors\":" << result.agent_errors
                      << ",\"engine_errors\":" << result.engine_errors << ",\"error_seat\":" << result.error_seat << "}\n";
        }
        return errors ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
