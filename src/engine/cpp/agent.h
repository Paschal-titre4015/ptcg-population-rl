#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <cstdint>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Only acting-player API fields used by the official policies. No engine State
// or hidden deck/hand is available to a policy through this interface.
namespace official {
struct CardView {
    bool present = false, is_pokemon = false;
    int id = 0, serial = -1, playerIndex = -1, hp = 0, maxHp = 0;
    bool appearThisTurn = false;
    std::vector<int> energies;
    std::vector<CardView> energyCards, tools, preEvolution;
    bool operator==(std::nullptr_t) const { return !present; }
};
struct PlayerView {
    std::vector<CardView> hand, active, bench, discard, prize;
    int handCount = 0, deckCount = 0;
    bool asleep = false, paralyzed = false;
};
struct CurrentView {
    int turn = 0, yourIndex = 0, firstPlayer = 0;
    bool supporterPlayed = false, energyAttached = false, has_looking = false;
    std::vector<CardView> stadium, looking;
    std::array<PlayerView, 2> players;
};
struct OptionView {
    int type = -1, number = -1, area = -1, index = -1, playerIndex = -1;
    int inPlayArea = -1, inPlayIndex = -1, attackId = -1;
};
struct SelectView {
    int context = 0, minCount = 0, maxCount = 0, remainDamageCounter = 0;
    bool has_deck = false;
    std::vector<OptionView> option;
    std::vector<CardView> deck;
    CardView effect, contextCard;
};
struct LogView {
    int type = -1, playerIndex = -1, attackId = -1, fromArea = -1, toArea = -1;
};
struct Observation {
    CurrentView current;
    SelectView select;
    std::vector<LogView> logs;
};
struct MasterView {
    int cardType = -1, weakness = -1, resistance = -1;
    bool megaEx = false, ex = false, stage1 = false, stage2 = false;
    std::string name;
};
using CardTableView = std::map<int, MasterView>;

inline CardView get_card(const Observation& obs, int area, int index, int player) {
    const auto& ps = obs.current.players.at(player);
    switch (area) {
    case 1: return obs.select.deck.at(index);
    case 2: return ps.hand.at(index);
    case 3: return ps.discard.at(index);
    case 4: return ps.active.at(index);
    case 5: return ps.bench.at(index);
    case 6: return ps.prize.at(index);
    case 7: return obs.current.stadium.at(index);
    case 12: return obs.current.looking.at(index);
    default: return {};
    }
}
// Python counts are signed, and its division always produces a floating value.
template<class T> int count_of(const T& values) { return static_cast<int>(values.size()); }
template<class T> T& at_index(std::vector<T>& values, int index) {
    return values.at(index < 0 ? count_of(values) + index : index);
}
template<class T> const T& at_index(const std::vector<T>& values, int index) {
    return values.at(index < 0 ? count_of(values) + index : index);
}
template<class T> std::vector<T> concat(std::vector<T> a, const std::vector<T>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}
template<class T> bool contains(const std::vector<T>& values, const T& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}
inline std::vector<int> ranked_indices(const std::vector<double>& scores) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    return indices;
}
inline std::vector<std::pair<int, double>> ranked_scores(const std::vector<double>& scores) {
    std::vector<std::pair<int, double>> out;
    for (int index : ranked_indices(scores)) out.emplace_back(index, scores[index]);
    return out;
}
inline std::vector<int> take(std::vector<int> values, int count) {
    if (count < 0 || count > count_of(values)) throw std::runtime_error("Invalid selection count");
    values.resize(count);
    return values;
}
class Agent {
public:
    virtual ~Agent() = default;
    virtual std::vector<int> decide(const Observation& obs) = 0;
    virtual void write_training_record(std::ostream&, const Observation&, const std::vector<int>&) const {};
};
struct SamplingOptions { bool enabled=false; std::uint32_t seed=0; double temperature=1.; };
struct AgentFactory {
    std::string schema_id, model_id;
    bool supports_sampling=false;
    std::function<std::unique_ptr<Agent>(SamplingOptions)> create;
};
} // namespace official
