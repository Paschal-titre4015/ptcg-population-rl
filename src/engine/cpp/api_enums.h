#pragma once
// Numeric values from the official cg.api observation contract.
namespace official {
namespace AreaType {
inline constexpr int ACTIVE = 4;
inline constexpr int BENCH = 5;
inline constexpr int DISCARD = 3;
inline constexpr int HAND = 2;
}
namespace CardType {
inline constexpr int BASIC_ENERGY = 5;
inline constexpr int POKEMON = 0;
inline constexpr int SPECIAL_ENERGY = 6;
inline constexpr int STADIUM = 4;
inline constexpr int SUPPORTER = 3;
inline constexpr int TOOL = 2;
}
namespace EnergyType {
inline constexpr int FIGHTING = 6;
}
namespace LogType {
inline constexpr int ATTACK = 15;
inline constexpr int MOVE_CARD = 6;
inline constexpr int TURN_END = 3;
}
namespace OptionType {
inline constexpr int ABILITY = 10;
inline constexpr int ATTACH = 8;
inline constexpr int ATTACK = 13;
inline constexpr int CARD = 3;
inline constexpr int ENERGY = 6;
inline constexpr int ENERGY_CARD = 5;
inline constexpr int EVOLVE = 9;
inline constexpr int NUMBER = 0;
inline constexpr int PLAY = 7;
inline constexpr int RETREAT = 12;
inline constexpr int YES = 1;
}
namespace SelectContext {
inline constexpr int ATTACH_FROM = 21;
inline constexpr int DAMAGE_COUNTER = 13;
inline constexpr int DAMAGE_COUNTER_ANY = 14;
inline constexpr int DISCARD = 8;
inline constexpr int IS_FIRST = 41;
inline constexpr int MAIN = 0;
inline constexpr int SETUP_ACTIVE_POKEMON = 1;
inline constexpr int SETUP_BENCH_POKEMON = 2;
inline constexpr int SWITCH = 3;
inline constexpr int TO_ACTIVE = 4;
inline constexpr int TO_BENCH = 5;
inline constexpr int TO_HAND = 7;
}
}
