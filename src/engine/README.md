# C++ arena usage

Run matches between the four official rule policies, GBDT policies, and Transformers in C++.
See the [root README](../../README.md#set-up-uv) for environment setup. Run commands from the repository root.

## Download and build

Configure Kaggle credentials and access to the [competition data](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/data).

```bash
uv run python src/tools/fetch_official_assets.py
uv run python src/tools/fetch_cg_engine.py
uv run python src/tools/build_cpp_engine.py
```

Decks are bundled at `src/agents/<agent>/deck.csv`. The engine is installed into `src/engine/cg/`; the executable is written to `build/arena_cpp`.
Refer to the downloaded `cg/README.md` and `cg/LICENSES/` for the engine's terms of use.

Use these options to change paths or verify an existing installation:

| Command | Option | Purpose |
|---|---|---|
| `fetch_cg_engine.py` | `--cache-dir PATH` | Download cache |
| `fetch_cg_engine.py` | `--destination PATH` | Engine installation directory |
| `fetch_cg_engine.py` | `--verify-only` | Verify the installed engine |
| `build_cpp_engine.py` | `--engine-dir PATH` | Engine to compile against |
| `build_cpp_engine.py` | `--output PATH` | Output executable |

Set the `CXX` environment variable to choose a compiler.

## Play matches

```bash
build/arena_cpp \
  --agent-a mega_lucario --agent-b dragapult \
  --deck-a src/agents/mega_lucario/deck.csv \
  --deck-b src/agents/dragapult/deck.csv --games 2
```

Both seats require `--agent-a` / `--agent-b` and `--deck-a` / `--deck-b`. Supply the deck corresponding to each policy.

| Deck | Rule policy name |
|---|---|
| Mega Lucario ex | `mega_lucario` |
| Dragapult ex | `dragapult` |
| Iono | `iono` |
| Mega Abomasnow ex | `mega_abomasnow` |

For trained policies, append `_gbdt` or `_transformer` and supply the corresponding `--model-a` / `--model-b`.
GBDT uses `.gbdt`; Transformer and PPO use `.bin`.

Set `--games N` for the match count and `--seed N` for reproducibility. Each match uses seed `N + game`, with `game` starting at zero.
Seats 0/1 do not specify who goes first.

The arena writes one JSON result per match to standard output.
`result` is the winner's seat (0/1) or a draw (2); `steps` counts decisions.
`agent_errors` / `engine_errors` count errors. `error_seat` identifies the responsible agent seat, or -1 if unknown.
Errors produce a nonzero exit code.

## Record matches

Add `--replay-dir PATH` to save the card dictionary and one JSONL file per match. Use an empty directory.

Use `--trace PATH` to save observations and actions, or `--metadata PATH` to save the card dictionary separately.
Create their parent directories beforehand.

For stochastic Transformer actions during training, set `--sample-seed-a N` / `--sample-seed-b N` for the relevant seats.
This also requires `--seed` and `--replay-dir`. Set `--temperature` to adjust the sampling distribution.

Training commands are in each [agent's](../../README.md#supported-agents) `gbdt/scripts/`, `transformer/scripts/`, and `ppo/scripts/` directories.
Run `build/arena_cpp --help` for all options.
