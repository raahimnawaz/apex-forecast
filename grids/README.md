# Starting grids, corrected for penalties

One CSV per round, `{season}_R{round:02d}.csv`, with exactly two columns:

```csv
driver,grid
NOR,1
LEC,2
PIA,3
```

`driver` is the three-letter code as FastF1 reports it. `grid` is the position the car
actually starts from, after every penalty and pit-lane start has been applied.

## Why this is typed in by hand

Everything under `data/` is gitignored because it can be rebuilt from FastF1 and Jolpica.
This cannot. **No upstream source this project can reach publishes the corrected grid
before the race runs:** FastF1 leaves `GridPosition` empty on a qualifying session and
only fills it in on the race session, which does not exist until the race has happened.
Penalties are public on Saturday evening — they are just not machine-readable.

So a grid here is *source*, not cache, which is why the directory sits outside `data/`
and is committed.

## Why it matters

The round 11 forecast was built off the qualifying classification and six drivers started
somewhere else: Hamilton P2 → P5, Antonelli P4 → P7, and four cars promoted underneath
them. Across 2026 this moves **16% of entries in 5 of 11 races**, and at the Belgian Grand
Prix it moved 20 of 22 cars by up to eleven places.

`build_strength.py` prefers, in order:

1. the actual starting grid, once the race session exists (nothing to do);
2. this file, if present;
3. the qualifying classification — with a loud warning, because penalties are not applied.

A partial file is rejected rather than merged. Mixing penalised and unpenalised positions
in one grid would be worse than either alone, and it would not be visible in the output.
