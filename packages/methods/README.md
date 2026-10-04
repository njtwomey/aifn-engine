# aifn-methods (packages/methods)

Applications of the aifn compute: named models, problems, environments, datasets and worked examples. What makes code an
application rather than compute, the import rules and the promotion rule are in `../../README.md`; the areas, their groups,
their order are in `../../modules.json` (and the generated "Areas" table in `../../README.md`).

## Layout

- `src/<area>/index.ts` is an area's public surface, imported as `aifn-methods/<area>` (e.g. `aifn-methods/learning`).
  Areas nest groups and modules (`src/learning/generalised/ordinal/`), imported by path
  (`aifn-methods/learning/generalised/ordinal`). A group's root files are its shared layer; siblings share only
  through the parent.
- `test/<area>/…/*.test.ts` mirrors the source tree. Root files: `test/names.test.ts` (the name-collision lint across
  both packages, `make names`), `test/registry.ts` and `test/model-fixtures.ts` (the model protocol test fits
  every registered model on tiny data), `test/protocol.ts`.
- `test/fixtures/<area>/…json` holds golden values written by the Python scripts in `test/fixtures/gen/` (run by
  `make fixtures`, which covers both packages); `test/fixtures.ts` loads them.
- Applications import any compute node through `aifn-compute/<family>/<module>`, their ancestors' shared files by relative path,
  and other areas only down the area DAG. No React, no DOM.

## Registries

Applications register their named variants with compute's `define`/`entries` (`aifn-compute/foundation/registry`), as compute does.

| Kind          | Table                                                                                             | Count | Kind-specific info                                                                                                           |
| ------------- | ------------------------------------------------------------------------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------- |
| `model`       | `learningModelRegistry` (`aifn-methods/learning`), `unsupervisedModelRegistry` (`…/unsupervised`) | 66    | `task`, declared `capabilities` (`decide`, `scores`, `predictive`, `expect`, `transform`, …), `hyper: Space`, `transductive` |
| `dataset`     | `datasetRegistry` (`aifn-methods/data`), `fontDatasetRegistry` (`aifn-methods/data/real/fonts`)   | 36    | `task`, `knobs: Space`, `truth` (a known generating model), `output`                                                         |
| `modifier`    | `modifierRegistry` (`aifn-methods/data`)                                                          | 9     | `params: Space`, `needs: 'labels'` where it applies                                                                          |
| `environment` | `environmentRegistry` (`aifn-methods/gym`)                                                        | 8     | `family`, `params: Space`, `observation`/`action` domain kinds, `capabilities` (`model`, `oracle`, `render`)                 |
| `agent`       | `agentRegistry` (`aifn-methods/gym`)                                                              | 22    | `params: Space`, `requires` (domain kinds, `model`, `families`); `validPairs()` lists the runnable pairs                     |
| `objective`   | `objectiveRegistry` (`aifn-methods/data`): optimisation test surfaces                             | 5     | `params: Space`, `dim`, `truth` (known minimisers)                                                                           |
| `log-density` | `logDensityRegistry` (`aifn-methods/data`): MCMC and VI targets                                   | 5     | `params: Space`, `dim`, `truth` (a known reference)                                                                          |

Models are built with compute's `defineModel` (`aifn-compute/learning/estimators`); the protocol test checks each one's
capabilities and `transductive` flag against its declaration. Datasets and modifiers are replayed by **recipes**: a
recipe names a base dataset and a list of modifiers with their parameters; `recipe` (in `aifn-methods/data`) builds
it, `meta.recipe` holds the normalised recipe that rebuilds the data, `meta.ignored` lists the knobs and steps that did
not apply, and `recipeSpace` is the `Space` of every base's knobs (for the lab's controls).

## Running

`make test` runs every package; one area with `npx vitest run --config packages/methods/vitest.config.ts
test/learning`.

## Style

The module contract of `../compute/README.md` applies: readable before clever, cited, typed, numerical failure reported.
Applications are read by learners as the worked form of an idea, so they favour a plain loop over a clever one.
