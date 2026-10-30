# Published projection parameters

`best_params.json` contains the fixed parameters used to reproduce the published
forest projections. The prediction script uses its XGBoost settings by default:
689 trees for basal area and 521 trees for broadleaf fraction.

This file contains fitting settings. The fitted models are supplied separately
in the `published_projection_ensemble_weights.tar.xz` release asset. Its bundled
`hyperparameters/best_params.json` is identical to this file.

For predictions from the saved ensemble, follow the repository README's
standalone Python example.
