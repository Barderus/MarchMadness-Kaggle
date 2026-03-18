# March Madness Kaggle

This project builds NCAA March Madness matchup predictions for Kaggle from regular-season data, engineered team features, tournament matchup datasets, and notebook-based model training.

The current workflow is notebook-driven and covers:

- team-level feature generation for men and women
- historical tournament matchup dataset construction
- advanced feature engineering for men
- baseline and tuned model comparison
- Kaggle submission generation for Stage 1 and Stage 2 style files

## Project Layout

`data/`

- raw Kaggle competition CSVs
- generated feature datasets
- saved model comparison outputs

`notebooks/`

- EDA and feature engineering notebooks
- modeling and tuning notebooks
- submission-building notebooks
- generated submission CSVs from notebook runs

`scripts/`

- supporting code used outside the notebooks

`plot_scripts/`

- plotting utilities and one-off chart helpers

## Notebook Summary

[`EDA.ipynb`](/C:/Users/Owner/PycharmProjects/MarchMadness-Kaggle/notebooks/EDA.ipynb)

- explores the raw regular-season, seed, conference, and ranking data
- generates `data/m_team_season_features.csv`
- generates `data/w_team_season_features.csv`

[`matchup_builder.ipynb`](/C:/Users/Owner/PycharmProjects/MarchMadness-Kaggle/notebooks/matchup_builder.ipynb)

- converts historical tournament games into team-vs-team training rows
- merges team season features into the matchup rows
- writes `data/m_tournament_training_dataset.csv`

[`adv_feature_engineering.ipynb`](/C:/Users/Owner/PycharmProjects/MarchMadness-Kaggle/notebooks/adv_feature_engineering.ipynb)

- adds interaction and derived features to the men's tournament training dataset
- writes `data/m_tournament_training_dataset_advanced.csv`

[`Modeling.ipynb`](/C:/Users/Owner/PycharmProjects/MarchMadness-Kaggle/notebooks/Modeling.ipynb)

- compares baseline classifiers on the men's tournament dataset
- writes `data/model_comparison_results_baseline.csv`

[`Modeling_champions_only.ipynb`](/C:/Users/Owner/PycharmProjects/MarchMadness-Kaggle/notebooks/Modeling_champions_only.ipynb)

- tunes Random Forest and XGBoost models
- writes tuned-model comparison results and feature importance CSVs

[`submission_matchup_builder.ipynb`](/C:/Users/Owner/PycharmProjects/MarchMadness-Kaggle/notebooks/submission_matchup_builder.ipynb)

- older submission notebook focused on a Stage 1 style path

[`kaggle_submission.ipynb`](/C:/Users/Owner/PycharmProjects/MarchMadness-Kaggle/notebooks/kaggle_submission.ipynb)

- earlier submission notebook
- not the preferred final path

[`neural_network.ipynb`](/C:/Users/Owner/PycharmProjects/MarchMadness-Kaggle/notebooks/neural_network.ipynb)

- alternative end-to-end neural network submission experiment

[`kaggle_submission_clean.ipynb`](/C:/Users/Owner/PycharmProjects/MarchMadness-Kaggle/notebooks/kaggle_submission_clean.ipynb)

- current best submission notebook
- handles Stage 2 men's and women's rows separately
- writes a corrected combined submission CSV

## Recommended Notebook Order

1. Run `EDA.ipynb`
2. Run `matchup_builder.ipynb`
3. Run `adv_feature_engineering.ipynb`
4. Run `Modeling.ipynb`
5. Run `Modeling_champions_only.ipynb`
6. Run `kaggle_submission_clean.ipynb`

## Main Generated Files

- `data/m_team_season_features.csv`
- `data/w_team_season_features.csv`
- `data/m_tournament_training_dataset.csv`
- `data/m_tournament_training_dataset_advanced.csv`
- `data/model_comparison_results_baseline.csv`
- `data/model_comparison_results_champions.csv`
- `notebooks/submission_stage2_rf_combined.csv`

## Remodel Plan

The proposed notebook restructure is documented in [`PROJECT_REMODEL.md`](/C:/Users/Owner/PycharmProjects/MarchMadness-Kaggle/PROJECT_REMODEL.md).
