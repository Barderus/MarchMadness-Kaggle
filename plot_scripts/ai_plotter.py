# AI wrote this whole file just to get some basic information about the datasets

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Data paths
DATA_DIR = Path("../data")
OUTPUT_DIR = Path("plots")


def load_data():
    """Load all necessary data files"""
    print("Loading data files...")

    # Teams
    m_teams = pd.read_csv(DATA_DIR / "MTeams.csv")
    w_teams = pd.read_csv(DATA_DIR / "WTeams.csv")

    # Seeds
    m_seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    w_seeds = pd.read_csv(DATA_DIR / "WNCAATourneySeeds.csv")

    # Tournament results (compact)
    m_tourney_compact = pd.read_csv(DATA_DIR / "MNCAATourneyCompactResults.csv")
    w_tourney_compact = pd.read_csv(DATA_DIR / "WNCAATourneyCompactResults.csv")

    # Detailed results (regular season and tournament)
    m_reg_detailed = pd.read_csv(DATA_DIR / "MRegularSeasonDetailedResults.csv")
    w_reg_detailed = pd.read_csv(DATA_DIR / "WRegularSeasonDetailedResults.csv")
    m_tourney_detailed = pd.read_csv(DATA_DIR / "MNCAATourneyDetailedResults.csv")
    w_tourney_detailed = pd.read_csv(DATA_DIR / "WNCAATourneyDetailedResults.csv")

    # Rankings (men only)
    m_rankings = pd.read_csv(DATA_DIR / "MMasseyOrdinals.csv")

    # Conferences
    m_conf = pd.read_csv(DATA_DIR / "MTeamConferences.csv")
    w_conf = pd.read_csv(DATA_DIR / "WTeamConferences.csv")

    # Sample submission
    sample_sub = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")

    return {
        "m_teams": m_teams,
        "w_teams": w_teams,
        "m_seeds": m_seeds,
        "w_seeds": w_seeds,
        "m_tourney_compact": m_tourney_compact,
        "w_tourney_compact": w_tourney_compact,
        "m_reg_detailed": m_reg_detailed,
        "w_reg_detailed": w_reg_detailed,
        "m_tourney_detailed": m_tourney_detailed,
        "w_tourney_detailed": w_tourney_detailed,
        "m_rankings": m_rankings,
        "m_conf": m_conf,
        "w_conf": w_conf,
        "sample_sub": sample_sub,
    }


def add_gender(data_dict):
    """Add gender column to all relevant dataframes"""
    # Men's data: 0
    # Women's data: 1

    data_dict["m_teams"]["Gender"] = 0
    data_dict["w_teams"]["Gender"] = 1

    data_dict["m_seeds"]["Gender"] = 0
    data_dict["w_seeds"]["Gender"] = 1

    data_dict["m_tourney_compact"]["Gender"] = 0
    data_dict["w_tourney_compact"]["Gender"] = 1

    data_dict["m_reg_detailed"]["Gender"] = 0
    data_dict["w_reg_detailed"]["Gender"] = 1

    data_dict["m_tourney_detailed"]["Gender"] = 0
    data_dict["w_tourney_detailed"]["Gender"] = 1

    data_dict["m_conf"]["Gender"] = 0
    data_dict["w_conf"]["Gender"] = 1

    return data_dict


# Load data
data_dict = load_data()
data_dict = add_gender(data_dict)


def prepare_winner_features(df):
    """Create features for each game from the winning team's perspective"""
    games = []
    for _, row in df.iterrows():
        game = {
            "Season": row["Season"],
            "WTeamID": row["WTeamID"],
            "LTeamID": row["LTeamID"],
            "Win": 1,
            "ScoreMargin": row["WScore"] - row["LScore"],
            "FGM": row["WFGM"],
            "FGA": row["WFGA"],
            "FGPct": row["WFGM"] / row["WFGA"] if row["WFGA"] > 0 else 0,
            "FGM3": row["WFGM3"],
            "FGA3": row["WFGA3"],
            "FG3Pct": row["WFGM3"] / row["WFGA3"] if row["WFGA3"] > 0 else 0,
            "FTM": row["WFTM"],
            "FTA": row["WFTA"],
            "FTPct": row["WFTM"] / row["WFTA"] if row["WFTA"] > 0 else 0,
            "OR": row["WOR"],
            "DR": row["WDR"],
            "TotalReb": row["WOR"] + row["WDR"],
            "Ast": row["WAst"],
            "TO": row["WTO"],
            "Stl": row["WStl"],
            "Blk": row["WBlk"],
            "PF": row["WPF"],
            "OppFGM": row["LFGM"],
            "OppFGA": row["LFGA"],
            "OppFGPct": row["LFGM"] / row["LFGA"] if row["LFGA"] > 0 else 0,
            "OppTO": row["LTO"],
            "OppOR": row["LOR"],
            "OppDR": row["LDR"],
        }
        games.append(game)

        game_loss = {
            "Season": row["Season"],
            "WTeamID": row["LTeamID"],
            "LTeamID": row["WTeamID"],
            "Win": 0,
            "ScoreMargin": row["LScore"] - row["WScore"],
            "FGM": row["LFGM"],
            "FGA": row["LFGA"],
            "FGPct": row["LFGM"] / row["LFGA"] if row["LFGA"] > 0 else 0,
            "FGM3": row["LFGM3"],
            "FGA3": row["LFGA3"],
            "FG3Pct": row["LFGM3"] / row["LFGA3"] if row["LFGA3"] > 0 else 0,
            "FTM": row["LFTM"],
            "FTA": row["LFTA"],
            "FTPct": row["LFTM"] / row["LFTA"] if row["LFTA"] > 0 else 0,
            "OR": row["LOR"],
            "DR": row["LDR"],
            "TotalReb": row["LOR"] + row["LDR"],
            "Ast": row["LAst"],
            "TO": row["LTO"],
            "Stl": row["LStl"],
            "Blk": row["LBlk"],
            "PF": row["LPF"],
            "OppFGM": row["WFGM"],
            "OppFGA": row["WFGA"],
            "OppFGPct": row["WFGM"] / row["WFGA"] if row["WFGA"] > 0 else 0,
            "OppTO": row["WTO"],
            "OppOR": row["WOR"],
            "OppDR": row["WDR"],
        }
        games.append(game_loss)

    return pd.DataFrame(games)


def compute_team_season_stats(reg_df, tourney_df):
    """Compute average team stats for the season and merge with tournament results"""
    team_stats = []

    for season in reg_df["Season"].unique():
        season_reg = reg_df[reg_df["Season"] == season]

        for team_id in set(season_reg["WTeamID"].unique()) | set(
            season_reg["LTeamID"].unique()
        ):
            wins = season_reg[season_reg["WTeamID"] == team_id]
            losses = season_reg[season_reg["LTeamID"] == team_id]

            games_won = len(wins)
            games_lost = len(losses)
            total_games = games_won + games_lost

            if total_games == 0:
                continue

            stats = {
                "Season": season,
                "TeamID": team_id,
                "WinPct": games_won / total_games if total_games > 0 else 0,
                "Games": total_games,
            }

            win_stats = wins[
                [
                    "WFGM",
                    "WFGA",
                    "WFGM3",
                    "WFGA3",
                    "WFTM",
                    "WFTA",
                    "WOR",
                    "WDR",
                    "WAst",
                    "WTO",
                    "WStl",
                    "WBlk",
                    "WScore",
                ]
            ].mean()
            loss_stats = losses[
                [
                    "LFGM",
                    "LFGA",
                    "LFGM3",
                    "LFGA3",
                    "LFTM",
                    "LFTA",
                    "LOR",
                    "LDR",
                    "LAst",
                    "LTO",
                    "LStl",
                    "LBlk",
                    "LScore",
                ]
            ].mean()

            stats["AvgPoints"] = (
                win_stats["WScore"] * games_won + loss_stats["LScore"] * games_lost
            ) / total_games
            stats["AvgFGPct"] = (
                (win_stats["WFGM"] / win_stats["WFGA"] if win_stats["WFGA"] > 0 else 0)
                * games_won
                / total_games
            )
            stats["AvgFG3Pct"] = (
                (
                    win_stats["WFGM3"] / win_stats["WFGA3"]
                    if win_stats["WFGA3"] > 0
                    else 0
                )
                * games_won
                / total_games
            )
            stats["AvgFTPct"] = (
                (win_stats["WFTM"] / win_stats["WFTA"] if win_stats["WFTA"] > 0 else 0)
                * games_won
                / total_games
            )
            stats["AvgOR"] = (
                win_stats["WOR"] * games_won + loss_stats["LOR"] * games_lost
            ) / total_games
            stats["AvgDR"] = (
                win_stats["WDR"] * games_won + loss_stats["LDR"] * games_lost
            ) / total_games
            stats["AvgAst"] = (
                win_stats["WAst"] * games_won + loss_stats["LAst"] * games_lost
            ) / total_games
            stats["AvgTO"] = (
                win_stats["WTO"] * games_won + loss_stats["LTO"] * games_lost
            ) / total_games
            stats["AvgStl"] = (
                win_stats["WStl"] * games_won + loss_stats["LStl"] * games_lost
            ) / total_games
            stats["AvgBlk"] = (
                win_stats["WBlk"] * games_won + loss_stats["LBlk"] * games_lost
            ) / total_games

            team_stats.append(stats)

    return pd.DataFrame(team_stats)


def plot_correlation_heatmaps():
    """Plot correlation heatmaps for key features vs winning"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    m_reg = data_dict["m_reg_detailed"]
    games_df = prepare_winner_features(m_reg)

    feature_cols = [
        "FGPct",
        "FG3Pct",
        "FTPct",
        "TotalReb",
        "Ast",
        "TO",
        "Stl",
        "Blk",
        "OppFGPct",
        "OppTO",
        "ScoreMargin",
    ]

    corr = games_df[["Win"] + feature_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        cmap="RdBu_r",
        center=0,
        fmt=".2f",
        square=True,
        linewidths=0.5,
    )
    plt.title("Correlation Heatmap: Team Features vs Winning", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved: correlation_heatmap.png")

    feature_corrs = corr["Win"].drop("Win").sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feature_corrs.plot(
        kind="barh", color=["green" if x > 0 else "red" for x in feature_corrs]
    )
    plt.xlabel("Correlation with Winning")
    plt.title("Feature Correlations with Winning")
    plt.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_correlations.png", dpi=150)
    plt.close()
    print("Saved: feature_correlations.png")


def plot_seed_correlation():
    """Plot correlation between seed and tournament success"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    m_seeds = data_dict["m_seeds"]
    m_tourney = data_dict["m_tourney_compact"]

    m_seeds["SeedNum"] = (
        m_seeds["Seed"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    )

    m_tourney = m_tourney.merge(
        m_seeds[["Season", "TeamID", "SeedNum"]],
        left_on=["Season", "WTeamID"],
        right_on=["Season", "TeamID"],
        how="inner",
    )
    m_tourney = m_tourney.dropna(subset=["SeedNum"])
    m_tourney["SeedNum"] = m_tourney["SeedNum"].astype(int)

    plt.figure(figsize=(10, 6))
    m_tourney.groupby("SeedNum").size().plot(kind="bar", color="steelblue")
    plt.xlabel("Seed Number")
    plt.ylabel("Number of Wins")
    plt.title("Tournament Wins by Seed Number")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wins_by_seed.png", dpi=150)
    plt.close()
    print("Saved: wins_by_seed.png")


def plot_win_pct_vs_tourney():
    """Plot regular season win percentage vs tournament success"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    m_reg = data_dict["m_reg_detailed"]
    m_tourney = data_dict["m_tourney_compact"]

    team_stats = compute_team_season_stats(m_reg, m_tourney)

    tourney_wins = (
        m_tourney.groupby(["Season", "WTeamID"]).size().reset_index(name="TourneyWins")
    )
    team_stats = team_stats.merge(
        tourney_wins,
        left_on=["Season", "TeamID"],
        right_on=["Season", "WTeamID"],
        how="left",
    )
    team_stats["TourneyWins"] = team_stats["TourneyWins"].fillna(0)

    plt.figure(figsize=(10, 6))
    plt.scatter(team_stats["WinPct"], team_stats["TourneyWins"], alpha=0.3, s=20)
    plt.xlabel("Regular Season Win %")
    plt.ylabel("Tournament Wins")
    plt.title("Regular Season Win % vs Tournament Success")

    z = np.polyfit(team_stats["WinPct"], team_stats["TourneyWins"], 1)
    p = np.poly1d(z)
    plt.plot(
        team_stats["WinPct"].sort_values(),
        p(team_stats["WinPct"].sort_values()),
        "r--",
        alpha=0.8,
        label=f"Trend line",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "winpct_vs_tourney.png", dpi=150)
    plt.close()
    print("Saved: winpct_vs_tourney.png")


def plot_key_metrics():
    """Plot distribution of key metrics for winners vs losers"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    m_reg = data_dict["m_reg_detailed"]
    games_df = prepare_winner_features(m_reg)

    metrics = [
        ("FGPct", "Field Goal %"),
        ("FG3Pct", "3-Point %"),
        ("FTPct", "Free Throw %"),
        ("Ast", "Assists"),
        ("TO", "Turnovers"),
        ("TotalReb", "Rebounds"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (col, title) in enumerate(metrics):
        winners = games_df[games_df["Win"] == 1][col]
        losers = games_df[games_df["Win"] == 0][col]

        axes[idx].hist(winners, bins=30, alpha=0.5, label="Winners", density=True)
        axes[idx].hist(losers, bins=30, alpha=0.5, label="Losers", density=True)
        axes[idx].set_title(title)
        axes[idx].legend()
        axes[idx].set_xlabel(col)

    plt.suptitle("Distribution of Key Metrics: Winners vs Losers", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "key_metrics_dist.png", dpi=150)
    plt.close()
    print("Saved: key_metrics_dist.png")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb


def prepare_ml_data():
    """Prepare data for ML model comparison"""
    m_reg = data_dict["m_reg_detailed"]
    m_seeds = data_dict["m_seeds"]
    m_tourney = data_dict["m_tourney_compact"]

    team_stats = compute_team_season_stats(m_reg, m_tourney)

    m_seeds["SeedNum"] = (
        m_seeds["Seed"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    )
    team_stats = team_stats.merge(
        m_seeds[["Season", "TeamID", "SeedNum"]],
        left_on=["Season", "TeamID"],
        right_on=["Season", "TeamID"],
        how="left",
    )
    team_stats["SeedNum"] = team_stats["SeedNum"].fillna(16)

    games = []
    for _, row in m_tourney.iterrows():
        w_stats = team_stats[
            (team_stats["Season"] == row["Season"])
            & (team_stats["TeamID"] == row["WTeamID"])
        ]
        l_stats = team_stats[
            (team_stats["Season"] == row["Season"])
            & (team_stats["TeamID"] == row["LTeamID"])
        ]

        if len(w_stats) == 0 or len(l_stats) == 0:
            continue

        w_stats = w_stats.iloc[0]
        l_stats = l_stats.iloc[0]

        game = {
            "Season": row["Season"],
            "WTeamID": row["WTeamID"],
            "LTeamID": row["LTeamID"],
            "Win": 1,
            "W_WinPct": w_stats["WinPct"],
            "L_WinPct": l_stats["WinPct"],
            "W_Seed": w_stats["SeedNum"],
            "L_Seed": l_stats["SeedNum"],
            "W_AvgPoints": w_stats["AvgPoints"],
            "L_AvgPoints": l_stats["AvgPoints"],
            "W_AvgFGPct": w_stats["AvgFGPct"],
            "L_AvgFGPct": l_stats["AvgFGPct"],
            "W_AvgAst": w_stats["AvgAst"],
            "L_AvgAst": l_stats["AvgAst"],
            "W_AvgTO": w_stats["AvgTO"],
            "L_AvgTO": l_stats["AvgTO"],
            "W_AvgStl": w_stats["AvgStl"],
            "L_AvgStl": l_stats["AvgStl"],
            "W_AvgBlk": w_stats["AvgBlk"],
            "L_AvgBlk": l_stats["AvgBlk"],
            "W_AvgOR": w_stats["AvgOR"],
            "L_AvgOR": l_stats["AvgOR"],
            "W_AvgDR": w_stats["AvgDR"],
            "L_AvgDR": l_stats["AvgDR"],
        }
        games.append(game)

        game_rev = dict(game)
        game_rev["Win"] = 0
        game_rev["WTeamID"], game_rev["LTeamID"] = game["LTeamID"], game["WTeamID"]
        game_rev["W_WinPct"], game_rev["L_WinPct"] = game["L_WinPct"], game["W_WinPct"]
        game_rev["W_Seed"], game_rev["L_Seed"] = game["L_Seed"], game["W_Seed"]
        game_rev["W_AvgPoints"], game_rev["L_AvgPoints"] = (
            game["L_AvgPoints"],
            game["W_AvgPoints"],
        )
        game_rev["W_AvgFGPct"], game_rev["L_AvgFGPct"] = (
            game["L_AvgFGPct"],
            game["W_AvgFGPct"],
        )
        game_rev["W_AvgAst"], game_rev["L_AvgAst"] = game["L_AvgAst"], game["W_AvgAst"]
        game_rev["W_AvgTO"], game_rev["L_AvgTO"] = game["L_AvgTO"], game["W_AvgTO"]
        game_rev["W_AvgStl"], game_rev["L_AvgStl"] = game["L_AvgStl"], game["W_AvgStl"]
        game_rev["W_AvgBlk"], game_rev["L_AvgBlk"] = game["L_AvgBlk"], game["W_AvgBlk"]
        game_rev["W_AvgOR"], game_rev["L_AvgOR"] = game["L_AvgOR"], game["W_AvgOR"]
        game_rev["W_AvgDR"], game_rev["L_AvgDR"] = game["L_AvgDR"], game["W_AvgDR"]
        games.append(game_rev)

    return pd.DataFrame(games)


def get_feature_sets():
    """Define different feature sets to test"""
    return {
        "Basic (Seed + Win%)": [
            "W_WinPct",
            "L_WinPct",
            "W_Seed",
            "L_Seed",
            "W_WinPct_L_WinPct",
            "W_Seed_L_Seed",
        ],
        "Scoring": ["W_AvgPoints", "L_AvgPoints", "W_AvgFGPct", "L_AvgFGPct"],
        "Ball Security": ["W_AvgAst", "L_AvgAst", "W_AvgTO", "L_AvgTO"],
        "Defense": [
            "W_AvgStl",
            "L_AvgStl",
            "W_AvgBlk",
            "L_AvgBlk",
            "W_AvgOR",
            "L_AvgOR",
            "W_AvgDR",
            "L_AvgDR",
        ],
        "All Stats": [
            "W_WinPct",
            "L_WinPct",
            "W_Seed",
            "L_Seed",
            "W_AvgPoints",
            "L_AvgPoints",
            "W_AvgFGPct",
            "L_AvgFGPct",
            "W_AvgAst",
            "L_AvgAst",
            "W_AvgTO",
            "L_AvgTO",
            "W_AvgStl",
            "L_AvgStl",
            "W_AvgBlk",
            "L_AvgBlk",
            "W_AvgOR",
            "L_AvgOR",
            "W_AvgDR",
            "L_AvgDR",
        ],
    }


def create_differential_features(df):
    """Create differential features (team A - team B)"""
    df = df.copy()
    df["W_WinPct_L_WinPct"] = df["W_WinPct"] - df["L_WinPct"]
    df["W_Seed_L_Seed"] = df["L_Seed"] - df["W_Seed"]
    df["W_AvgPoints_L_AvgPoints"] = df["W_AvgPoints"] - df["L_AvgPoints"]
    df["W_AvgFGPct_L_AvgFGPct"] = df["W_AvgFGPct"] - df["L_AvgFGPct"]
    df["W_AvgAst_L_AvgAst"] = df["W_AvgAst"] - df["L_AvgAst"]
    df["W_AvgTO_L_AvgTO"] = df["W_AvgTO"] - df["L_AvgTO"]
    df["W_AvgStl_L_AvgStl"] = df["W_AvgStl"] - df["L_AvgStl"]
    df["W_AvgBlk_L_AvgBlk"] = df["W_AvgBlk"] - df["L_AvgBlk"]
    df["W_AvgOR_L_AvgOR"] = df["W_AvgOR"] - df["L_AvgOR"]
    df["W_AvgDR_L_AvgDR"] = df["W_AvgDR"] - df["L_AvgDR"]
    return df


def plot_model_feature_comparison():
    """Compare model performance across different feature sets"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Preparing ML data...")
    df = prepare_ml_data()
    df = create_differential_features(df)

    feature_sets = get_feature_sets()
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    }

    results = []

    for feat_name, features in feature_sets.items():
        if feat_name == "Basic (Seed + Win%)":
            features = ["W_WinPct_L_WinPct", "W_Seed_L_Seed"]
        elif feat_name == "Scoring":
            features = ["W_AvgPoints_L_AvgPoints", "W_AvgFGPct_L_AvgFGPct"]
        elif feat_name == "Ball Security":
            features = ["W_AvgAst_L_AvgAst", "W_AvgTO_L_AvgTO"]
        elif feat_name == "Defense":
            features = [
                "W_AvgStl_L_AvgStl",
                "W_AvgBlk_L_AvgBlk",
                "W_AvgOR_L_AvgOR",
                "W_AvgDR_L_AvgDR",
            ]
        elif feat_name == "All Stats":
            features = [f for f in df.columns if "_L_" in f]

        X = df[features].fillna(0).values
        y = df["Win"].values

        for model_name, model in models.items():
            scores = []
            for train_season in range(2015, 2023):
                train_idx = df["Season"] < train_season
                test_idx = df["Season"] == train_season

                if test_idx.sum() == 0:
                    continue

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if len(set(y_train)) < 2:
                    continue

                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                scores.append(accuracy_score(y_test, pred))

            if scores:
                results.append(
                    {
                        "Feature Set": feat_name,
                        "Model": model_name,
                        "Accuracy": np.mean(scores),
                    }
                )

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(12, 8))
    pivot = results_df.pivot(index="Model", columns="Feature Set", values="Accuracy")
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=0.5)
    plt.title("Model Accuracy by Feature Set", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_feature_heatmap.png", dpi=150)
    plt.close()
    print("Saved: model_feature_heatmap.png")

    plt.figure(figsize=(12, 6))
    for model in results_df["Model"].unique():
        model_data = results_df[results_df["Model"] == model]
        plt.plot(
            model_data["Feature Set"], model_data["Accuracy"], marker="o", label=model
        )
    plt.xlabel("Feature Set")
    plt.ylabel("Accuracy")
    plt.title("Model Performance by Feature Set")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_feature_lines.png", dpi=150)
    plt.close()
    print("Saved: model_feature_lines.png")

    best_combos = results_df.loc[results_df.groupby("Model")["Accuracy"].idxmax()]
    plt.figure(figsize=(10, 6))
    plt.barh(best_combos["Model"], best_combos["Accuracy"])
    plt.xlabel("Best Accuracy")
    plt.title("Best Feature Set per Model")
    for i, (idx, row) in enumerate(best_combos.iterrows()):
        plt.text(
            row["Accuracy"] + 0.002, i, row["Feature Set"], va="center", fontsize=9
        )
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_features_per_model.png", dpi=150)
    plt.close()
    print("Saved: best_features_per_model.png")

    return results_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "ml":
        print("Running ML feature comparison...")
        plot_model_feature_comparison()
    else:
        print("Generating correlation plots...")
        plot_correlation_heatmaps()
        plot_seed_correlation()
        plot_win_pct_vs_tourney()
        plot_key_metrics()
    print("Done! Plots saved to plots/")
