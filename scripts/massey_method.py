import numpy as np
from collections import defaultdict

def massey_method(df, min_games=1):
    # Get all unique teams from both winners and losers
    teams = set(df["WTeamID"].unique()) | set(df["LTeamID"].unique())
    team_list = sorted(list(teams))

    # Create mapping from team ID to matrix index
    team_to_idx = {team: idx for idx, team in enumerate(team_list)}
    n_teams = len(team_list)

    # Extract game results: winner, loser, and point differential
    games = []
    for _, row in df.iterrows():
        winner = row["WTeamID"]
        loser = row["LTeamID"]
        w_score = row["WScore"]
        l_score = row["LScore"]
        point_diff = w_score - l_score
        games.append((winner, loser, point_diff))

    # Build the Massey matrix M and results vector p
    # M is an n_teams x n_teams matrix
    # p is the point differential vector
    M = np.zeros((n_teams, n_teams))
    p = np.zeros(n_teams)

    # Fill in the matrix based on game results
    # For each game, we add:
    #   M[winner, winner] += 1
    #   M[winner, loser] -= 1
    #   M[loser, loser] += 1
    #   M[loser, winner] -= 1
    #   p[winner] += point_diff
    #   p[loser] -= point_diff
    for winner, loser, point_diff in games:
        w_idx = team_to_idx[winner]
        l_idx = team_to_idx[loser]

        M[w_idx, w_idx] += 1
        M[w_idx, l_idx] -= 1
        M[l_idx, l_idx] += 1
        M[l_idx, w_idx] -= 1

        p[w_idx] += point_diff
        p[l_idx] -= point_diff

    # Add constraint to make the system solvable
    # The last row is all 1s and the last element of p is 0
    # This ensures the sum of all ratings equals 0 (centering constraint)
    M[-1, :] = 1
    p[-1] = 0

    # Solve the system M * r = p for the rating vector r
    try:
        ratings = np.linalg.solve(M, p)
    except np.linalg.LinAlgError:
        # Use least squares if matrix is singular
        ratings = np.linalg.lstsq(M, p, rcond=None)[0]

    # Convert ratings array to dictionary
    ratings_dict = {team_list[i]: ratings[i] for i in range(n_teams)}

    # Track each team's opponents for schedule strength calculation
    team_games = defaultdict(lambda: {"wins": 0, "losses": 0, "opponents": []})  # type: ignore
    for winner, loser, _ in games:
        team_games[winner]["wins"] += 1
        team_games[winner]["opponents"].append(loser)
        team_games[loser]["losses"] += 1
        team_games[loser]["opponents"].append(winner)

    # Calculate schedule of strength (SOS) for each team
    # SOS is the average rating of all opponents faced
    schedule_strength = {}
    for team in team_list:
        opponents = team_games[team]["opponents"]
        if len(opponents) >= min_games:
            sos = np.mean([ratings_dict[opp] for opp in opponents])
            schedule_strength[team] = sos
        else:
            schedule_strength[team] = np.nan

    return ratings_dict, schedule_strength, team_list



# Compute ratings and schedule strength
if __name__ == "__main__":
    print("Massey method module loaded. Run this script directly to test.")