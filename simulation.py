import pickle
import numpy as np
import pandas as pd
import os
from scipy.special import gammaln, logsumexp

with open('fifa_wc_2026.pkl', 'rb') as f:
    model_package = pickle.load(f)

ratings = model_package['ratings']
dibp_params = model_package['dibp_params']
MAX_SCORE_GRID = model_package.get('max_score_grid', 10)

ratings_lookup = (
    ratings
    .set_index('team')[['off_eff', 'def_eff']]
    .to_dict('index')
)

COMPLETED_MATCHES_FILE = "Completed Matches - Group Stage.csv"


def load_completed_matches():

    if not os.path.exists(COMPLETED_MATCHES_FILE):
        return {}

    df = pd.read_csv(COMPLETED_MATCHES_FILE)

    lookup = {}

    for _, row in df.iterrows():

        key = frozenset([
            str(row["Team A"]).strip(),
            str(row["Team B"]).strip()
        ])

        lookup[key] = {
            "team_a": str(row["Team A"]).strip(),
            "team_b": str(row["Team B"]).strip(),
            "team_a_goals": int(row["Team A Goals"]),
            "team_b_goals": int(row["Team B Goals"]),
            "group": row["Group"],
            "date": row["Date"]
        }

    return lookup


completed_matches_lookup = load_completed_matches()

def get_completed_match_result(team_a, team_b):

    key = frozenset([team_a, team_b])

    if key not in completed_matches_lookup:
        return None

    result = completed_matches_lookup[key]

    if result["team_a"] == team_a:

        return (
            result["team_a_goals"],
            result["team_b_goals"]
        )

    return (
        result["team_b_goals"],
        result["team_a_goals"]
    )


def log_dibp_pmf_single(x, y, lam1, lam2, lam3):
    min_xy = int(min(x, y))

    ks = np.arange(min_xy + 1)

    log_terms = (
        (x - ks) * np.log(lam1) - gammaln(x - ks + 1)
        + (y - ks) * np.log(lam2) - gammaln(y - ks + 1)
        + ks * np.log(lam3) - gammaln(ks + 1)
    )

    return -(lam1 + lam2 + lam3) + logsumexp(log_terms)


def dibp_pmf_single(x, y, lam1, lam2, lam3):
    return np.exp(log_dibp_pmf_single(x, y, lam1, lam2, lam3))


def inflated_pmf_single(x, y, lam1, lam2, lam3, phi, max_score=10):
    p_xy = dibp_pmf_single(x, y, lam1, lam2, lam3)

    z = 0.0

    for i in range(max_score + 1):
        for j in range(max_score + 1):
            p = dibp_pmf_single(i, j, lam1, lam2, lam3)
            z += p * (1.0 + (phi if i == j else 0.0))

    z = max(z, 1e-16)

    return max(
        p_xy * (1.0 + (phi if x == y else 0.0)) / z,
        1e-300
    )


def compute_lambdas_from_params(params, off_H, def_H, off_A, def_A, neutral_flags):

    alpha, b_off, b_def, delta, lam3_global, phi = params

    off_H = np.maximum(off_H, 1e-8)
    def_A = np.maximum(def_A, 1e-8)
    off_A = np.maximum(off_A, 1e-8)
    def_H = np.maximum(def_H, 1e-8)

    home_adv_array = np.where(neutral_flags, 0.0, 1.0)

    log_lam1 = (
        alpha
        + b_off * np.log(off_H)
        + b_def * np.log(def_A)
        + delta * home_adv_array
    )

    log_lam2 = (
        alpha
        + b_off * np.log(off_A)
        + b_def * np.log(def_H)
    )

    log_lam1 = np.clip(log_lam1, -8, 4)
    log_lam2 = np.clip(log_lam2, -8, 4)

    lam1 = np.exp(log_lam1)
    lam2 = np.exp(log_lam2)

    return lam1, lam2, lam3_global, phi

def score_probability_matrix(home_team, away_team, neutral=False, max_score=10):

    off_H = ratings_lookup[home_team]['off_eff']
    def_H = ratings_lookup[home_team]['def_eff']

    off_A = ratings_lookup[away_team]['off_eff']
    def_A = ratings_lookup[away_team]['def_eff']
    
    lam1, lam2, lam3, phi = compute_lambdas_from_params(
        dibp_params,
        np.array([off_H]),
        np.array([def_H]),
        np.array([off_A]),
        np.array([def_A]),
        np.array([neutral])
    )

    lam1 = lam1[0]
    lam2 = lam2[0]

    mat = np.zeros((max_score + 1, max_score + 1))

    for hG in range(max_score + 1):
        for aG in range(max_score + 1):
            mat[hG, aG] = dibp_pmf_single(
                hG,
                aG,
                lam1,
                lam2,
                lam3
            )

    for s in range(max_score + 1):
        mat[s, s] *= (1.0 + phi)

    mat = mat / mat.sum()

    return mat

matchup_cache = {}

def simulate_game_once(home_team, away_team, neutral=False, max_score=10):

    key = (home_team, away_team, neutral)

    if key not in matchup_cache:

        mat = score_probability_matrix(
            home_team=home_team,
            away_team=away_team,
            neutral=neutral,
            max_score=max_score
        )

        matchup_cache[key] = mat.ravel()

    flat_probs = matchup_cache[key]

    idx = np.random.choice(
        len(flat_probs),
        p=flat_probs
    )

    return np.unravel_index(
        idx,
        (max_score + 1, max_score + 1)
    )

import itertools

def get_head_to_head_points(results, tied_teams):
    """
    Returns a dict of head-to-head points earned
    among teams tied on total points.
    """

    h2h_points = {team: 0 for team in tied_teams}

    for match in results:

        home = match['home_team']
        away = match['away_team']

        if home not in tied_teams or away not in tied_teams:
            continue

        hG = match['home_score']
        aG = match['away_score']

        if hG > aG:
            h2h_points[home] += 3

        elif aG > hG:
            h2h_points[away] += 3

        else:
            h2h_points[home] += 1
            h2h_points[away] += 1

    return h2h_points

def simulate_round_robin_once(group_teams, host_team=None, max_score=10):
    if len(group_teams) != 4:
        raise ValueError("group_teams must contain exactly 4 teams.")

    missing_teams = [
        team for team in group_teams
        if team not in set(ratings['team'])
    ]

    if missing_teams:
        raise ValueError(f"These teams are missing from ratings: {missing_teams}")

    if host_team is not None and host_team not in group_teams:
        raise ValueError("host_team must be one of the group_teams.")

    table = {
        team: {
            'team': team,
            'played': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_for': 0,
            'goals_against': 0,
            'goal_diff': 0,
            'points': 0
        }
        for team in group_teams
    }

    results = []

    for team_a, team_b in itertools.combinations(group_teams, 2):

        if host_team is not None and team_a == host_team:
            home_team = team_a
            away_team = team_b
            neutral = False
        elif host_team is not None and team_b == host_team:
            home_team = team_b
            away_team = team_a
            neutral = False
        else:
            home_team = team_a
            away_team = team_b
            neutral = True

        completed_result = get_completed_match_result(
            home_team,
            away_team
        )
        
        if completed_result is not None:
        
            hG, aG = completed_result
        
        else:
        
            hG, aG = simulate_game_once(
                home_team=home_team,
                away_team=away_team,
                neutral=neutral,
                max_score=max_score
            )

        #print(f"{home_team} {hG}-{aG} {away_team}")

        results.append({
            'home_team': home_team,
            'away_team': away_team,
            'home_score': hG,
            'away_score': aG,
            'neutral': neutral
        })

        table[home_team]['played'] += 1
        table[away_team]['played'] += 1

        table[home_team]['goals_for'] += hG
        table[home_team]['goals_against'] += aG

        table[away_team]['goals_for'] += aG
        table[away_team]['goals_against'] += hG

        if hG > aG:
            table[home_team]['wins'] += 1
            table[away_team]['losses'] += 1
            table[home_team]['points'] += 3
        elif hG < aG:
            table[away_team]['wins'] += 1
            table[home_team]['losses'] += 1
            table[away_team]['points'] += 3
        else:
            table[home_team]['draws'] += 1
            table[away_team]['draws'] += 1
            table[home_team]['points'] += 1
            table[away_team]['points'] += 1

    for team in group_teams:
        table[team]['goal_diff'] = (
            table[team]['goals_for'] -
            table[team]['goals_against']
        )

    standings = pd.DataFrame(table.values())

    standings['h2h_points'] = 0

    for points_value in standings['points'].unique():
    
        tied_teams = standings.loc[
            standings['points'] == points_value,
            'team'
        ].tolist()
    
        if len(tied_teams) < 2:
            continue
    
        h2h = get_head_to_head_points(
            results,
            tied_teams
        )

        for team, pts in h2h.items():
            standings.loc[
                standings['team'] == team,
                'h2h_points'
            ] = pts

    standings = standings.sort_values(
        by=[
            'points',
            'h2h_points',
            'goal_diff',
            'goals_for'
        ],
        ascending=[
            False,
            False,
            False,
            False
        ]
    )

    standings['rank'] = standings.index + 1

    standings = standings[
        [
            'rank',
            'team',
            'played',
            'wins',
            'draws',
            'losses',
            'goals_for',
            'goals_against',
            'goal_diff',
            'points'
        ]
    ]

    sorted_teams = standings['team'].tolist()

    return standings

def simulate_group_finish_table(group, n_sims=1000, max_score=10):
    finish_counts = {
        team: {
            '1st': 0,
            '2nd': 0,
            '3rd': 0,
            '4th': 0
        }
        for team in group
    }

    hosts = ['United States', 'Mexico', 'Canada']

    host_team = next(
        (team for team in hosts if team in group),
        None
    )

    for _ in range(n_sims):
        sorted_teams = simulate_round_robin_once(
            group_teams=group,
            host_team=host_team,
            max_score=max_score
        )

        for rank, team in enumerate(sorted_teams, start=1):
            finish_counts[team][f'{rank}st' if rank == 1 else f'{rank}nd' if rank == 2 else f'{rank}rd' if rank == 3 else f'{rank}th'] += 1

    finish_table = pd.DataFrame.from_dict(
        finish_counts,
        orient='index'
    ).reset_index()

    finish_table = finish_table.rename(columns={'index': 'team'})

    finish_cols = ['1st', '2nd', '3rd', '4th']

    for col in finish_cols:
        finish_table[col] = finish_table[col] / n_sims * 100

    finish_table = finish_table.sort_values(
        by=['1st', '2nd', '3rd', '4th'],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    return finish_table

world_cup_groups = {
    'A': ['Mexico', 'Czech Republic', 'South Africa', 'South Korea'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Switzerland', 'Qatar'],
    'C': ['Brazil', 'Haiti', 'Morocco', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Ecuador', 'Curacao', 'Ivory Coast'],
    'F': ['Japan', 'Netherlands', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Uruguay', 'Saudi Arabia', 'Cape Verde'],
    'I': ['France', 'Norway', 'Senegal', 'Iraq'],
    'J': ['Argentina', 'Austria', 'Algeria', 'Jordan'],
    'K': ['Portugal', 'Colombia', 'DR Congo', 'Uzbekistan'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama']
}

def simulate_world_cup_groups_once(world_cup_groups):

    group_results = {}

    hosts = ['United States', 'Mexico', 'Canada']

    for group_name, group_teams in world_cup_groups.items():

        host_team = next(
            (team for team in hosts if team in group_teams),
            None
        )

        standings = simulate_round_robin_once(
            group_teams=group_teams,
            host_team=host_team
        )

        group_results[group_name] = standings

    return group_results

def extract_qualifiers(group_results):

    automatic = []
    third_place = []

    for group, standings in group_results.items():

        automatic.append({
            'group': group,
            'position': 1,
            'team': standings.iloc[0]['team']
        })

        automatic.append({
            'group': group,
            'position': 2,
            'team': standings.iloc[1]['team']
        })

        third_place.append({
            'group': group,
            'team': standings.iloc[2]['team'],
            'points': standings.iloc[2]['points'],
            'goal_diff': standings.iloc[2]['goal_diff'],
            'goals_for': standings.iloc[2]['goals_for']
        })

    third_place = pd.DataFrame(third_place)

    third_place = third_place.sort_values(
        ['points', 'goal_diff', 'goals_for'],
        ascending=False
    )

    best_8 = third_place.head(8)

    return automatic, best_8

group_results = simulate_world_cup_groups_once(world_cup_groups)
automatic, best_8 = extract_qualifiers(group_results)

import random

def simulate_knockout_match(a, b):
    if a in ("United States", "Mexico", "Canada"):
        aG, bG = simulate_game_once(a, b, neutral = False, max_score = 10)
    elif b in ("United States", "Mexico", "Canada"):
        bG, aG = simulate_game_once(b, a, neutral = False, max_score = 10)
    else:
        aG, bG = simulate_game_once(a, b, neutral = True, max_score = 10)

    return aG, bG

def simulate_knockout_round(teams):
    advance = []
    l = int(len(teams) / 2)
    for i in range(l):
        a = teams[i*2]
        b = teams[i*2 + 1]
    
        aG, bG = simulate_knockout_match(a, b)
        
        if aG > bG:
            advance.append(a)
        elif bG > aG:
            advance.append(b)
        else:
            aG, bG = simulate_game_once(a, b, neutral = True, max_score = 10)
            if (aG - bG) >= 2:
                advance.append(a)
            elif (bG - aG) >= 2:
                advance.append(b)
            else:
                advance.append(random.choice([a, b]))

    return advance

conditions = pd.read_csv("3rd place Conditions.csv")

def simulate_knockout_match_with_winner(a, b):

    aG, bG = simulate_knockout_match(a, b)

    if aG > bG:
        winner = a
    elif bG > aG:
        winner = b
    else:
        winner = random.choice([a, b])

    return aG, bG, winner


def simulate_world_cup_once(world_cup_groups, conditions):

    hosts = ["United States", "Mexico", "Canada"]

    # ----------------------------
    # Simulate all groups
    # ----------------------------

    group_results = {}

    for group_name, group_teams in world_cup_groups.items():

        host_team = next(
            (team for team in hosts if team in group_teams),
            None
        )

        standings = simulate_round_robin_once(
            group_teams=group_teams,
            host_team=host_team
        )

        group_results[group_name] = standings

    # ----------------------------
    # Store placements
    # ----------------------------

    placements = {}

    for group, standings in group_results.items():

        placements[f"{group}1"] = standings.iloc[0]["team"]
        placements[f"{group}2"] = standings.iloc[1]["team"]
        placements[f"{group}3"] = standings.iloc[2]["team"]
        placements[f"{group}4"] = standings.iloc[3]["team"]

    # ----------------------------
    # Rank third-place teams
    # ----------------------------

    third_place_rows = []

    for group, standings in group_results.items():

        third = standings.iloc[2]

        third_place_rows.append({
            "group": group,
            "team": third["team"],
            "points": third["points"],
            "goal_diff": third["goal_diff"],
            "goals_for": third["goals_for"]
        })

    third_place_df = pd.DataFrame(third_place_rows)

    qualified_thirds = (
        third_place_df
        .sort_values(
            ["points", "goal_diff", "goals_for"],
            ascending=False
        )
        .head(8)
        .reset_index(drop=True)
    )

    # ----------------------------
    # Identify FIFA scenario
    # ----------------------------

    scenario = "".join(
        sorted(qualified_thirds["group"].tolist())
    )

    condition_row = conditions.loc[
        conditions["Top 8 3rd place teams"] == scenario
    ]

    if len(condition_row) == 0:
        raise ValueError(
            f"No FIFA third-place scenario found for '{scenario}'"
        )

    condition_row = condition_row.iloc[0]

    # ----------------------------
    # Create lookup of advancing
    # third-place teams
    # ----------------------------

    third_lookup = {}

    for _, row in qualified_thirds.iterrows():

        third_lookup[f"3{row['group']}"] = row["team"]

    # ----------------------------
    # Assign opponents for
    # group winners
    # ----------------------------

    third_assignments = {}

    for winner_slot in [
        "1A",
        "1B",
        "1D",
        "1E",
        "1G",
        "1I",
        "1K",
        "1L"
    ]:

        third_assignments[winner_slot] = (
            third_lookup[condition_row[winner_slot]]
        )

    # ----------------------------
    # Build Round of 32 bracket
    # ----------------------------

    r32 = [

        placements["E1"],
        third_assignments["1E"],

        placements["I1"],
        third_assignments["1I"],

        placements["A2"],
        placements["B2"],

        placements["F1"],
        placements["C2"],

        placements["K2"],
        placements["L2"],

        placements["H1"],
        placements["J2"],

        placements["D1"],
        third_assignments["1D"],

        placements["G1"],
        third_assignments["1G"],

        placements["C1"],
        placements["F2"],

        placements["E2"],
        placements["I2"],

        placements["A1"],
        third_assignments["1A"],

        placements["L1"],
        third_assignments["1L"],

        placements["J1"],
        placements["H2"],

        placements["D2"],
        placements["G2"],

        placements["B1"],
        third_assignments["1B"],

        placements["K1"],
        third_assignments["1K"]
    ]
    #print(placements)
    return {
        "r32": r32,
        "group_results": group_results,
        "placements": placements,
        "qualified_thirds": qualified_thirds,
        "scenario": scenario
    }

def simulate_world_cup_many(n_sims=1000):

    teams = ratings["team"].tolist()

    stage_counts = pd.DataFrame(
        0,
        index=teams,
        columns=[
            "R32",
            "R16",
            "QF",
            "SF",
            "Final",
            "Champion"
        ]
    )

    for sim in range(n_sims):

        if sim % 100 == 0:
            print("Simulation", sim)

        wc = simulate_world_cup_once(
            world_cup_groups,
            conditions
        )

        r32 = wc["r32"]

        for team in r32:
            stage_counts.loc[team, "R32"] += 1

        teams_remaining = r32.copy()

        while len(teams_remaining) > 1:

            if len(teams_remaining) == 32:
                next_stage = "R16"
            elif len(teams_remaining) == 16:
                next_stage = "QF"
            elif len(teams_remaining) == 8:
                next_stage = "SF"
            elif len(teams_remaining) == 4:
                next_stage = "Final"
            else:
                next_stage = "Champion"

            winners = []

            for i in range(0, len(teams_remaining), 2):

                a = teams_remaining[i]
                b = teams_remaining[i + 1]

                _, _, winner = (
                    simulate_knockout_match_with_winner(a, b)
                )

                winners.append(winner)

            for team in winners:
                stage_counts.loc[team, next_stage] += 1

            teams_remaining = winners

    stage_counts = (
        stage_counts
        .div(n_sims)
        .mul(100)
        .round(1)
    )

    return (
        stage_counts
        .reset_index()
        .rename(columns={"index": "Team"})
        .sort_values("Champion", ascending=False)
    )

def simulate_group_stage_many(n_sims=10000):

    results = {}

    for group_name, teams in world_cup_groups.items():

        finish_counts = {
            team: {
                "1st": 0,
                "2nd": 0,
                "3rd": 0,
                "4th": 0
            }
            for team in teams
        }

        host_team = next(
            (
                t
                for t in [
                    "United States",
                    "Mexico",
                    "Canada"
                ]
                if t in teams
            ),
            None
        )

        for _ in range(n_sims):

            standings = simulate_round_robin_once(
                teams,
                host_team
            )

            for rank in range(4):

                team = standings.iloc[rank]["team"]

                finish_counts[team][
                    ["1st", "2nd", "3rd", "4th"][rank]
                ] += 1

        df = (
            pd.DataFrame.from_dict(
                finish_counts,
                orient="index"
            )
            .reset_index()
            .rename(columns={"index": "Team"})
        )

        for col in ["1st", "2nd", "3rd", "4th"]:
            df[col] = (
                df[col]
                / n_sims
                * 100
            ).round(1)

        df["Advance"] = (
            df["1st"]
            + df["2nd"]
        ).round(1)

        results[group_name] = df

    return results

def matchup_matrix(
    home_team,
    away_team,
    sims=10000
):

    counts = np.zeros((11,11))

    home_wins = 0
    draws = 0
    away_wins = 0

    neutral = not (
        home_team in [
            "United States",
            "Mexico",
            "Canada"
        ]
    )

    for _ in range(sims):

        hG, aG = simulate_game_once(
            home_team,
            away_team,
            neutral=neutral
        )

        counts[hG, aG] += 1

        if hG > aG:
            home_wins += 1
        elif hG < aG:
            away_wins += 1
        else:
            draws += 1

    matrix = (
        pd.DataFrame(
            counts / sims * 100
        )
        .round(2)
    )

    summary = pd.DataFrame({
        "Outcome": [
            f"{home_team} Win",
            "Draw",
            f"{away_team} Win"
        ],
        "Probability": [
            home_wins / sims * 100,
            draws / sims * 100,
            away_wins / sims * 100
        ]
    })

    return summary, matrix



