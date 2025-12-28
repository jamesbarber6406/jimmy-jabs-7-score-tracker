
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

import json
from datetime import datetime, timezone
import pandas as pd
import random
from contextlib import contextmanager



def now_utc_iso() -> str:
    """Return an ISO-8601 timestamp in UTC (timezone-aware)."""
    return datetime.now(timezone.utc).isoformat()

APP_TITLE = "Jimmy Jabs (Tournament Tracker)"
DEFAULT_PLAYERS = list("ABCDEFGHI")

EVENTS = ["Beer Pong", "Telestrations", "Spoons", "Secret Hitler", "Breathalyzer"]

# Beer Pong scoring
BEERPONG_WIN_PTS = 4
BEERPONG_LOSS_PTS = 2
BEERPONG_CLOSE_LOSS_PTS = 3

# Secret Hitler scoring
SH_WIN_PTS = 4
SH_LOSS_PTS = 2
SH_SPICY_BONUS = 1

# Breathalyzer scoring
BREATH_PTS = {1: 5, 2: 3, 3: 2}

# ----------------------------
# Secrets + DB (SQLite locally, Postgres in the cloud)
# ----------------------------
def safe_get_secret(key: str):
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        return None
    return None

class DB:
    """
    Lightweight DB wrapper:
    - Local: sqlite3 (default)
    - Cloud: Postgres if DATABASE_URL provided (psycopg2)
    """
    def __init__(self):
        self.database_url = safe_get_secret("DATABASE_URL")
        self.sqlite_path = safe_get_secret("SQLITE_PATH") or "jimmy_jabs.db"

        if self.database_url:
            import psycopg2
            self.kind = "postgres"
            self._driver = psycopg2
        else:
            import sqlite3
            self.kind = "sqlite"
            self._driver = sqlite3

    @contextmanager
    def connect(self):
        if self.kind == "postgres":
            conn = self._driver.connect(self.database_url)
            try:
                yield conn
            finally:
                conn.close()
        else:
            conn = self._driver.connect(self.sqlite_path, check_same_thread=False)
            try:
                yield conn
            finally:
                conn.close()

    def _q(self, sql: str) -> str:
        # Convert sqlite '?' placeholders to postgres '%s' when needed.
        if self.kind == "postgres":
            return sql.replace("?", "%s")
        return sql

    def execute(self, sql: str, params=None):
        tries = 2 if self.kind == "postgres" else 1
        last_err = None
        for _ in range(tries):
            try:
                with self.connect() as conn:
                    cur = conn.cursor()
                    cur.execute(self._q(sql), params or [])
                    conn.commit()
                return
            except Exception as e:
                # Postgres poolers can occasionally drop SSL connections; retry once.
                last_err = e
                if self.kind != "postgres" or e.__class__.__name__ not in ("OperationalError", "InterfaceError"):
                    raise
        raise last_err

    def fetchall(self, sql: str, params=None):
        tries = 2 if self.kind == "postgres" else 1
        last_err = None
        for _ in range(tries):
            try:
                with self.connect() as conn:
                    cur = conn.cursor()
                    cur.execute(self._q(sql), params or [])
                    rows = cur.fetchall()
                return rows
            except Exception as e:
                last_err = e
                if self.kind != "postgres" or e.__class__.__name__ not in ("OperationalError", "InterfaceError"):
                    raise
        raise last_err

    def fetchone(self, sql: str, params=None):
        tries = 2 if self.kind == "postgres" else 1
        last_err = None
        for _ in range(tries):
            try:
                with self.connect() as conn:
                    cur = conn.cursor()
                    cur.execute(self._q(sql), params or [])
                    row = cur.fetchone()
                return row
            except Exception as e:
                last_err = e
                if self.kind != "postgres" or e.__class__.__name__ not in ("OperationalError", "InterfaceError"):
                    raise
        raise last_err

db = DB()

def init_db():
    if db.kind == "postgres":
        db.execute("""
        CREATE TABLE IF NOT EXISTS players (
            letter TEXT PRIMARY KEY,
            name TEXT
        )""")
        db.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )""")
        db.execute("""
        CREATE TABLE IF NOT EXISTS event_results (
            id SERIAL PRIMARY KEY,
            event TEXT NOT NULL,
            round_no INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        db.execute("""
        CREATE TABLE IF NOT EXISTS beerpong_schedule (
            id SERIAL PRIMARY KEY,
            round_no INTEGER NOT NULL,
            payload_json TEXT NOT NULL
        )""")
        db.execute("""
        CREATE TABLE IF NOT EXISTS adjustments (
            id SERIAL PRIMARY KEY,
            event TEXT NOT NULL,
            player TEXT NOT NULL,
            delta INTEGER NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL
        )""")
    else:
        db.execute("""
        CREATE TABLE IF NOT EXISTS players (
            letter TEXT PRIMARY KEY,
            name TEXT
        )""")
        db.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )""")
        db.execute("""
        CREATE TABLE IF NOT EXISTS event_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT NOT NULL,
            round_no INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        db.execute("""
        CREATE TABLE IF NOT EXISTS beerpong_schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            round_no INTEGER NOT NULL,
            payload_json TEXT NOT NULL
        )""")
        db.execute("""
        CREATE TABLE IF NOT EXISTS adjustments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT NOT NULL,
            player TEXT NOT NULL,
            delta INTEGER NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL
        )""")

def seed_default_players():
    row = db.fetchone("SELECT COUNT(*) FROM players")
    n = int(row[0]) if row else 0
    if n == 0:
        for letter in DEFAULT_PLAYERS:
            db.execute("INSERT INTO players(letter, name) VALUES(?, ?)", [letter, ""])

def upsert_setting(key: str, value: str):
    if db.kind == "postgres":
        db.execute("""
        INSERT INTO settings(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value
        """, [key, value])
    else:
        db.execute("""
        INSERT INTO settings(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, [key, value])

def get_setting(key: str, default=None):
    row = db.fetchone("SELECT value FROM settings WHERE key=?", [key])
    return row[0] if row else default

def set_event_completed(event: str, completed: bool):
    upsert_setting(f"event_completed::{event}", "1" if completed else "0")

def is_event_completed(event: str) -> bool:
    return get_setting(f"event_completed::{event}", "0") == "1"

def set_event_locked(event: str, locked: bool):
    upsert_setting(f"event_locked::{event}", "1" if locked else "0")

def is_event_locked(event: str) -> bool:
    return get_setting(f"event_locked::{event}", "0") == "1"

def insert_event_result(event: str, round_no: int, payload: dict):
    db.execute(
        "INSERT INTO event_results(event, round_no, payload_json, created_at) VALUES(?, ?, ?, ?)",
        [event, int(round_no), json.dumps(payload), now_utc_iso()]
    )

def fetch_event_results(event: str):
    try:
        rows = db.fetchall(
            "SELECT id, round_no, payload_json, created_at FROM event_results WHERE event=? ORDER BY id ASC",
            [event],
        )
    except Exception as e:
        # Avoid hard-crashing the whole app on transient DB/network issues.
        if e.__class__.__name__ in ("OperationalError", "InterfaceError"):
            st.error(f"Database connection error while loading {event} results. Please retry / refresh. ({e})")
            return []
        raise
    out = []
    for rid, round_no, payload_json, created_at in rows:
        out.append({"id": rid, "round_no": round_no, "payload": json.loads(payload_json), "created_at": created_at})
    return out

def delete_result(result_id: int):
    db.execute("DELETE FROM event_results WHERE id=?", [int(result_id)])

def clear_schedule():
    db.execute("DELETE FROM beerpong_schedule")

def store_schedule(round_no: int, payload: dict):
    db.execute("INSERT INTO beerpong_schedule(round_no, payload_json) VALUES(?, ?)", [int(round_no), json.dumps(payload)])

def load_schedule():
    rows = db.fetchall("SELECT round_no, payload_json FROM beerpong_schedule ORDER BY round_no ASC")
    schedule = {}
    for r, p in rows:
        schedule[int(r)] = json.loads(p)
    return schedule

def get_players_df():
    rows = db.fetchall("SELECT letter, name FROM players ORDER BY letter ASC")
    return pd.DataFrame(rows, columns=["letter", "name"])

def add_adjustment(event: str, player: str, delta: int, note: str):
    db.execute(
        "INSERT INTO adjustments(event, player, delta, note, created_at) VALUES(?, ?, ?, ?, ?)",
        [event, player, int(delta), note, now_utc_iso()]
    )

def list_adjustments(event: str | None = None):
    if event:
        rows = db.fetchall("SELECT id, event, player, delta, note, created_at FROM adjustments WHERE event=? ORDER BY id ASC", [event])
    else:
        rows = db.fetchall("SELECT id, event, player, delta, note, created_at FROM adjustments ORDER BY id ASC")
    return pd.DataFrame(rows, columns=["id", "event", "player", "delta", "note", "created_at"])

def delete_adjustment(adj_id: int):
    db.execute("DELETE FROM adjustments WHERE id=?", [int(adj_id)])

def adjustments_sum_by_player(event: str, players: list[str]):
    df = list_adjustments(event)
    out = {p: 0 for p in players}
    if df.empty:
        return out
    for _, r in df.iterrows():
        if r["player"] in out and r["event"] == event:
            out[r["player"]] += int(r["delta"])
    return out

def reset_tournament(clear_names: bool):
    db.execute("DELETE FROM event_results")
    db.execute("DELETE FROM beerpong_schedule")
    db.execute("DELETE FROM adjustments")
    # clear completion + locks
    db.execute("DELETE FROM settings WHERE key LIKE 'event_completed::%%'")
    db.execute("DELETE FROM settings WHERE key LIKE 'event_locked::%%'")
    if clear_names:
        db.execute("UPDATE players SET name=''")

# ----------------------------
# Display helpers
# ----------------------------
def display_player(letter: str, name_map: dict) -> str:
    nm = (name_map.get(letter) or "").strip()
    return f"{nm} ({letter})" if nm else letter

def display_team(team, name_map):
    return [display_player(p, name_map) for p in team]

# ----------------------------
# Tie-aware helpers
# ----------------------------
def assign_points_by_place(groups, place_to_points, default_points=None, no_skipping=True):
    pts = {}
    place = 1
    for grp in groups:
        p = place_to_points.get(place, default_points)
        for player in grp:
            pts[player] = p
        place = place + 1 if no_skipping else place + len(grp)
    return pts


# ----------------------------
# Tie groups + JJ skipped placements (no letter fallback)
# ----------------------------
def build_tie_groups(players, primary, tiebreakers):
    """Return tie groups ordered best->worst using only provided criteria."""
    def key(p):
        parts = [primary.get(p, 0)]
        for d, higher_is_better in tiebreakers:
            v = d.get(p, 0)
            parts.append(v if higher_is_better else -v)
        return tuple(parts)

    ordered = sorted(players, key=key, reverse=True)
    groups = []
    i = 0
    while i < len(ordered):
        k = key(ordered[i])
        grp = [ordered[i]]
        i += 1
        while i < len(ordered) and key(ordered[i]) == k:
            grp.append(ordered[i])
            i += 1
        groups.append(grp)
    return groups

def jj_points_skipping_from_groups(groups, n_players):
    """Two tie for 1st => both get 9, next gets 7 (skipped placements)."""
    pts = {p: 0 for grp in groups for p in grp}
    place = 1
    for grp in groups:
        jj_for_place = n_players - (place - 1)
        for p in grp:
            pts[p] = jj_for_place
        place += len(grp)
    return pts

# ----------------------------
# Beer Pong schedule generator (equal games)
# ----------------------------
def score_schedule(rounds_payload, players):
    teammate = {p: {q: 0 for q in players} for p in players}
    opponent = {p: {q: 0 for q in players} for p in players}
    games = {p: 0 for p in players}
    byes = {p: 0 for p in players}

    for r in rounds_payload:
        byes[r["bye"]] += 1
        for m in r["matches"]:
            a = m["team_a"]
            b = m["team_b"]
            for p in a + b:
                games[p] += 1
            teammate[a[0]][a[1]] += 1
            teammate[a[1]][a[0]] += 1
            teammate[b[0]][b[1]] += 1
            teammate[b[1]][b[0]] += 1
            for p in a:
                for q in b:
                    opponent[p][q] += 1
                    opponent[q][p] += 1

    rep_team_pen = 0
    rep_opp_pen = 0
    for i, p in enumerate(players):
        for q in players[i+1:]:
            c = teammate[p][q]
            rep_team_pen += (c - 1) * 12 if c > 1 else (2 if c == 1 else 0)
            c2 = opponent[p][q]
            rep_opp_pen += (c2 - 2) * 4 if c2 > 2 else (1 if c2 == 2 else 0)

    gp = list(games.values())
    mean_gp = sum(gp) / len(gp)
    var_gp = sum((x - mean_gp) ** 2 for x in gp) / len(gp)

    bp = list(byes.values())
    mean_bp = sum(bp) / len(bp)
    var_bp = sum((x - mean_bp) ** 2 for x in bp) / len(bp)

    return rep_team_pen + rep_opp_pen + (var_gp * 10) + (var_bp * 10)


def generate_equal_beerpong_schedule(players, rounds=5, games_per_player=4, tries=6000, seed=42):
    """Generate a 2v2 Beer Pong schedule.

    Goal: each player plays exactly `games_per_player` matches.
    With 9 players and games_per_player=4 -> total matches = 9.
    We then pack those matches into `rounds` rounds (default 5), with up to 2 matches per round.
    """
    rng = random.Random(seed)
    players = list(players)

    total_slots = len(players) * games_per_player
    if total_slots % 4 != 0:
        raise ValueError("games_per_player * n_players must be divisible by 4 for 2v2 matches.")
    total_matches = total_slots // 4

    # Helpers to score schedule quality: balance teammates/opponents and bye distribution across rounds.
    def schedule_score(matches):
        teammate = {p: {q: 0 for q in players} for p in players}
        opp = {p: {q: 0 for q in players} for p in players}
        for (a1,a2,b1,b2) in matches:
            A = [a1,a2]; B=[b1,b2]
            teammate[a1][a2] += 1; teammate[a2][a1] += 1
            for x in A:
                for y in B:
                    opp[x][y] += 1
                    opp[y][x] += 1
        # Penalize repeated teammates heavily, repeated opponents lightly
        rep_team = sum(max(0, teammate[p][q]-1) for p in players for q in players if p<q)
        rep_opp = sum(max(0, opp[p][q]-2) for p in players for q in players if p<q)
        return rep_team*5 + rep_opp*1

    best_matches, best_score = None, float("inf")

    for _ in range(tries):
        remaining = {p: games_per_player for p in players}
        matches = []

        # Build matches greedily but randomized
        while len(matches) < total_matches:
            # pick 4 players with remaining games, prefer those with highest remaining
            pool = [p for p in players if remaining[p] > 0]
            if len(pool) < 4:
                break
            pool.sort(key=lambda p: (remaining[p], rng.random()), reverse=True)
            pick = pool[:min(len(pool), 7)]
            four = rng.sample(pick, 4)

            # try a few team splits and keep the one with minimal local repeat teammates
            best_split = None
            best_local = None
            a,b,c,d = four
            splits = [((a,b),(c,d)), ((a,c),(b,d)), ((a,d),(b,c))]
            rng.shuffle(splits)
            for (ta,tb) in splits:
                # local repeat measure: avoid same teammate pairs in this candidate set
                local = 0
                for (x,y) in [(ta[0],ta[1]), (tb[0],tb[1])]:
                    for (u,v,w,z) in matches[-6:]:
                        if set([x,y])==set([u,v]) or set([x,y])==set([w,z]):
                            local += 2
                if best_local is None or local < best_local:
                    best_local = local
                    best_split = (ta, tb)

            (ta, tb) = best_split
            (a1,a2) = ta; (b1,b2) = tb
            # ensure all have remaining
            if min(remaining[a1],remaining[a2],remaining[b1],remaining[b2]) <= 0:
                continue
            matches.append((a1,a2,b1,b2))
            for p in [a1,a2,b1,b2]:
                remaining[p] -= 1

        if len(matches) != total_matches:
            continue

        score = schedule_score(matches)
        if score < best_score:
            best_score = score
            best_matches = matches

    if best_matches is None:
        raise ValueError("Could not generate a schedule with the requested constraints.")

    # Pack matches into rounds (up to 2 matches per round) without player overlap within a round.
    rounds_list = [[] for _ in range(rounds)]
    used_in_round = [set() for _ in range(rounds)]
    # shuffle to pack varied
    rng.shuffle(best_matches)
    for match in best_matches:
        a1,a2,b1,b2 = match
        players_in = {a1,a2,b1,b2}
        placed = False
        for r in range(rounds):
            if len(rounds_list[r]) >= 2:
                continue
            if used_in_round[r].isdisjoint(players_in):
                rounds_list[r].append(match)
                used_in_round[r].update(players_in)
                placed = True
                break
        if not placed:
            # if we can't place without overlap, just place in the round with space (overlap allowed as last resort)
            for r in range(rounds):
                if len(rounds_list[r]) < 2:
                    rounds_list[r].append(match)
                    used_in_round[r].update(players_in)
                    placed = True
                    break
        if not placed:
            # shouldn't happen
            rounds_list.append([match])
            used_in_round.append(set(players_in))

    # Build stored schedule format expected by UI: each round has 2 match slots; if missing, use None.
    schedule = []
    for i, matches in enumerate(rounds_list, start=1):
        entry = {"round_no": i, "matches": []}
        for mm in matches:
            a1,a2,b1,b2 = mm
            entry["matches"].append({"team_a": [a1,a2], "team_b": [b1,b2]})
        while len(entry["matches"]) < 2:
            entry["matches"].append(None)
        schedule.append(entry)

    return schedule, best_score


def compute_beerpong_raw(players):
    results = fetch_event_results("Beer Pong")
    raw = {p: 0 for p in players}
    wins = {p: 0 for p in players}
    games = {p: 0 for p in players}
    cups_sunk = {p: 0 for p in players}

    for r in results:
        pl = r["payload"]
        team_a = pl["team_a"]
        team_b = pl["team_b"]
        winner = pl["winner"]
        close = pl.get("close_loss", False)

        a_rem = int(pl.get("cups_remaining_a", 0))
        b_rem = int(pl.get("cups_remaining_b", 0))
        a_sunk = 6 - a_rem
        b_sunk = 6 - b_rem

        win_team = team_a if winner == "A" else team_b
        lose_team = team_b if winner == "A" else team_a

        for p in team_a + team_b:
            games[p] += 1
        for p in team_a:
            cups_sunk[p] += a_sunk
        for p in team_b:
            cups_sunk[p] += b_sunk

        for p in win_team:
            raw[p] += BEERPONG_WIN_PTS
            wins[p] += 1
        for p in lose_team:
            raw[p] += (BEERPONG_CLOSE_LOSS_PTS if close else BEERPONG_LOSS_PTS)

    # manual adjustments apply to raw
    adj = adjustments_sum_by_player("Beer Pong", players)
    for p in players:
        raw[p] += adj[p]

    return raw, wins, games, cups_sunk, adj

def compute_telestrations_raw(players):
    results = fetch_event_results("Telestrations")
    raw = {p: 0 for p in players}
    booklet_wins = {p: 0 for p in players}
    response_pts = {p: 0 for p in players}

    for r in results:
        pl = r["payload"]
        winners = pl.get("booklet_winners", [])
        responses = pl["response_points"]
        for w in winners:
            booklet_wins[w] += 1
        for p in players:
            rp = int(responses.get(p, 0))
            response_pts[p] += rp
            raw[p] += rp + (10 if p in winners else 0)

    adj = adjustments_sum_by_player("Telestrations", players)
    for p in players:
        raw[p] += adj[p]

    return raw, booklet_wins, response_pts, adj

def compute_spoons_raw(players):
    results = fetch_event_results("Spoons")
    raw = {p: 0 for p in players}
    for r in results:
        order = r["payload"]["elimination_order"]
        for idx, p in enumerate(order):
            raw[p] += (idx + 1)

    adj = adjustments_sum_by_player("Spoons", players)
    for p in players:
        raw[p] += adj[p]

    return raw, adj


def compute_spoons_countback(players):
    """
    Countback tie-breaker for Spoons:
    More 1st-place finishes wins; if tied, more 2nd-place finishes; etc.
    We represent each player\'s countback as a tuple: (c1, c2, ..., c9)
    """
    results = fetch_event_results("Spoons")
    counts = {p: [0]*len(players) for p in players}  # index 0 => 1st place
    for r in results:
        elim = r["payload"]["elimination_order"]
        placement = list(reversed(elim))  # placement[0] is 1st place
        for place_idx, p in enumerate(placement):
            counts[p][place_idx] += 1
    return {p: tuple(counts[p]) for p in players}

def compute_secret_hitler_raw(players):
    results = fetch_event_results("Secret Hitler")
    raw = {p: 0 for p in players}
    wins = {p: 0 for p in players}
    spicy_wins = {p: 0 for p in players}

    for r in results:
        pl = r["payload"]
        participants = pl.get("participants") or players
        fascists = pl.get("fascists") or []
        liberals = [p for p in participants if p not in fascists]
        win_side = pl["winner_side"]
        spicy_type = pl.get("spicy_type", "None")

        if win_side == "Fascists":
            winners_side = set(fascists)
            losers_side = set(liberals)
            spicy_applies = (spicy_type == "Hitler elected")
        else:
            winners_side = set(liberals)
            losers_side = set(fascists)
            spicy_applies = (spicy_type == "Hitler killed")

        for p in participants:
            if p in winners_side:
                raw[p] += SH_WIN_PTS + (SH_SPICY_BONUS if spicy_applies else 0)
                wins[p] += 1
                spicy_wins[p] += (1 if spicy_applies else 0)
            elif p in losers_side:
                raw[p] += SH_LOSS_PTS

    adj = adjustments_sum_by_player("Secret Hitler", players)
    for p in players:
        raw[p] += adj[p]

    return raw, wins, spicy_wins, adj

def compute_breathalyzer_raw(players):
    results = fetch_event_results("Breathalyzer")
    raw = {p: 0 for p in players}
    closest_count = {p: 0 for p in players}
    avg_error_sum = {p: 0.0 for p in players}
    err_n = {p: 0 for p in players}

    for r in results:
        pl = r["payload"]
        actual = float(pl["actual"])
        guesses = pl["guesses"]
        errors = {p: abs(float(g) - actual) for p, g in guesses.items()}

        items = sorted(errors.items(), key=lambda x: (x[1], x[0]))
        groups = []
        i = 0
        while i < len(items):
            v = items[i][1]
            grp = [items[i][0]]
            i += 1
            while i < len(items) and items[i][1] == v:
                grp.append(items[i][0])
                i += 1
            groups.append(grp)

        pts = assign_points_by_place(groups, BREATH_PTS, default_points=1, no_skipping=True)
        for p, pt in pts.items():
            raw[p] += int(pt)

        if groups:
            for p in groups[0]:
                closest_count[p] += 1

        for p, e in errors.items():
            avg_error_sum[p] += e
            err_n[p] += 1

    avg_error = {p: (avg_error_sum[p] / err_n[p]) if err_n[p] else float("inf") for p in players}

    adj = adjustments_sum_by_player("Breathalyzer", players)
    for p in players:
        raw[p] += adj[p]

    return raw, closest_count, avg_error, adj

def compute_all(players):
    # Beer Pong tie-breakers: points -> wins -> cups sunk -> cups remaining total -> tie
    bp_raw, bp_wins, bp_games, bp_cups, bp_adj = compute_beerpong_raw(players)
    # In v4 beer pong compute returns cups_sunk only; compute cups_remaining_total from saved payload fields
    # We'll reconstruct cups_remaining_total here for tiebreaking.
    bp_results = fetch_event_results("Beer Pong")
    bp_rem = {p: 0 for p in players}
    for r in bp_results:
        pl = r["payload"]
        a_rem = int(pl.get("cups_remaining_a", 0))
        b_rem = int(pl.get("cups_remaining_b", 0))
        for p in pl["team_a"]:
            bp_rem[p] += a_rem
        for p in pl["team_b"]:
            bp_rem[p] += b_rem

    bp_groups = build_tie_groups(players, bp_raw, [(bp_wins, True), (bp_cups, True), (bp_rem, True)])
    bp_jj = jj_points_skipping_from_groups(bp_groups, len(players))

    # Telestrations: points -> booklet wins -> tie
    tel_raw, tel_bw, tel_rp, tel_adj = compute_telestrations_raw(players)
    tel_groups = build_tie_groups(players, tel_raw, [(tel_bw, True)])
    tel_jj = jj_points_skipping_from_groups(tel_groups, len(players))

    # Spoons: points -> countback (more 1sts, then 2nds, ...) -> tie
    sp_raw, sp_adj = compute_spoons_raw(players)
    sp_cb = compute_spoons_countback(players)
    sp_groups = build_tie_groups(players, sp_raw, [(sp_cb, True)])
    sp_jj = jj_points_skipping_from_groups(sp_groups, len(players))

    # Secret Hitler: points -> wins -> spicy wins -> tie
    sh_raw, sh_wins, sh_spicy, sh_adj = compute_secret_hitler_raw(players)
    sh_groups = build_tie_groups(players, sh_raw, [(sh_wins, True), (sh_spicy, True)])
    sh_jj = jj_points_skipping_from_groups(sh_groups, len(players))

    # Breathalyzer: points -> closest count -> lower avg error -> tie
    br_raw, br_closest, br_avgerr, br_adj = compute_breathalyzer_raw(players)
    br_groups = build_tie_groups(players, br_raw, [(br_closest, True), (br_avgerr, False)])
    br_jj = jj_points_skipping_from_groups(br_groups, len(players))

    computed = {
        "Beer Pong": {"raw": bp_raw, "groups": bp_groups, "jj": bp_jj, "wins": bp_wins, "cups": bp_cups, "cups_rem": bp_rem, "adj": bp_adj},
        "Telestrations": {"raw": tel_raw, "groups": tel_groups, "jj": tel_jj, "booklet_wins": tel_bw, "response_points": tel_rp, "adj": tel_adj},
        "Spoons": {"raw": sp_raw, "groups": sp_groups, "jj": sp_jj, "adj": sp_adj},
        "Secret Hitler": {"raw": sh_raw, "groups": sh_groups, "jj": sh_jj, "wins": sh_wins, "spicy_wins": sh_spicy, "adj": sh_adj},
        "Breathalyzer": {"raw": br_raw, "groups": br_groups, "jj": br_jj, "closest": br_closest, "avg_error": br_avgerr, "adj": br_adj},
    }

    # JJ points only count if event completed
    for ev in EVENTS:
        if not is_event_completed(ev):
            computed[ev]["jj"] = {p: 0 for p in players}

    return computed


# ----------------------------
# Admin auth (PIN)
# ----------------------------
def is_admin() -> bool:
    pin = safe_get_secret("ADMIN_PIN")
    if not pin:
        # No pin set => treat as "no admin features"
        return False
    return st.session_state.get("admin_ok", False)

def admin_login_ui():
    st.markdown("### Admin")
    pin = safe_get_secret("ADMIN_PIN")
    if not pin:
        st.warning("ADMIN_PIN is not set in Streamlit secrets. Admin features are disabled.")
        return
    if is_admin():
        st.success("Admin unlocked for this session.")
        if st.button("Lock admin"):
            st.session_state["admin_ok"] = False
            st.rerun()
        return
    entered = st.text_input("Enter admin PIN", type="password")
    if st.button("Unlock admin"):
        if entered == str(pin):
            st.session_state["admin_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect PIN.")

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

init_db()
seed_default_players()

def refresh_players():
    pdf = get_players_df()
    p_list = pdf["letter"].tolist()
    nmap = {row["letter"]: row["name"] for _, row in pdf.iterrows()}
    return pdf, p_list, nmap

players_df, players, name_map = refresh_players()

tabs = st.tabs(["Setup", "Beer Pong", "Telestrations", "Spoons", "Secret Hitler", "Breathalyzer", "Standings", "Admin"])

with tabs[0]:
    st.subheader("Setup")
    st.caption(f"Database backend: **{db.kind}**")
    edited = st.data_editor(players_df, width="stretch", num_rows="fixed")
    if st.button("Save player names"):
        for _, row in edited.iterrows():
            db.execute("UPDATE players SET name=? WHERE letter=?", [row["name"], row["letter"]])
        st.success("Saved.")
        # refresh name_map everywhere immediately
        players_df, players, name_map = refresh_players()
        st.rerun()

    st.divider()
    st.markdown("### Event status")
    cols = st.columns(5)
    for i, ev in enumerate(EVENTS):
        with cols[i]:
            done = is_event_completed(ev)
            locked = is_event_locked(ev)
            new_done = st.checkbox(f"{ev} completed", value=done, key=f"done_{ev}")
            new_locked = st.checkbox(f"{ev} locked", value=locked, key=f"lock_{ev}", disabled=not is_admin())
            if new_done != done:
                set_event_completed(ev, new_done)
            if new_locked != locked:
                set_event_locked(ev, new_locked)

    if not is_admin():
        st.info("Tip: Locking events requires Admin PIN (Admin tab).")

# --- Beer Pong ---
with tabs[1]:
    st.subheader("Beer Pong")
    if is_event_locked("Beer Pong"):
        st.warning("Beer Pong is locked. Admin can unlock in Setup.")
    schedule = load_schedule()

    c1, c2 = st.columns([1, 1])
    with c1:
        st.write("Rounds: 5 (target: everyone plays 4 games; total 9 matches)")
        seed = st.number_input(
            "Schedule seed",
            min_value=1,
            step=1,
            value=int(get_setting("beerpong_seed", "42")),
            disabled=is_event_locked("Beer Pong"),
        )
        tries = st.number_input(
            "Generator tries",
            min_value=500,
            step=500,
            value=int(get_setting("beerpong_tries", "6000")),
            disabled=is_event_locked("Beer Pong"),
        )
        if st.button("Generate schedule", disabled=is_event_locked("Beer Pong")):
            best, best_score = generate_equal_beerpong_schedule(
                players, rounds=5, games_per_player=4, tries=int(tries), seed=int(seed)
            )
            clear_schedule()
            for r in best:
                store_schedule(r["round_no"], r)
            upsert_setting("beerpong_rounds", "5")
            upsert_setting("beerpong_seed", str(int(seed)))
            upsert_setting("beerpong_tries", str(int(tries)))
            st.success(f"Generated. Score (lower better): {best_score:.2f}")
            schedule = load_schedule()

    with c2:
        if schedule:
            preview = []
            for rno in sorted(schedule.keys()):
                r = schedule[rno]
                matches = r.get("matches", []) or []
                m1 = matches[0] if len(matches) > 0 else None
                m2 = matches[1] if len(matches) > 1 else None

                preview.append({
                    "Round": rno,
                    "Bye": (display_player(r["bye"], name_map) if r.get("bye") else "—"),
                    "Match 1": (f'{display_team(m1["team_a"], name_map)} vs {display_team(m1["team_b"], name_map)}' if m1 else "—"),
                    "Match 2": (f'{display_team(m2["team_a"], name_map)} vs {display_team(m2["team_b"], name_map)}' if m2 else "—"),
                })
            st.dataframe(pd.DataFrame(preview), width="stretch")
        else:
            st.info("No schedule yet. Generate one.")

    st.divider()
    if schedule:
        st.markdown("### Log a match")
        rno = st.selectbox("Round to log", sorted(schedule.keys()))
        r = schedule[rno]

        # Only show matches that exist (some rounds may have 1 match)
        available = [i + 1 for i, mm in enumerate(r["matches"]) if mm is not None]
        if not available:
            st.info("No matches scheduled for this round.")
        else:
            match_idx = st.radio("Match", available, horizontal=True)
            m = r["matches"][match_idx - 1]

            st.write(
                "Team A:", display_team(m["team_a"], name_map),
                "vs Team B:", display_team(m["team_b"], name_map)
            )

            with st.form(f"bp_log_form_{rno}_{match_idx}", clear_on_submit=True):
                entered_by = st.selectbox(
                    "Entered by",
                    [""] + players,
                    format_func=lambda p: "—" if p == "" else display_player(p, name_map),
                    disabled=is_event_locked("Beer Pong"),
                )

                winner = st.radio("Winner", ["A", "B"], horizontal=True, disabled=is_event_locked("Beer Pong"))
                a_rem = st.number_input(
                    "Cups remaining (Team A)", min_value=0, max_value=6, value=0, step=1,
                    disabled=is_event_locked("Beer Pong")
                )
                b_rem = st.number_input(
                    "Cups remaining (Team B)", min_value=0, max_value=6, value=0, step=1,
                    disabled=is_event_locked("Beer Pong")
                )
                suggested_close = (winner == "A" and a_rem == 0 and b_rem == 1) or (winner == "B" and b_rem == 0 and a_rem == 1)
                close_loss = st.checkbox(
                    "Close loss (both teams were at 1 cup)", value=suggested_close,
                    disabled=is_event_locked("Beer Pong")
                )

                if st.form_submit_button("Save match result", disabled=is_event_locked("Beer Pong")):
                    insert_event_result("Beer Pong", int(rno), {
                        "round_no": int(rno),
                        "match_idx": int(match_idx),
                        "team_a": m["team_a"],
                        "team_b": m["team_b"],
                        "winner": winner,
                        "cups_remaining_a": int(a_rem),
                        "cups_remaining_b": int(b_rem),
                        "close_loss": bool(close_loss),
                        "entered_by": (entered_by if entered_by else None),
                    })
                    st.success("Saved.")
                    st.rerun()
    st.divider()
    st.markdown("### Logged matches")
    bp_results = fetch_event_results("Beer Pong")
    if bp_results:
        rows = []
        for rr in bp_results:
            pl = rr["payload"]
            rows.append({
                "id": rr["id"],
                "Round": rr["round_no"],
                "Match": pl["match_idx"],
                "Team A": str(display_team(pl["team_a"], name_map)),
                "Team B": str(display_team(pl["team_b"], name_map)),
                "Winner": pl["winner"],
                "A sunk": 6 - int(pl.get("cups_remaining_a", 0)),
                "B sunk": 6 - int(pl.get("cups_remaining_b", 0)),
                "CloseLoss": pl.get("close_loss", False),
                "Created": rr["created_at"],
            })
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.info("No Beer Pong matches logged yet.")

    st.markdown("### Current Beer Pong standings (raw)")
    bp_raw, bp_wins, bp_games, bp_cups, bp_adj = compute_beerpong_raw(players)
    bp_results2 = fetch_event_results("Beer Pong")
    bp_rem = {p: 0 for p in players}
    for rr in bp_results2:
        pl = rr["payload"]
        a_rem = int(pl.get("cups_remaining_a", 0))
        b_rem = int(pl.get("cups_remaining_b", 0))
        for p in pl["team_a"]:
            bp_rem[p] += a_rem
        for p in pl["team_b"]:
            bp_rem[p] += b_rem
    bp_groups = build_tie_groups(players, bp_raw, [(bp_wins, True), (bp_cups, True), (bp_rem, True)])
    bp_order = [p for grp in bp_groups for p in grp]
    detail = []
    for i, p in enumerate(bp_order):
        detail.append({
            "rank": i+1,
            "player": display_player(p, name_map),
            "points": bp_raw[p],
            "wins": bp_wins[p],
            "cups_sunk": bp_cups[p],
            "cups_remaining_total": bp_rem[p],
            "wins": bp_wins[p],
            "games": bp_games[p],
            "adjustment": bp_adj[p],
        })
    st.dataframe(pd.DataFrame(detail), width="stretch")
    st.caption("Tie-breakers: points → wins → cups sunk → cups remaining total → tie (no letter fallback).")

# --- Telestrations ---
with tabs[2]:
    st.subheader("Telestrations")
    locked = is_event_locked("Telestrations")
    if locked:
        st.warning("Telestrations is locked. Admin can unlock in Setup.")

    round_no = st.number_input(
        "Round #", min_value=1, step=1, value=1, key="tel_round",
        disabled=is_event_locked("Telestrations")
    )

    with st.form("tel_round_form", clear_on_submit=True):
        entered_by = st.selectbox(
            "Entered by",
            [""] + players,
            format_func=lambda p: "—" if p == "" else display_player(p, name_map),
            disabled=is_event_locked("Telestrations"),
        )

        winners = st.multiselect(
            "Booklet winners (can be empty)",
            players,
            default=[],
            key="tel_winners",
            format_func=lambda x: display_player(x, name_map),
            disabled=is_event_locked("Telestrations"),
        )

        df = pd.DataFrame({
            "player": players,
            "name": [display_player(p, name_map) for p in players],
            "response_points": [0] * len(players),
        })
        edited = st.data_editor(
            df,
            width="stretch",
            num_rows="fixed",
            key="tel_editor",
            disabled=is_event_locked("Telestrations"),
            column_config={
                "player": st.column_config.TextColumn("Player (letter)", disabled=True),
                "name": st.column_config.TextColumn("Player", disabled=True),
                "response_points": st.column_config.NumberColumn("Response points", min_value=0, step=1),
            },
        )

        if st.form_submit_button("Save telestrations round", disabled=is_event_locked("Telestrations")):
            resp = {row["player"]: int(row["response_points"]) for _, row in edited.iterrows()}
            insert_event_result(
                "Telestrations",
                int(round_no),
                {"round_no": int(round_no), "booklet_winners": winners, "response_points": resp, "entered_by": (entered_by if entered_by else None)},
            )
            st.success("Saved.")
            st.rerun()
    st.divider()
    tel_results = fetch_event_results("Telestrations")
    if tel_results:
        st.markdown("### Logged rounds")
        st.dataframe(pd.DataFrame([
            {"id": r["id"], "Round": r["round_no"],
             "Booklet winners": ", ".join([display_player(w, name_map) for w in r["payload"].get("booklet_winners", [])]) or "(none)",
             "Created": r["created_at"]}
            for r in tel_results
        ]), width="stretch")
    else:
        st.info("No telestrations rounds logged yet.")

    st.markdown("### Current Telestrations standings (raw)")
    tel_raw, tel_bw, tel_rp, tel_adj = compute_telestrations_raw(players)
    tel_groups = build_tie_groups(players, tel_raw, [(tel_bw, True)])

    detail = []
    place = 1
    for group in tel_groups:
        for p in group:
            detail.append({
                "place": place,
                "player": display_player(p, name_map),
                "raw_points": tel_raw[p],
                "booklet_wins": tel_bw[p],
                "adjustment": tel_adj[p],
            })
        place += len(group)

    st.dataframe(pd.DataFrame(detail), width="stretch")
    st.caption("Tie-breakers: points → booklet wins → tie (no letter fallback).")

# --- Spoons ---
with tabs[3]:
    st.subheader("Spoons")
    locked = is_event_locked("Spoons")
    if locked:
        st.warning("Spoons is locked. Admin can unlock in Setup.")

    round_no = st.number_input(
        "Round #", min_value=1, step=1, value=1, key="sp_round",
        disabled=is_event_locked("Spoons")
    )

    with st.form("sp_round_form", clear_on_submit=True):
        entered_by = st.selectbox(
            "Entered by",
            [""] + players,
            format_func=lambda p: "—" if p == "" else display_player(p, name_map),
            disabled=is_event_locked("Spoons"),
        )

        order = st.multiselect(
            "Elimination order (select all 9 in exact order)",
            players,
            default=[],
            key="sp_order",
            format_func=lambda x: display_player(x, name_map),
            disabled=is_event_locked("Spoons"),
        )

        if st.form_submit_button("Save spoons round", disabled=is_event_locked("Spoons")):
            if len(order) != len(players):
                st.error("Select all 9 players in order.")
            else:
                insert_event_result(
                    "Spoons",
                    int(round_no),
                    {"round_no": int(round_no), "elimination_order": order, "entered_by": (entered_by if entered_by else None)},
                )
                st.success("Saved.")
                st.rerun()
    st.divider()
    sp_results = fetch_event_results("Spoons")
    if sp_results:
        st.markdown("### Logged rounds + placements")
        rows = []
        for r in sp_results:
            elim = r["payload"]["elimination_order"]
            # placements: last out is 1st place
            placement = list(reversed(elim))
            rows.append({
                "id": r["id"],
                "Round": r["round_no"],
                "1st": display_player(placement[0], name_map),
                "2nd": display_player(placement[1], name_map),
                "3rd": display_player(placement[2], name_map),
                "4th": display_player(placement[3], name_map),
                "5th": display_player(placement[4], name_map),
                "6th": display_player(placement[5], name_map),
                "7th": display_player(placement[6], name_map),
                "8th": display_player(placement[7], name_map),
                "9th": display_player(placement[8], name_map),
                "Created": r["created_at"],
            })
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.info("No spoons rounds logged yet.")

    st.markdown("### Current Spoons standings (raw)")
    sp_raw, sp_adj = compute_spoons_raw(players)
    sp_cb = compute_spoons_countback(players)
    sp_groups = build_tie_groups(players, sp_raw, [(sp_cb, True)])

    detail = []
    place = 1
    for grp in sp_groups:
        for p in grp:
            detail.append({
                "place": place,
                "player": display_player(p, name_map),
                "raw_points": sp_raw[p],
                "1st_finishes": sp_cb[p][0],
                "2nd_finishes": sp_cb[p][1],
                "3rd_finishes": sp_cb[p][2],
                "adjustment": sp_adj[p],
            })
        place += len(grp)

    st.dataframe(pd.DataFrame(detail), width="stretch")
    st.caption("Tie-breakers: points → countback (more 1sts, then 2nds, …) → tie (no letter fallback).")

# --- Secret Hitler ---
with tabs[4]:
    st.subheader("Secret Hitler")
    locked = is_event_locked("Secret Hitler")
    if locked:
        st.warning("Secret Hitler is locked. Admin can unlock in Setup.")

    game_no = st.number_input(
        "Game #", min_value=1, step=1, value=1, key="sh_game",
        disabled=is_event_locked("Secret Hitler")
    )


    # Role selection (kept outside the form so the Hitler dropdown can react to fascist selection)
    fascists = st.multiselect(
        "Fascists (include Hitler here)",
        players,
        default=[],
        key="sh_fascists",
        format_func=lambda x: display_player(x, name_map),
        disabled=is_event_locked("Secret Hitler"),
    )
    liberals = [p for p in players if p not in fascists]
    st.caption(f"Liberals inferred: {', '.join(display_player(p, name_map) for p in liberals) if liberals else '—'}")

    hitler = st.selectbox(
        "Hitler (must be in Fascists)",
        [""] + fascists,
        key="sh_hitler",
        format_func=lambda x: "—" if x == "" else display_player(x, name_map),
        disabled=is_event_locked("Secret Hitler"),
    )

    with st.form("sh_game_form", clear_on_submit=True):
        entered_by = st.selectbox(
            "Entered by",
            [""] + players,
            format_func=lambda p: "—" if p == "" else display_player(p, name_map),
            disabled=is_event_locked("Secret Hitler"),
        )

        winner_side = st.radio(
            "Winner side",
            ["Liberals", "Fascists"],
            horizontal=True,
            key="sh_win_side",
            disabled=is_event_locked("Secret Hitler"),
        )

        spicy_type = st.selectbox(
            "Spicy ending type",
            ["None", "Hitler killed", "Hitler elected"],
            key="sh_spicy_type",
            disabled=is_event_locked("Secret Hitler"),
        )

        if st.form_submit_button("Save Secret Hitler game", disabled=is_event_locked("Secret Hitler")):
            if not fascists or (hitler == "" or hitler not in fascists):
                st.error("Select fascists and pick Hitler from that list.")
            else:
                insert_event_result(
                    "Secret Hitler",
                    int(game_no),
                    {
                        "game_no": int(game_no),
                        "participants": players,
                        "fascists": fascists,
                        "hitler": hitler,
                        "winner_side": winner_side,
                        "spicy_type": spicy_type,
                        "entered_by": (entered_by if entered_by else None),
                    },
                )
                st.success("Saved.")
                st.rerun()

    st.divider()
    sh_results = fetch_event_results(\"Secret Hitler\")
    if sh_results:
        st.markdown("### Logged games")
        rows = []
        for r in sh_results:
            pl = r["payload"]
            rows.append({
                "id": r["id"],
                "Game": pl["game_no"],
                "Fascists": ", ".join([display_player(x, name_map) for x in pl["fascists"]]),
                "Hitler": display_player(pl["hitler"], name_map),
                "Winner": pl["winner_side"],
                "Spicy": pl.get("spicy_type", "None"),
                "Created": r["created_at"],
            })
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.info("No Secret Hitler games logged yet.")

    st.markdown("### Current Secret Hitler standings (raw)")
    sh_raw, sh_wins, sh_spicy, sh_adj = compute_secret_hitler_raw(players)
    sh_groups = build_tie_groups(players, sh_raw, [(sh_wins, True), (sh_spicy, True)])

    detail = []
    place = 1
    for grp in sh_groups:
        for p in grp:
            detail.append({
                "place": place,
                "player": display_player(p, name_map),
                "raw_points": sh_raw[p],
                "wins": sh_wins[p],
                "spicy_wins": sh_spicy[p],
                "adjustment": sh_adj[p],
            })
        place += len(grp)

    st.dataframe(pd.DataFrame(detail), width="stretch")
    st.caption("Tie-breakers: points → wins → spicy wins → tie (no letter fallback).")

# --- Breathalyzer ---
with tabs[5]:
    st.subheader("Breathalyzer")
    locked = is_event_locked("Breathalyzer")
    if locked:
        st.warning("Breathalyzer is locked. Admin can unlock in Setup.")

    with st.form("br_sess_form", clear_on_submit=True):
        entered_by = st.selectbox(
            "Entered by",
            [""] + players,
            format_func=lambda p: "—" if p == "" else display_player(p, name_map),
            disabled=is_event_locked("Breathalyzer"),
        )

        sess = st.number_input(
            "Session/Blow #",
            min_value=1,
            step=1,
            value=1,
            key="br_sess",
            disabled=is_event_locked("Breathalyzer"),
        )
        blower = st.selectbox(
            "Blower",
            players,
            key="br_blower",
            format_func=lambda x: display_player(x, name_map),
            disabled=is_event_locked("Breathalyzer"),
        )
        actual = st.number_input(
            "Actual BAC",
            min_value=0.0,
            step=0.001,
            value=0.000,
            format="%.3f",
            key="br_actual",
            disabled=is_event_locked("Breathalyzer"),
        )

        df = pd.DataFrame({"player": players, "guess": [""] * len(players)})
        edited = st.data_editor(
            df,
            width="stretch",
            num_rows="fixed",
            key="br_editor",
            disabled=is_event_locked("Breathalyzer"),
        )

        if st.form_submit_button("Save breathalyzer session", disabled=is_event_locked("Breathalyzer")):
            guesses = {}
            for _, row in edited.iterrows():
                g = str(row["guess"]).strip()
                if g == "" or g.lower() == "nan":
                    continue
                try:
                    guesses[row["player"]] = float(g)
                except Exception:
                    pass

            insert_event_result(
                "Breathalyzer",
                int(sess),
                {
                    "session_no": int(sess),
                    "sess": int(sess),  # backward-compat

                    "blower": blower,
                    "actual": float(actual),
                    "guesses": guesses,
                    "entered_by": (entered_by if entered_by else None),
                },
            )
            st.success("Saved.")
            st.rerun()
    st.divider()
    br_results = fetch_event_results("Breathalyzer")
    if br_results:
        st.markdown("### Logged sessions")
        st.dataframe(pd.DataFrame([
            {"id": r["id"], "Session": r["payload"].get("session_no", r["payload"].get("sess", r.get("round_no"))),
             "Blower": display_player(r["payload"]["blower"], name_map),
             "Actual": r["payload"]["actual"],
             "Guesses": len(r["payload"]["guesses"]),
             "Created": r["created_at"]}
            for r in br_results
        ]), width="stretch")
    else:
        st.info("No breathalyzer sessions logged yet.")

    st.markdown("### Current Breathalyzer standings (raw)")
    br_raw, br_closest, br_avgerr, br_adj = compute_breathalyzer_raw(players)
    br_groups = build_tie_groups(players, br_raw, [(br_closest, True), (br_avgerr, False)])

    detail = []
    place = 1
    for grp in br_groups:
        for p in grp:
            detail.append({
                "place": place,
                "player": display_player(p, name_map),
                "raw_points": br_raw[p],
                "closest_count": br_closest[p],
                "avg_error": (None if br_avgerr[p] == float("inf") else round(br_avgerr[p], 3)),
                "adjustment": br_adj[p],
            })
        place += len(grp)

    st.dataframe(pd.DataFrame(detail), width="stretch")
    st.caption("Tie-breakers: points → closest count → lower avg error → tie (no letter fallback).")

# --- Standings ---
with tabs[6]:
    st.subheader("Standings")

    # Optional auto-refresh (polling). Only affects this view-only tab.
    if st_autorefresh is not None:
        c_a, c_b = st.columns([1, 1])
        with c_a:
            auto = st.checkbox("Auto-refresh", value=False, help="Re-check the database periodically so you see other devices' updates.")
        with c_b:
            interval_s = st.selectbox("Interval", [2, 5, 10, 15], index=1, disabled=not auto)
        if auto:
            st_autorefresh(interval=int(interval_s) * 1000, key="standings_autorefresh")
    else:
        st.caption("Auto-refresh unavailable (missing dependency).")

    computed = compute_all(players)

    totals = {p: 0 for p in players}
    for ev in EVENTS:
        for p in players:
            totals[p] += computed[ev]["jj"].get(p, 0)

    st.markdown("### Overall (Jimmy Jabs points) — only completed events count")
    df_total = pd.DataFrame([{"player": display_player(p, name_map), "jj_total": totals[p]} for p in players])\
        .sort_values(["jj_total", "player"], ascending=[False, True])
    st.dataframe(df_total, width="stretch")

    st.divider()
    st.markdown("### By event (JJ points)")
    rows = []
    for p in players:
        row = {"player": display_player(p, name_map)}
        for ev in EVENTS:
            row[ev] = computed[ev]["jj"][p]
        row["Total"] = totals[p]
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).sort_values(["Total", "player"], ascending=[False, True]), width="stretch")

    st.divider()
    st.markdown("### How ties are resolved")
    st.write("""
When players have the same raw score, the app uses event-specific tie-breakers.
If they are still tied, they remain tied (no letter fallback). JJ points then use **skipped placements** (e.g., two tie for 1st → both get 9, next gets 7).
This prevents random reshuffling in the standings.
""")

# --- Admin ---
with tabs[7]:
    admin_login_ui()

    if is_admin():
        st.divider()
        st.markdown("## Tournament reset")
        clear_names = st.checkbox("Also clear player names (A–I mapping)", value=True)
        confirm = st.text_input("Type RESET to enable the reset button")
        if st.button("RESET TOURNAMENT", disabled=(confirm.strip().upper() != "RESET")):
            reset_tournament(clear_names=clear_names)
            st.success("Tournament reset. Refreshing…")
            st.rerun()

        st.divider()
        st.markdown("## Manual point adjustments (raw score deltas)")
        st.caption("Use this for rulings like “-1 point to player X in event Y”. These adjustments affect raw standings and therefore JJ points once event is marked completed.")
        ev = st.selectbox("Event", EVENTS, key="adj_event")
        pl = st.selectbox("Player", players, key="adj_player", format_func=lambda x: display_player(x, name_map))
        delta = st.number_input("Delta (can be negative)", value=0, step=1)
        note = st.text_input("Note (optional)")
        if st.button("Add adjustment"):
            if delta == 0:
                st.warning("Delta is 0 — nothing to add.")
            else:
                add_adjustment(ev, pl, int(delta), note)
                st.success("Adjustment added.")
                st.rerun()

        st.markdown("### Existing adjustments")
        adf = list_adjustments()
        if not adf.empty:
            # show with display names
            adf2 = adf.copy()
            adf2["player"] = adf2["player"].apply(lambda x: display_player(x, name_map))
            st.dataframe(adf2, width="stretch")
            del_id = st.selectbox("Delete adjustment id", adf["id"].tolist())
            if st.button("Delete selected adjustment"):
                delete_adjustment(int(del_id))
                st.success("Deleted.")
                st.rerun()
        else:
            st.info("No adjustments yet.")

        st.divider()
        st.markdown("## Delete logged entries")
        event = st.selectbox("Event to manage entries", EVENTS, key="adm_event")
        results = fetch_event_results(event)
        if results:
            label_map = {r["id"]: f'#{r["id"]} | round/game {r["round_no"]} | {r["created_at"]}' for r in results}
            sel_id = st.selectbox("Select entry", list(label_map.keys()), format_func=lambda x: label_map[x])
            if st.button("Delete selected entry"):
                delete_result(int(sel_id))
                st.success("Deleted.")
                st.rerun()
        else:
            st.info("No entries yet for this event.")