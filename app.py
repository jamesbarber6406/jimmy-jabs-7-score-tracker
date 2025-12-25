
import streamlit as st
import json
from datetime import datetime
import pandas as pd
import random
from contextlib import contextmanager

APP_TITLE = "Jimmy Jabs (Tournament Tracker)"
DEFAULT_PLAYERS = list("ABCDEFGHI")

EVENTS = ["Beer Pong", "Telestrations", "Spoons", "Secret Hitler", "Breathalyzer"]

# Beer Pong scoring
BEERPONG_WIN_PTS = 2
BEERPONG_LOSS_PTS = 0
BEERPONG_CLOSE_LOSS_PTS = 1

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
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(self._q(sql), params or [])
            conn.commit()

    def fetchall(self, sql: str, params=None):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(self._q(sql), params or [])
            rows = cur.fetchall()
        return rows

    def fetchone(self, sql: str, params=None):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(self._q(sql), params or [])
            row = cur.fetchone()
        return row

db = DB()

def init_db():
    # Slight differences in SQL types between sqlite/postgres; keep it compatible.
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

def insert_event_result(event: str, round_no: int, payload: dict):
    db.execute(
        "INSERT INTO event_results(event, round_no, payload_json, created_at) VALUES(?, ?, ?, ?)",
        [event, int(round_no), json.dumps(payload), datetime.utcnow().isoformat()]
    )

def fetch_event_results(event: str):
    rows = db.fetchall(
        "SELECT id, round_no, payload_json, created_at FROM event_results WHERE event=? ORDER BY id ASC",
        [event],
    )
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

def generate_equal_beerpong_schedule(players, rounds=9, tries=3000, seed=42):
    rng = random.Random(seed)
    players = list(players)
    if (8 * rounds) % len(players) != 0:
        raise ValueError("Equal games with 2v2 requires rounds multiple of 9 (9,18,...).")

    games_per_player = (8 * rounds) // len(players)
    byes_per_player = rounds - games_per_player
    bye_list = []
    for p in players:
        bye_list.extend([p] * byes_per_player)
    assert len(bye_list) == rounds

    best, best_score = None, float("inf")
    for _ in range(tries):
        rng.shuffle(bye_list)
        rounds_payload = []
        for r in range(rounds):
            bye = bye_list[r]
            active = [p for p in players if p != bye]
            rng.shuffle(active)
            matches = []
            for chunk in [active[:4], active[4:8]]:
                rng.shuffle(chunk)
                matches.append({"team_a": chunk[:2], "team_b": chunk[2:]})
            rounds_payload.append({"round_no": r+1, "bye": bye, "matches": matches})
        sc = score_schedule(rounds_payload, players)
        if sc < best_score:
            best, best_score = rounds_payload, sc
    return best, best_score

# ----------------------------
# Event computations
# ----------------------------
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

    return raw, wins, games, cups_sunk

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
    return raw, booklet_wins, response_pts

def compute_spoons_raw(players):
    results = fetch_event_results("Spoons")
    raw = {p: 0 for p in players}
    for r in results:
        order = r["payload"]["elimination_order"]
        for idx, p in enumerate(order):
            raw[p] += (idx + 1)
    return raw

def compute_secret_hitler_raw(players):
    results = fetch_event_results("Secret Hitler")
    raw = {p: 0 for p in players}
    wins = {p: 0 for p in players}
    spicy_wins = {p: 0 for p in players}

    for r in results:
        pl = r["payload"]
        participants = pl["participants"]
        fascists = pl["fascists"]
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

    return raw, wins, spicy_wins

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
    return raw, closest_count, avg_error

def jj_points_from_order(players_ordered):
    n = len(players_ordered)
    return {p: n - i for i, p in enumerate(players_ordered)}

def sort_with_tiebreakers(players, primary, tiebreakers, higher_is_better=True):
    def key(p):
        vals = [primary.get(p, 0)]
        for tb_dict, hib in tiebreakers:
            v = tb_dict.get(p, 0)
            vals.append(v if hib else -v)
        vals.append(p)
        return tuple(vals)
    return sorted(players, key=key, reverse=higher_is_better)

def compute_all(players):
    bp_raw, bp_wins, bp_games, bp_cups = compute_beerpong_raw(players)
    bp_order = sort_with_tiebreakers(players, bp_raw, [(bp_cups, True), (bp_wins, True), (bp_games, True)], True)

    tel_raw, tel_bw, tel_rp = compute_telestrations_raw(players)
    tel_order = sort_with_tiebreakers(players, tel_raw, [(tel_bw, True), (tel_rp, True)], True)

    sp_raw = compute_spoons_raw(players)
    sp_order = sort_with_tiebreakers(players, sp_raw, [], True)

    sh_raw, sh_wins, sh_spicy = compute_secret_hitler_raw(players)
    sh_order = sort_with_tiebreakers(players, sh_raw, [(sh_wins, True), (sh_spicy, True)], True)

    br_raw, br_closest, br_avgerr = compute_breathalyzer_raw(players)
    br_order = sort_with_tiebreakers(players, br_raw, [(br_closest, True), (br_avgerr, False)], True)

    computed = {
        "Beer Pong": {"raw": bp_raw, "order": bp_order, "jj": jj_points_from_order(bp_order), "cups": bp_cups},
        "Telestrations": {"raw": tel_raw, "order": tel_order, "jj": jj_points_from_order(tel_order)},
        "Spoons": {"raw": sp_raw, "order": sp_order, "jj": jj_points_from_order(sp_order)},
        "Secret Hitler": {"raw": sh_raw, "order": sh_order, "jj": jj_points_from_order(sh_order)},
        "Breathalyzer": {"raw": br_raw, "order": br_order, "jj": jj_points_from_order(br_order)},
    }

    for ev in EVENTS:
        if not is_event_completed(ev):
            computed[ev]["jj"] = {p: 0 for p in players}
    return computed

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

init_db()
seed_default_players()

players_df = get_players_df()
players = players_df["letter"].tolist()
name_map = {row["letter"]: row["name"] for _, row in players_df.iterrows()}

tabs = st.tabs(["Setup", "Beer Pong", "Telestrations", "Spoons", "Secret Hitler", "Breathalyzer", "Standings", "Admin"])

with tabs[0]:
    st.subheader("Setup")
    st.caption(f"Database backend: **{db.kind}**")
    if db.kind == "sqlite":
        st.info("Local mode (SQLite). For a shareable URL with everyone entering scores, deploy to Streamlit Community Cloud and set DATABASE_URL for Postgres.")
    edited = st.data_editor(players_df, use_container_width=True, num_rows="fixed")
    if st.button("Save player names"):
        for _, row in edited.iterrows():
            db.execute("UPDATE players SET name=? WHERE letter=?", [row["name"], row["letter"]])
        st.success("Saved.")

    st.divider()
    st.markdown("### Event completion (JJ points awarded only when complete)")
    cols = st.columns(5)
    for i, ev in enumerate(EVENTS):
        with cols[i]:
            done = is_event_completed(ev)
            new_done = st.checkbox(ev, value=done, key=f"done_{ev}")
            if new_done != done:
                set_event_completed(ev, new_done)

with tabs[1]:
    st.subheader("Beer Pong")
    schedule = load_schedule()

    c1, c2 = st.columns([1, 1])
    with c1:
        rounds = st.number_input("Rounds", min_value=9, step=9, value=int(get_setting("beerpong_rounds", "9")))
        seed = st.number_input("Schedule seed", min_value=1, step=1, value=int(get_setting("beerpong_seed", "42")))
        tries = st.number_input("Generator tries", min_value=500, step=500, value=int(get_setting("beerpong_tries", "3000")))
        if st.button("Generate schedule"):
            if (8 * int(rounds)) % len(players) != 0:
                st.error("Equal games not possible unless rounds is a multiple of 9 (9, 18, 27...).")
            else:
                best, best_score = generate_equal_beerpong_schedule(players, rounds=int(rounds), tries=int(tries), seed=int(seed))
                clear_schedule()
                for r in best:
                    store_schedule(r["round_no"], r)
                upsert_setting("beerpong_rounds", str(int(rounds)))
                upsert_setting("beerpong_seed", str(int(seed)))
                upsert_setting("beerpong_tries", str(int(tries)))
                st.success(f"Generated. Score (lower better): {best_score:.2f}")
                schedule = load_schedule()

    with c2:
        if schedule:
            preview = []
            for rno in sorted(schedule.keys()):
                r = schedule[rno]
                m1, m2 = r["matches"]
                preview.append({
                    "Round": rno,
                    "Bye": display_player(r["bye"], name_map),
                    "Match 1": f'{display_team(m1["team_a"], name_map)} vs {display_team(m1["team_b"], name_map)}',
                    "Match 2": f'{display_team(m2["team_a"], name_map)} vs {display_team(m2["team_b"], name_map)}',
                })
            st.dataframe(pd.DataFrame(preview), use_container_width=True)
        else:
            st.info("No schedule yet. Generate one.")

    st.divider()
    if schedule:
        rno = st.selectbox("Round to log", sorted(schedule.keys()))
        r = schedule[rno]
        match_idx = st.radio("Match", [1, 2], horizontal=True)
        m = r["matches"][match_idx - 1]
        st.write("Team A:", display_team(m["team_a"], name_map), "vs Team B:", display_team(m["team_b"], name_map))

        winner = st.radio("Winner", ["A", "B"], horizontal=True)
        a_rem = st.number_input("Cups remaining (Team A)", min_value=0, max_value=6, value=0, step=1)
        b_rem = st.number_input("Cups remaining (Team B)", min_value=0, max_value=6, value=0, step=1)
        suggested_close = (winner == "A" and a_rem == 0 and b_rem == 1) or (winner == "B" and b_rem == 0 and a_rem == 1)
        close_loss = st.checkbox("Close loss (both teams were at 1 cup)", value=suggested_close)

        if st.button("Save match result"):
            insert_event_result("Beer Pong", int(rno), {
                "round_no": int(rno),
                "match_idx": int(match_idx),
                "team_a": m["team_a"],
                "team_b": m["team_b"],
                "winner": winner,
                "cups_remaining_a": int(a_rem),
                "cups_remaining_b": int(b_rem),
                "close_loss": bool(close_loss),
            })
            st.success("Saved.")

with tabs[2]:
    st.subheader("Telestrations")
    round_no = st.number_input("Round #", min_value=1, step=1, value=1, key="tel_round")
    winners = st.multiselect("Booklet winners (can be empty)", players, default=[], key="tel_winners", format_func=lambda x: display_player(x, name_map))
    df = pd.DataFrame({"player": players, "response_points": [0]*len(players)})
    edited = st.data_editor(df, use_container_width=True, num_rows="fixed", key="tel_editor")
    if st.button("Save telestrations round"):
        resp = {row["player"]: int(row["response_points"]) for _, row in edited.iterrows()}
        insert_event_result("Telestrations", int(round_no), {"round_no": int(round_no), "booklet_winners": winners, "response_points": resp})
        st.success("Saved.")

with tabs[3]:
    st.subheader("Spoons")
    round_no = st.number_input("Round #", min_value=1, step=1, value=1, key="sp_round")
    order = st.multiselect("Elimination order (select all 9 in exact order)", players, default=[], key="sp_order", format_func=lambda x: display_player(x, name_map))
    if st.button("Save spoons round"):
        if len(order) != len(players):
            st.error("Select all 9 players in order.")
        else:
            insert_event_result("Spoons", int(round_no), {"round_no": int(round_no), "elimination_order": order})
            st.success("Saved.")

with tabs[4]:
    st.subheader("Secret Hitler")
    game_no = st.number_input("Game #", min_value=1, step=1, value=1, key="sh_game")
    participants = st.multiselect("Participants", players, default=players, key="sh_part", format_func=lambda x: display_player(x, name_map))
    fascists = st.multiselect("Fascists (include Hitler)", participants, default=[], key="sh_fasc", format_func=lambda x: display_player(x, name_map))
    hitler = st.selectbox("Hitler (must be in Fascists)", fascists if fascists else [""], key="sh_hitler")
    winner_side = st.radio("Winner side", ["Liberals", "Fascists"], horizontal=True, key="sh_win_side")
    spicy_type = st.selectbox("Spicy ending type", ["None", "Hitler elected", "Hitler killed"], key="sh_spicy_type")
    if st.button("Save Secret Hitler game"):
        if not fascists or (hitler == "" or hitler not in fascists):
            st.error("Select fascists and pick Hitler from that list.")
        else:
            insert_event_result("Secret Hitler", int(game_no), {
                "game_no": int(game_no),
                "participants": participants,
                "fascists": fascists,
                "hitler": hitler,
                "winner_side": winner_side,
                "spicy_type": spicy_type,
            })
            st.success("Saved.")

with tabs[5]:
    st.subheader("Breathalyzer")
    sess = st.number_input("Session/Blow #", min_value=1, step=1, value=1, key="br_sess")
    blower = st.selectbox("Blower", players, key="br_blower", format_func=lambda x: display_player(x, name_map))
    actual = st.number_input("Actual BAC", min_value=0.0, step=0.001, value=0.000, format="%.3f", key="br_actual")
    df = pd.DataFrame({"player": players, "guess": [""]*len(players)})
    edited = st.data_editor(df, use_container_width=True, num_rows="fixed", key="br_editor")
    if st.button("Save breathalyzer session"):
        guesses = {}
        for _, row in edited.iterrows():
            g = str(row["guess"]).strip()
            if g and g.lower() != "nan":
                guesses[row["player"]] = float(g)
        if not guesses:
            st.error("Enter at least one guess.")
        else:
            insert_event_result("Breathalyzer", int(sess), {"session_no": int(sess), "blower": blower, "actual": float(actual), "guesses": guesses})
            st.success("Saved.")

with tabs[6]:
    st.subheader("Standings")
    computed = compute_all(players)
    totals = {p: 0 for p in players}
    for ev in EVENTS:
        for p in players:
            totals[p] += computed[ev]["jj"].get(p, 0)

    st.markdown("### Overall (Jimmy Jabs points) — only completed events count")
    df_total = pd.DataFrame([{"player": display_player(p, name_map), "jj_total": totals[p]} for p in players])\
        .sort_values(["jj_total", "player"], ascending=[False, True])
    st.dataframe(df_total, use_container_width=True)

with tabs[7]:
    st.subheader("Admin (delete entries)")
    event = st.selectbox("Event", EVENTS, key="adm_event")
    results = fetch_event_results(event)
    if results:
        label_map = {r["id"]: f'#{r["id"]} | round/game {r["round_no"]} | {r["created_at"]}' for r in results}
        sel_id = st.selectbox("Select entry", list(label_map.keys()), format_func=lambda x: label_map[x])
        if st.button("Delete selected"):
            delete_result(int(sel_id))
            st.success("Deleted.")
    else:
        st.info("No entries yet for this event.")
