# Jimmy Jabs Tournament Tracker (Streamlit)

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud + Supabase Postgres)
1) Create a GitHub repo and push these files.
2) Create a Supabase project and copy the Postgres connection string.
3) In Streamlit Community Cloud, deploy the repo and set a secret:
   - `DATABASE_URL = "<your_supabase_connection_string>"`

Note: Supabase direct connections can be IPv6-only; if your host needs IPv4, use Supabase's pooler (Supavisor) connection string.


## Notes
- Players are letters A–I by default (map to names in Setup if you want).
- Beer Pong schedule is generated for **9 rounds** to ensure each player plays **8 games** with **1 bye** (equal games).
- Scoring rules implemented:
  - Beer Pong: win=2, loss=0, close-loss=1 (if both teams at 1 cup)
  - Telestrations: response(0–8) + 10 for booklet winner
  - Spoons: elimination order gives 1..9 per round (last out=9)
  - Secret Hitler: win=4, loss=2, spicy bonus +1 to winners
  - Breathalyzer: closest=5, 2nd=3, 3rd=2, others=1; ties don't split; places not skipped
- Standings converts each event into Jimmy Jabs points (1st=9 … last=1) and totals overall.


# Jimmy Jabs Tournament Tracker (Streamlit) — v2

## What changed vs v1
- Beer Pong: log cups remaining for each team; app computes cups sunk and uses it as a tiebreaker.
- Beer Pong schedule display uses names (if you mapped letters → names).
- Telestrations: **multiple** booklet winners per round (or none).
- Secret Hitler: enter **teams** (Fascists including Hitler) and winner side + spicy type.
- Standings: **Jimmy Jabs points are only awarded after you mark an event complete** (Setup tab). Incomplete events contribute 0 JJ points.


# Jimmy Jabs Tournament Tracker — v3 (Shareable URL ready)

## What changed vs v2
This version supports:
- Local run with SQLite (`jimmy_jabs.db`)
- Cloud deployment with Postgres if you set `DATABASE_URL` in Streamlit secrets


## Note
Updated to use `width='stretch'` instead of deprecated `use_container_width`.


# Jimmy Jabs Tournament Tracker — v4

## Fixes / additions
- Names now propagate immediately after saving (forces rerun + refresh from DB).
- Each event tab shows logged rounds/games and a “current standings (raw)” table.
- Tie-breakers are shown and are deterministic (final fallback = letter).
- Admin PIN support (set `ADMIN_PIN` in Streamlit secrets).
- Event lock toggle (admin-only) to prevent edits after completion.
- Tournament reset button (admin-only).
- Manual point adjustments per event/player (admin-only), applied to raw standings.


## v5 tie update
JJ points now use skipped placements on ties; no letter fallback.


## v5.1
Added Spoons countback tie-breaker (more 1sts, then 2nds, etc.).

## v5.2
Fixed tie-group UI functions and further Spoons countback logic.

## v5.3
- Halved beer pong games - Doubled scoring
- Changed tables and inputs to be batch-submit forms - app no longer rewrites every time there's an input
- Added auto-refresh toggle to Standings tab so other devices can see changes without full refreshing
- Added "entered by" dropdown for user submit forms - track each score input

## v5.3.1
- fixed broken functions

## v5.3.2
- fixed broken reset function

## v5.3.3
- fixed issues with bye rounds for beer pong

## v5.4
- hotfix for session number issues with breathalyzer

## v5.5
- changed telestrations table to reflect name changes
- fixed secret hitler hitler dropdown to update immediately as fascists chosen
- added retry for postgres so app streamlit error won't crash entire app

## Edited Requirements
- Added auto refresh for streamlit - was already in code but not in requirements