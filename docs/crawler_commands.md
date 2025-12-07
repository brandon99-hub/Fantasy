## Manager Crawler Command Reference

Run all commands from `C:\Users\USERR\PythonProject\FPLDataFetch` with your virtual environment activated. Every command below uses:

```
python -m backend.src.cli.crawl_managers <command> [options]
```

### Core crawl commands

| Command | Purpose | Key options |
| --- | --- | --- |
| `start` | Begin or restart the sequential ID crawl | `--start-id`, `--end-id`, `--batch-size` |
| `league` | Crawl a single classic league standings table | `--league-id`, `--max-pages`, `--start-page` |
| `seed` | Crawl all IDs listed in `MANAGER_SEED_LEAGUES` sequentially | `--max-pages` (per league) |
| `resume-range` | Resume the sequential ID crawl after a crash/stop | none (state read from DB) |
| `resume-league` | Resume the last league crawl (or a specific `--league-id`) | `--league-id` (optional) |

Examples:

```
python -m backend.src.cli.crawl_managers league --league-id 131 --max-pages 50
python -m backend.src.cli.crawl_managers seed --max-pages 40
python -m backend.src.cli.crawl_managers resume-range
```

### Live control (pause / resume / stop)

These commands work even while another terminal is running a crawl. The crawler checks for control signals and responds within a second or two.

| Command | Description |
| --- | --- |
| `pause` | Gracefully pause the active crawl. Workers finish in-flight requests and then idle. |
| `resume` | Resume a crawl paused via `pause`. |
| `stop` | Stop the active crawl entirely. Remaining IDs/pages are skipped until you manually resume via `resume-range` / `resume-league`. |

Each command accepts `--scope` with one of:

- `auto` (default) – apply to whichever crawl is active.
- `range` – target the sequential ID crawl specifically.
- `league` – target the league/seed crawl specifically.

Examples:

```
python -m backend.src.cli.crawl_managers pause --scope league
python -m backend.src.cli.crawl_managers resume --scope league
python -m backend.src.cli.crawl_managers stop --scope auto
```

### Status

```
python -m backend.src.cli.crawl_managers status [--scope auto|range|league]
```

Shows the current or most recent crawl for the requested scope, including ID/page progress, totals indexed/failed/deleted, and last heartbeat timestamps.

### League coverage & fixing status

- **League summary (all leagues):**

```bash
python -m backend.src.cli.crawl_managers league-summary
```

Lists every league we have indexed in `manager_league_memberships` with:
- `league_id`
- `league_name` (from DB)
- `manager_count` (how many managers we have for that league)
- last crawl `status` from `crawl_progress`

- **League summary (single league with coverage check):**

```bash
python -m backend.src.cli.crawl_managers league-summary --league-id 131
```

Shows, for that league:
- `league_id`
- `league_name` (from DB)
- managers indexed in DB
- last crawl status
- total entries from the FPL API (if reachable)
- coverage percentage = managers indexed / total entries

- **Fix a bad league status (e.g. stuck as running):**

```bash
python -m backend.src.cli.crawl_managers fix-league-status --league-id 131 --status completed
```

Updates the `crawl_progress` row(s) for that league to the given status (default is `completed`) so that
`status --scope league` stops reporting a stale `running`/`paused` state.

### Typical workflow

1. Seed crawl for your leagues:
   ```
   python -m backend.src.cli.crawl_managers seed --max-pages 50
   ```
2. If you need to pause (e.g., to save bandwidth):
   ```
   python -m backend.src.cli.crawl_managers pause --scope league
   ```
3. Resume when ready:
   ```
   python -m backend.src.cli.crawl_managers resume --scope league
   ```
4. To fully stop:
   ```
   python -m backend.src.cli.crawl_managers stop --scope auto
   ```
5. Later, resume the league crawl from its last saved page:
   ```
   python -m backend.src.cli.crawl_managers resume-league --league-id 131
   ```

### Notes

- Control commands write a signal file inside `backend/data/`. The running crawler reads the file and applies the requested action almost immediately.
- `resume-range` and `resume-league` restart crawls using database progress snapshots. Use these after a full `stop` or a crash.
- Environment variables `CRAWLER_MAX_WORKERS`, `CRAWLER_RATE_LIMIT`, and `MANAGER_SEED_LEAGUES` determine concurrency and which leagues the `seed` command walks through.

