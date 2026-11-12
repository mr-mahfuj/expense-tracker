# Expense Agent

Ask questions about your daily spending in plain English, or paste messy notes
and have them saved. A local Ollama model writes the SQL; Postgres runs it.

```
you> yesterday lunch 180, rickshaw 60, bkash to Rafi 500
you> how much did I spend on food in the last 10 days?
you> spending per category this month
```

## How it works

```
             +--------------+  1. which tool next?   +-------+
 you ------> | Orchestrator | <--------------------> |  LLM  |
             +--------------+                        +-------+
               |          |
   2. question |          | 4. the SQL from step 3
               v          v
       +--------------+  +--------------+
       | generate_sql |  | execute_sql  |
       |   (Tool 1)   |  |   (Tool 2)   |
       +--------------+  +--------------+
        schema + LLM      guard -> Postgres
        -> SQL (3)        -> rows
```

- **Orchestrator** (`orchestrator.py`) is the only thing that connects the
  tools. The LLM only chooses *which* tool runs next. The orchestrator passes
  your original text to Tool 1 and Tool 1's SQL to Tool 2, so a small model
  cannot garble pasted data or long SQL while copying it.
- **Tool 1 `generate_sql`** reads the live schema from Postgres, adds today's
  date, calls the LLM, and returns one SQL statement. It never runs it.
- **Tool 2 `execute_sql`** checks the SQL with a guard, then runs it. It never
  calls the LLM or Tool 1.
- Tests assert that neither tool references the other.

Every request first passes a relevance guard before routing, schema reads, or
cache access. It allows recording and querying expenses, and denies unrelated
questions, mixed unrelated tasks, destructive operations, and attempts to
override instructions. Invalid classifier output or an unavailable model stops
the request. The SQL writer's `SELECT 'UNSUPPORTED'` sentinel is also refused
before execution. Scope classification uses the same `OLLAMA_MODEL`
(`qwen2.5-coder:7b` by default) and sampling settings as the router.

Answers stream to the terminal by default. Routing JSON and SQL generation
remain buffered. Python callers can use `agent.run(question, on_token=callback)`
for incremental answer chunks; `agent.run(question)` remains buffered. The
returned `AgentResult.answer` contains the complete successful answer. An empty
stream or an answer-model failure before output falls back to a table; a stream
interrupted after output returns an error and marks the response incomplete.
Relevance tests use model doubles and verify enforcement, rather than real
model classification accuracy.

## Safety: three layers

The model writes the SQL, so it is not trusted.

1. **Guard** (`guards.py`): only one `SELECT` or `INSERT` statement; rejects
   `DELETE`, `DROP`, `UPDATE`, `ALTER`, `CREATE`, `pg_*` functions and so on.
2. **Read-only transaction**: SELECTs run with `read_only=True`, so Postgres
   itself refuses a write hidden inside one.
3. **Database role**: `expense_agent` is granted only `SELECT` and `INSERT`
   (`sql/02_schema.sql`). Even if layers 1 and 2 failed, it cannot delete or
   drop anything.

## Setup

### 1. Install Postgres and pgAdmin (skip what you have)

```bash
sudo apt install -y postgresql
sudo systemctl status postgresql --no-pager
```

Install pgAdmin from https://www.pgadmin.org/download/pgadmin-4-apt/ and
connect to `localhost:5432` as the `postgres` user.

### 2. Create the database and user

Open **Query Tool** on the default `postgres` database and open
`sql/01_create_database.sql`.

1. Replace `CHANGE_ME` with a password you choose.
2. Run the three statements **one at a time**: select one, press F5.
   `CREATE DATABASE` cannot run in a batch with other statements.

### 3. Create the table

In the pgAdmin tree: **Databases > expense_db** (refresh if it is missing),
right-click it, open **Query Tool**, load `sql/02_schema.sql`, and run it all
with F5. This creates the `expenses` table and gives `expense_agent` only
SELECT and INSERT.

Optional: run `sql/03_seed_sample.sql` in the same window for 12 sample rows
with dates relative to today.

### 4. Get the model

```bash
curl -fsSL --retry 5 https://ollama.com/install.sh | sh   # or: sudo snap install ollama
ollama pull qwen2.5-coder:7b
ollama list
```

### 5. Python environment

```bash
cd expense-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`: set the password in `DATABASE_URL`. If the password contains
characters like `@` `/` `#` or `:`, URL-encode them (`@` becomes `%40`).

### 6. Run

```bash
python main.py                                   # interactive
python main.py "how much did I spend on food in the last 10 days?"
python main.py --file notes.txt                  # pasted data from a file
python main.py --no-cache "total spend today"    # skip the cache once
python main.py --no-stream "total spend today"   # print the answer all at once
```

## Check it works

1. Run the seed script, then ask: `spending per category in the last 10 days`.
   Compare against pgAdmin: run the printed SQL there and the numbers match.
2. Ask `yesterday lunch 180, rickshaw 60`. In pgAdmin run
   `SELECT * FROM expenses ORDER BY id DESC LIMIT 5;` and see two new rows.
3. Ask the same question twice. The second run prints `cache hit`.
4. Ask `delete all my expenses`. It is refused and nothing changes.
5. Open `logs/agent_log.jsonl`.

## Logging

`logs/agent_log.jsonl` is JSON Lines: one object per line, appended, so a crash
never corrupts earlier entries. Every line has `ts` (ISO time in your
timezone), `run_id` (ties together all lines of one question) and `event`.

| event | key fields |
|---|---|
| `question` | `question` |
| `request_allowed` / `request_rejected` | `model` or rejection `reason` |
| `decision` | `step`, `action`, `reason`, `source` (`llm`, `state_machine`, `fallback_after_bad_json`) |
| `sql_generated` | `sql`, `raw_output`, `from_cache`, `retry`, `model`, `temperature`, `top_k`, `top_p`, `duration_ms` |
| `sql_rejected` | `sql`, `reason` |
| `sql_executed` | `sql`, `kind`, `ok`, `rowcount`, `error`, `duration_ms` |
| `answer` / `error` | `answer` or `message`, `sql`, `steps` |

Useful queries (needs `jq`, `sudo apt install jq`):

```bash
jq -c 'select(.event=="sql_generated") | {ts, sql}' logs/agent_log.jsonl
jq -c 'select(.event=="sql_executed" and .ok==false)' logs/agent_log.jsonl
jq -c 'select(.run_id=="<id>")' logs/agent_log.jsonl
```

The log holds your questions and the SQL, so it is git-ignored.

## Caching

`data/sql_cache.json` stores the **SQL the model wrote**, never the query
results, because results change as you add expenses. The saved SQL is re-run
each time, so answers stay current while the slow model call is skipped.

- The key hashes question + live schema + model name + **today's date**.
  Changing a table invalidates old entries; "last 10 days" cannot reuse
  yesterday's SQL.
- SQL is cached only **after it ran successfully**. A cached statement that
  later fails is deleted.
- Writes are atomic (temp file then rename); a corrupt file acts as empty.
- Oldest entries are evicted past `CACHE_MAX_ENTRIES`.
- Turn off with `CACHE_ENABLED=false` or `--no-cache`.

## Temperature, top_k, top_p

Set per role in `.env`:

| role | temperature | top_k | top_p | why |
|---|---|---|---|---|
| router (picks the next tool) | 0.0 | 20 | 0.8 | must be deterministic |
| SQL writer | 0.1 | 40 | 0.9 | accuracy matters more than variety |
| answer wording | 0.5 | 40 | 0.9 | natural phrasing |

## Swapping parts

- **Model backend:** implement `LLMClient.chat` and `chat_stream` (see `llm.py`) and change one
  line in `cli.py:build_agent`.
- **Database:** implement `Database.get_schema` and `run_sql` (see `db.py`).
- **Different data, same agent:** change `sql/02_schema.sql` and the three
  examples in `tools.py:SQL_SYSTEM_PROMPT`. Small models copy examples closely,
  so keep them matched to your table.

## Tests

```bash
python -m unittest discover -s tests -t .
```

These use a fake model and a fake database, so they need neither Ollama nor
Postgres.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `Config error: DATABASE_URL is not set` | create `.env` from `.env.example` |
| `Cannot connect to Postgres` | service down (`systemctl status postgresql`), wrong password, or special characters not URL-encoded |
| `permission denied for table expenses` | `02_schema.sql` was run while connected to the wrong database; re-run it inside `expense_db` |
| `Ollama call failed` | `ollama list` / `systemctl status ollama`; check `OLLAMA_MODEL` matches a pulled model |
| First question is slow | the model is loading from disk; `OLLAMA_KEEP_ALIVE` keeps it in RAM afterwards |
| Wrong or invented SQL | read `raw_output` in the log; fix the examples in `tools.py` or try a bigger model |
| `Ran out of steps` | the model kept picking bad actions; try `ROUTER_TEMPERATURE=0` or a larger model |
