import json
import tempfile
import unittest
from pathlib import Path

from expense_agent.cache import JsonCache, NullCache
from expense_agent.clock import Clock
from expense_agent.db import DatabaseError, QueryResult
from expense_agent.llm import Sampling
from expense_agent.logger import JsonlLogger
from expense_agent.orchestrator import Orchestrator
from expense_agent.tools import SqlExecutorTool, SqlGeneratorTool

SAMPLING = Sampling(temperature=0.0, top_k=1, top_p=1.0, num_predict=10)
SELECT_SQL = "SELECT SUM(amount) FROM expenses WHERE spent_on >= CURRENT_DATE - 10"
GENERATE, EXECUTE, FINAL = "generate_sql", "execute_sql", "final"


class FakeLLM:
    """Plays all three roles. The role is told apart by how it is called."""

    model = "fake-model"

    def __init__(self, router_actions, sql_outputs, answer="You spent 810."):
        self.router_actions = list(router_actions)
        self.sql_outputs = list(sql_outputs)
        self.answer = answer
        self.sql_requests: list[str] = []
        self.answer_calls = 0

    def chat(self, messages, sampling, json_schema=None):
        if json_schema is not None:
            return json.dumps({"action": self.router_actions.pop(0), "reason": "test"})
        if messages[0]["content"].startswith("You write PostgreSQL"):
            self.sql_requests.append(messages[1]["content"])
            return self.sql_outputs.pop(0)
        self.answer_calls += 1
        return self.answer


class FakeDatabase:
    def __init__(self, results):
        self.results = list(results)
        self.executed: list[tuple[str, bool]] = []

    def get_schema(self):
        return "TABLE expenses (id bigint, spent_on date, amount numeric, category text)"

    def run_sql(self, sql, read_only):
        self.executed.append((sql, read_only))
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def rows_result():
    return QueryResult(["sum"], [["810.00"]], 1, False)


def insert_result():
    return QueryResult([], [], 2, False)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.dir = Path(self._dir.name)
        self.clock = Clock("Asia/Dhaka")
        self.log_path = self.dir / "log.jsonl"

    def make_agent(self, llm, db, use_cache=True, max_retries=2):
        logger = JsonlLogger(self.log_path, self.clock)
        self.cache = JsonCache(self.dir / "cache.json", 50, self.clock) if use_cache else NullCache()
        generator = SqlGeneratorTool(llm, db, self.cache, logger, self.clock, SAMPLING)
        executor = SqlExecutorTool(db, logger)
        return Orchestrator(
            llm=llm,
            generator=generator,
            executor=executor,
            cache=self.cache,
            logger=logger,
            router_sampling=SAMPLING,
            answer_sampling=SAMPLING,
            max_steps=10,
            max_sql_retries=max_retries,
            max_rows_to_llm=20,
        )

    def log_events(self):
        lines = self.log_path.read_text().splitlines()
        return [json.loads(line) for line in lines]

    def test_select_happy_path_logs_sql_and_answers(self):
        llm = FakeLLM([GENERATE, EXECUTE, FINAL], ["```sql\n" + SELECT_SQL + ";\n```"])
        db = FakeDatabase([rows_result()])
        result = self.make_agent(llm, db).run("deposits in the last 10 days")

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "You spent 810.")
        self.assertEqual(result.sql, SELECT_SQL)  # markdown fence removed
        self.assertEqual(db.executed, [(SELECT_SQL, True)])  # SELECT runs read-only

        events = self.log_events()
        generated = [e for e in events if e["event"] == "sql_generated"][0]
        self.assertEqual(generated["sql"], SELECT_SQL)
        self.assertEqual({e["run_id"] for e in events}, {result.run_id})
        self.assertTrue(all("ts" in e for e in events))

    def test_second_identical_question_uses_cache(self):
        llm = FakeLLM([GENERATE, EXECUTE, FINAL, GENERATE, EXECUTE, FINAL], [SELECT_SQL])
        db = FakeDatabase([rows_result(), rows_result()])
        agent = self.make_agent(llm, db)

        first = agent.run("total spend last 10 days")
        second = agent.run("total   spend last 10 days")  # extra spaces: same question

        self.assertFalse(first.from_cache)
        self.assertTrue(second.from_cache)
        self.assertEqual(len(llm.sql_requests), 1)  # the model wrote SQL only once
        self.assertEqual(len(db.executed), 2)  # but the query still ran both times

    def test_cache_off_always_calls_model(self):
        llm = FakeLLM([GENERATE, EXECUTE, FINAL] * 2, [SELECT_SQL, SELECT_SQL])
        db = FakeDatabase([rows_result(), rows_result()])
        agent = self.make_agent(llm, db, use_cache=False)
        agent.run("q")
        agent.run("q")
        self.assertEqual(len(llm.sql_requests), 2)

    def test_failed_sql_is_retried_with_error_and_not_cached(self):
        bad_sql = "SELECT SUM(amt) FROM expenses"
        llm = FakeLLM([GENERATE, EXECUTE, GENERATE, EXECUTE, FINAL], [bad_sql, SELECT_SQL])
        db = FakeDatabase([DatabaseError('column "amt" does not exist'), rows_result()])
        agent = self.make_agent(llm, db)

        result = agent.run("total spend")

        self.assertTrue(result.ok)
        self.assertEqual(result.sql, SELECT_SQL)
        self.assertIn('column "amt" does not exist', llm.sql_requests[1])  # error fed back
        self.assertIn(bad_sql, llm.sql_requests[1])
        cached_sqls = [e["sql"] for e in json.loads((self.dir / "cache.json").read_text())["entries"].values()]
        self.assertEqual(cached_sqls, [SELECT_SQL])

    def test_gives_up_after_max_retries(self):
        llm = FakeLLM([GENERATE, EXECUTE] * 3, ["SELECT bad1", "SELECT bad2", "SELECT bad3"])
        errors = [DatabaseError("boom")] * 3
        result = self.make_agent(llm, FakeDatabase(errors), max_retries=2).run("q")
        self.assertFalse(result.ok)
        self.assertIn("boom", result.answer)

    def test_dangerous_sql_never_reaches_database(self):
        llm = FakeLLM([GENERATE, EXECUTE] * 3, ["DROP TABLE expenses"] * 3)
        db = FakeDatabase([])
        result = self.make_agent(llm, db, max_retries=2).run("delete everything")
        self.assertFalse(result.ok)
        self.assertEqual(db.executed, [])
        self.assertIn("sql_rejected", [e["event"] for e in self.log_events()])

    def test_insert_runs_in_write_mode_and_reports_rows(self):
        insert_sql = "INSERT INTO expenses (amount, category) VALUES (180, 'food'), (60, 'transport')"
        llm = FakeLLM([GENERATE, EXECUTE, FINAL], [insert_sql])
        db = FakeDatabase([insert_result()])
        result = self.make_agent(llm, db).run("lunch 180, rickshaw 60")
        self.assertEqual(db.executed, [(insert_sql, False)])
        self.assertIn("2 row(s)", result.answer)
        self.assertEqual(llm.answer_calls, 0)  # no model call needed to say "saved"

    def test_model_picking_final_too_early_does_not_loop(self):
        # Router says "final" first (invalid); the orchestrator then takes the
        # valid next step itself instead of asking the model again.
        llm = FakeLLM([FINAL, EXECUTE, FINAL], [SELECT_SQL])
        db = FakeDatabase([rows_result()])
        result = self.make_agent(llm, db).run("q")
        self.assertTrue(result.ok)
        sources = [e["source"] for e in self.log_events() if e["event"] == "decision"]
        self.assertIn("state_machine", sources)

    def test_unusable_router_output_falls_back_to_state_machine(self):
        class GarbageRouterLLM(FakeLLM):
            def chat(self, messages, sampling, json_schema=None):
                if json_schema is not None:
                    return "not json at all"
                return super().chat(messages, sampling, json_schema)

        llm = GarbageRouterLLM([], [SELECT_SQL])
        db = FakeDatabase([rows_result()])
        result = self.make_agent(llm, db).run("q")
        self.assertTrue(result.ok)

    def test_tool_modules_do_not_reference_each_other(self):
        source = Path(__file__).resolve().parent.parent.joinpath("expense_agent", "tools.py").read_text()
        generator_part, executor_part = source.split("class SqlExecutorTool")
        self.assertNotIn("SqlExecutorTool", generator_part)
        self.assertNotIn("SqlGeneratorTool", executor_part)


if __name__ == "__main__":
    unittest.main()
