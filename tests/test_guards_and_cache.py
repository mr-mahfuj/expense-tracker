import json
import tempfile
import unittest
from pathlib import Path

from expense_agent.cache import JsonCache, build_key
from expense_agent.clock import Clock
from expense_agent.guards import SqlRejected, extract_sql, validate


class ExtractSqlTests(unittest.TestCase):
    def test_strips_markdown_fence(self):
        raw = "```sql\nSELECT 1;\n```"
        self.assertEqual(extract_sql(raw), "SELECT 1")

    def test_strips_fence_without_language(self):
        self.assertEqual(extract_sql("```\nSELECT 1\n```"), "SELECT 1")

    def test_strips_unclosed_fence(self):
        self.assertEqual(extract_sql("```sql\nSELECT 1"), "SELECT 1")

    def test_drops_intro_text(self):
        self.assertEqual(extract_sql("Here is the query:\nSELECT 1;"), "SELECT 1")

    def test_drops_trailing_explanation(self):
        raw = "SELECT 1; -- done\nThis query returns one."
        self.assertEqual(extract_sql(raw), "SELECT 1")

    def test_keeps_semicolon_inside_string(self):
        raw = "SELECT * FROM expenses WHERE note = 'a;b';"
        self.assertEqual(extract_sql(raw), "SELECT * FROM expenses WHERE note = 'a;b'")

    def test_text_without_sql_passes_through_and_is_rejected_by_the_guard(self):
        # Kept as-is on purpose: the log then shows exactly what the model said
        # (e.g. "DROP TABLE ..."), and validate() names the reason.
        text = extract_sql("sorry, I cannot help")
        self.assertEqual(text, "sorry, I cannot help")
        with self.assertRaises(SqlRejected):
            validate(text)


class ValidateTests(unittest.TestCase):
    def test_accepts_select_insert_with(self):
        self.assertEqual(validate("SELECT 1;").kind, "SELECT")
        self.assertEqual(validate("insert into expenses (amount) values (5)").kind, "INSERT")
        self.assertEqual(validate("WITH t AS (SELECT 1) SELECT * FROM t").kind, "SELECT")

    def test_keyword_inside_string_is_fine(self):
        validate("SELECT * FROM expenses WHERE note = 'drop table; delete'")

    def test_keyword_inside_comment_is_fine(self):
        validate("SELECT 1 -- delete everything")

    def test_updated_at_style_names_are_not_keywords(self):
        validate("SELECT created_at FROM expenses")

    def test_rejects_writes_and_ddl(self):
        for sql in (
            "DROP TABLE expenses",
            "DELETE FROM expenses",
            "UPDATE expenses SET amount = 1",
            "TRUNCATE expenses",
            "CREATE TABLE x (a int)",
            "ALTER TABLE expenses ADD COLUMN x int",
        ):
            with self.assertRaises(SqlRejected, msg=sql):
                validate(sql)

    def test_rejects_multiple_statements(self):
        with self.assertRaises(SqlRejected):
            validate("SELECT 1; DROP TABLE expenses")

    def test_rejects_write_hidden_in_cte(self):
        with self.assertRaises(SqlRejected):
            validate("WITH d AS (DELETE FROM expenses RETURNING *) SELECT * FROM d")

    def test_rejects_select_into(self):
        with self.assertRaises(SqlRejected):
            validate("SELECT * INTO copy FROM expenses")

    def test_rejects_dangerous_functions_and_catalogs(self):
        for sql in (
            "SELECT pg_sleep(100)",
            "SELECT pg_read_file('/etc/passwd')",
            "SELECT * FROM information_schema.tables",
            "SELECT set_config('x', 'y', false)",
        ):
            with self.assertRaises(SqlRejected, msg=sql):
                validate(sql)

    def test_rejects_unbalanced_quote_and_dollar_quote(self):
        for sql in ("SELECT 'abc", "SELECT $$x$$", "SELECT /* open"):
            with self.assertRaises(SqlRejected, msg=sql):
                validate(sql)

    def test_rejects_empty(self):
        with self.assertRaises(SqlRejected):
            validate("  ;  ")


class CacheTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = Path(self._dir.name) / "cache.json"
        self.clock = Clock("Asia/Dhaka")

    def test_put_get_delete(self):
        cache = JsonCache(self.path, 10, self.clock)
        self.assertIsNone(cache.get("k"))
        cache.put("k", "question", "SELECT 1")
        self.assertEqual(cache.get("k"), "SELECT 1")
        cache.delete("k")
        self.assertIsNone(cache.get("k"))

    def test_file_is_readable_json_with_question(self):
        JsonCache(self.path, 10, self.clock).put("k", "how much?", "SELECT 1")
        entry = json.loads(self.path.read_text())["entries"]["k"]
        self.assertEqual(entry["question"], "how much?")
        self.assertIn("created_at", entry)

    def test_corrupt_file_behaves_like_empty(self):
        self.path.write_text("{not json")
        cache = JsonCache(self.path, 10, self.clock)
        self.assertIsNone(cache.get("k"))
        cache.put("k", "q", "SELECT 1")
        self.assertEqual(cache.get("k"), "SELECT 1")

    def test_evicts_oldest_when_full(self):
        cache = JsonCache(self.path, 2, self.clock)
        for index in range(3):
            cache.put(f"k{index}", "q", f"SELECT {index}")
        entries = json.loads(self.path.read_text())["entries"]
        self.assertEqual(len(entries), 2)
        self.assertIn("k2", entries)

    def test_key_changes_with_schema_or_date(self):
        base = build_key("q", "schema-a", "model", "2026-11-09")
        self.assertNotEqual(base, build_key("q", "schema-b", "model", "2026-11-09"))
        self.assertNotEqual(base, build_key("q", "schema-a", "model", "2026-11-10"))
        self.assertEqual(base, build_key("q", "schema-a", "model", "2026-11-09"))


if __name__ == "__main__":
    unittest.main()
