import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from expense_agent.llm import LLMError, OllamaClient, Sampling

SAMPLING = Sampling(0, 20, 0.8, 64)


class OllamaStreamingTests(unittest.TestCase):
    def test_stream_passes_options_and_yields_content_only(self):
        with patch("expense_agent.llm.ollama.Client") as backend:
            backend.return_value.chat.return_value = iter([
                SimpleNamespace(message=SimpleNamespace(content="Hello ")),
                SimpleNamespace(message=SimpleNamespace(content="")),
                SimpleNamespace(message=SimpleNamespace(content="world")),
            ])
            client = OllamaClient("test-model", "http://localhost:11434", 4096, "30m")
            self.assertEqual(list(client.chat_stream([], SAMPLING)), ["Hello ", "world"])
            kwargs = backend.return_value.chat.call_args.kwargs
            self.assertTrue(kwargs["stream"])
            self.assertEqual(kwargs["model"], "test-model")
            self.assertEqual(kwargs["options"]["num_ctx"], 4096)
            self.assertEqual(kwargs["options"]["temperature"], 0)
            self.assertEqual(kwargs["keep_alive"], "30m")

    def test_iterator_network_errors_are_wrapped(self):
        def broken():
            yield SimpleNamespace(message=SimpleNamespace(content="partial"))
            raise OSError("disconnected")

        with patch("expense_agent.llm.ollama.Client") as backend:
            backend.return_value.chat.return_value = broken()
            stream = OllamaClient("test", "localhost", 4096, "30m").chat_stream([], SAMPLING)
            self.assertEqual(next(stream), "partial")
            with self.assertRaises(LLMError):
                next(stream)


class CliStreamingTests(unittest.TestCase):
    def test_default_streaming_and_buffered_flag(self):
        from expense_agent.cli import _parse_args
        self.assertFalse(_parse_args([]).no_stream)
        self.assertTrue(_parse_args(["--no-stream", "total spend"]).no_stream)

    def test_streamed_answer_is_printed_once(self):
        from expense_agent.cli import _run_request
        from expense_agent.orchestrator import AgentResult
        agent = Mock()

        def run(question, on_token=None):
            on_token("You spent ")
            on_token("810.")
            return AgentResult(True, "You spent 810.", "SELECT 1", "test", 3, False)

        agent.run.side_effect = run
        with patch("sys.stdout", new_callable=StringIO) as output:
            _run_request(agent, "total spend", stream=True)
        self.assertEqual(output.getvalue().count("You spent 810."), 1)
        self.assertIn("SQL: SELECT 1", output.getvalue())

    def test_buffered_mode_and_rejection_print_answer(self):
        from expense_agent.cli import _run_request
        from expense_agent.orchestrator import AgentResult
        for stream in (True, False):
            agent = Mock()
            agent.run.return_value = AgentResult(False, "Expense requests only.", None, "test", 0, False)
            with patch("sys.stdout", new_callable=StringIO) as output:
                result = _run_request(agent, "weather", stream=stream)
            self.assertFalse(result.ok)
            self.assertEqual(output.getvalue().count("Expense requests only."), 1)
            if not stream:
                agent.run.assert_called_once_with("weather")

    def test_partial_stream_prints_failure_after_partial_answer(self):
        from expense_agent.cli import _run_request
        from expense_agent.orchestrator import AgentResult
        agent = Mock()

        def run(question, on_token=None):
            on_token("Partial answer")
            return AgentResult(False, "Answer stream interrupted.", "SELECT 1", "test", 0, False)

        agent.run.side_effect = run
        with patch("sys.stdout", new_callable=StringIO) as output:
            _run_request(agent, "total spend", stream=True)
        self.assertIn("Partial answer\nAnswer stream interrupted.", output.getvalue())
