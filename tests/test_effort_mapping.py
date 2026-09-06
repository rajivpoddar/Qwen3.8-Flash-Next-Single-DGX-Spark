"""Focused behavior tests against generated pinned-runtime adapter sources."""
import ast
from pathlib import Path
from types import SimpleNamespace as N
import unittest

FILES = Path(__file__).resolve().parents[1] / "files"


def method(filename, name):
    tree = ast.parse((FILES / filename).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    ns = dict(AnthropicCountTokensRequest=type("Count", (), {}),
              AnthropicOutputConfig=N, ChatCompletionRequest=N, AnthropicMessagesRequest=N)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), ns)
    return ns[name]


class EffortMappingTest(unittest.TestCase):
    def test_levels_and_omitted_thinking(self):
        normalize = method("anthropic_protocol_patched.py", "normalize_qwen_next_ultracode")
        apply = method("anthropic_serving_patched.py", "_handle_output_config")
        for level, expected in [(None, "low"), ("low", "low"), ("medium", "medium"),
                                ("high", "xhigh"), ("xhigh", "xhigh"),
                                ("max", "xhigh"), ("ultracode", "xhigh")]:
            with self.subTest(level=level):
                data = dict(model="qwen3.8-flash-next", output_config=dict(effort=level))
                data = normalize(None, data)
                req = N(reasoning_effort=None, chat_template_kwargs={"enable_thinking": True})
                apply(None, req, N(model=data["model"], output_config=N(format=None, **data["output_config"])))
                self.assertEqual(req.reasoning_effort, expected)
                self.assertEqual(req.chat_template_kwargs["reasoning_effort"], expected)
                self.assertEqual(req.chat_template_kwargs["enable_thinking"], level is not None)
        req = N(reasoning_effort=None, chat_template_kwargs=None)
        apply(None, req, N(model="qwen3.8-flash-next", output_config=None))
        self.assertFalse(req.chat_template_kwargs["enable_thinking"])

    def test_other_models_unchanged(self):
        normalize = method("anthropic_protocol_patched.py", "normalize_qwen_next_ultracode")
        data = dict(model="other", output_config=dict(effort="ultracode"))
        self.assertIs(normalize(None, data), data)
        req = N(reasoning_effort=None, chat_template_kwargs=None)
        method("anthropic_serving_patched.py", "_handle_output_config")(
            None, req, N(model="other", output_config=N(format=None, effort="high")))
        self.assertEqual(req.reasoning_effort, "high")
        self.assertIsNone(req.chat_template_kwargs)


if __name__ == "__main__":
    unittest.main()
