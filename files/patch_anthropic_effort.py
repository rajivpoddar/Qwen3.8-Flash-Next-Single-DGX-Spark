"""Map Claude effort levels to the Qwen Next checkpoint's supported levels."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE / "anthropic_serving_patched.py.orig").read_text()
anchor = "            req.reasoning_effort = output_config.effort\n"
if source.count(anchor) != 1:
    raise SystemExit("Anthropic effort patch anchor drift; refusing patch")
replacement = anchor + '''
        # Qwen Next accepts low, medium and xhigh only. Keep other models intact.
        if anthropic_request.model == "qwen3.8-flash-next":
            effort = output_config.effort if output_config else None
            req.reasoning_effort = {
                None: "low", "low": "low", "medium": "medium",
                "high": "xhigh", "xhigh": "xhigh", "max": "xhigh",
                "ultracode": "xhigh",
            }[effort]
            req.chat_template_kwargs = dict(req.chat_template_kwargs or {})
            req.chat_template_kwargs["reasoning_effort"] = req.reasoning_effort
            if effort is None:
                # Use a valid template effort value, but explicitly bypass thinking.
                req.chat_template_kwargs["enable_thinking"] = False
'''
patched = source.replace(anchor, replacement)
compile(patched, "anthropic_serving_patched.py", "exec")
(HERE / "anthropic_serving_patched.py").write_text(patched)
protocol = (HERE / "anthropic_protocol_patched.py.orig").read_text()
anchor = 'class AnthropicMessagesRequest(BaseModel):\n    """Anthropic Messages API request"""\n'
if protocol.count(anchor) != 1:
    raise SystemExit("Anthropic protocol patch anchor drift; refusing patch")
protocol = protocol.replace(anchor, anchor + '''

    @model_validator(mode="before")
    @classmethod
    def normalize_qwen_next_ultracode(cls, data):
        # Normalize before AnthropicOutputConfig validates its literal values.
        # Do not broaden accepted effort values for other models.
        if isinstance(data, dict) and data.get("model") == "qwen3.8-flash-next":
            config = data.get("output_config")
            if isinstance(config, dict) and config.get("effort") == "ultracode":
                data = dict(data)
                data["output_config"] = dict(config, effort="xhigh")
        return data
''')
compile(protocol, "anthropic_protocol_patched.py", "exec")
(HERE / "anthropic_protocol_patched.py").write_text(protocol)
print("Anthropic Qwen Next effort mapping generated")
