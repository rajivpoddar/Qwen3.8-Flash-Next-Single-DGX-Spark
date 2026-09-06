"""Pin this deployment's Claude requests to low effort, including auxiliary turns."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE / "anthropic_serving_patched.py.orig").read_text()
anchor = "            req.reasoning_effort = output_config.effort\n"
if source.count(anchor) != 1:
    raise SystemExit("Anthropic effort patch anchor drift; refusing patch")
replacement = anchor + '''
        # HeyDonna Qwen Next profile: enforce low even when Claude auxiliary
        # requests inherit high from settings. Do not change other models.
        if anthropic_request.model == "qwen3.8-flash-next":
            req.reasoning_effort = "low"
            req.chat_template_kwargs = dict(req.chat_template_kwargs or {})
            req.chat_template_kwargs["reasoning_effort"] = "low"
'''
patched = source.replace(anchor, replacement)
compile(patched, "anthropic_serving_patched.py", "exec")
(HERE / "anthropic_serving_patched.py").write_text(patched)
print("Anthropic Qwen Next effort pinned to low")
