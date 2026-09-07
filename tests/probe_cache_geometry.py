"""Bounded live cold/repeat/tool-followup/branch smoke. No customer content."""
import json
import time
import urllib.request
from pathlib import Path

base = 'http://192.168.68.113:30000'
key = Path('/Users/rajiv/.config/ornith15/api-key').read_text().strip()
headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
           'anthropic-version': '2023-06-01'}
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def call(messages):
    body = dict(model='qwen3.8-flash-next', max_tokens=32, temperature=0, messages=messages)
    start = time.monotonic()
    with opener.open(urllib.request.Request(base+'/v1/messages', data=json.dumps(body).encode(),
                                          headers=headers), timeout=120) as response:
        result = json.load(response)
    content = ''.join(b.get('text', '') for b in result['content'] if b['type'] == 'text').strip()
    return dict(seconds=round(time.monotonic()-start, 3), text=content,
                usage=result.get('usage'), stop=result.get('stop_reason'))

prefix = 'Synthetic cache geometry probe ' + str(time.time_ns()) + '\n'
prefix += '\n'.join(f'Record {i}: amber cedar river stone.' for i in range(1100))
question = '\nIgnore the records. Reply with exactly CACHE_OK and no other text.'
messages = [dict(role='user', content=prefix+question)]
results = []
for label, prompt in [('cold', messages), ('repeat', messages),
    ('tool_followup', messages + [dict(role='assistant', content=[
        dict(type='tool_use', id='probe_tool_1', name='synthetic_lookup', input={})]),
        dict(role='user', content=[dict(type='tool_result', tool_use_id='probe_tool_1',
            content='Synthetic lookup complete. Reply with exactly CACHE_OK.')])]),
    ('shorter_branch', [dict(role='user', content=prefix[:len(prefix)//2]+question)])]:
    result = call(prompt)
    print(json.dumps(dict(case=label, **result)), flush=True)
    assert result['text'] == 'CACHE_OK', result
    assert result['stop'] == 'end_turn', result
    results.append(result)
assert results[0]['text'] == results[1]['text']
