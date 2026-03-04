from core.prompts import (
    generate_shared_prompt,
    get_prompt_faithfulness,
    get_prompt_expressiveness,
    get_zh_natural_polish_prompt,
)
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich import box
from core.utils import *
import re
import json
console = Console()

def valid_translate_result(result: dict, required_keys: list, required_sub_keys: list):
    # Check for the required key
    if not all(key in result for key in required_keys):
        return {"status": "error", "message": f"Missing required key(s): {', '.join(set(required_keys) - set(result.keys()))}"}
    
    # Check for required sub-keys in all items
    for key in result:
        if not all(sub_key in result[key] for sub_key in required_sub_keys):
            return {"status": "error", "message": f"Missing required sub-key(s) in item {key}: {', '.join(set(required_sub_keys) - set(result[key].keys()))}"}

    return {"status": "success", "message": "Translation completed"}


def _is_chinese_target_language():
    try:
        tgt = str(load_key("target_language")).lower()
    except Exception:
        return False
    return ("中文" in tgt) or ("chinese" in tgt) or tgt.startswith("zh")


def _basic_zh_surface_cleanup(line: str) -> str:
    text = str(line or "").replace("\n", " ").strip()
    text = re.sub(r'\s*([，。！？；：、])\s*', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


_ZH_TRANSLATIONESE_PATTERNS = [
    re.compile(r'在.+(之后|以后|之前)'),
    re.compile(r'如果.+的话'),
    re.compile(r'原因是因为'),
    re.compile(r'是因为.+的原因'),
    re.compile(r'进行(分析|研究|讨论|处理|比较|测试|检查|说明|观察|评估|优化|调整)'),
    re.compile(r'对于.+来说'),
]


def _collect_zh_polish_candidates(lines):
    candidates = []
    for i, line in enumerate(lines, start=1):
        txt = _basic_zh_surface_cleanup(line)
        if not txt:
            continue
        if any(p.search(txt) for p in _ZH_TRANSLATIONESE_PATTERNS):
            candidates.append({"id": str(i), "text": txt})
    return candidates


def _llm_polish_zh_lines(lines, log_title="translate_zh_polish"):
    if not _is_chinese_target_language():
        return [_basic_zh_surface_cleanup(x) for x in lines]

    polished = [_basic_zh_surface_cleanup(x) for x in lines]
    candidates = _collect_zh_polish_candidates(polished)
    if not candidates:
        return polished

    prompt = get_zh_natural_polish_prompt(candidates_json=json.dumps(candidates, ensure_ascii=False, indent=2))

    def valid_polish(resp):
        if not isinstance(resp, dict) or "items" not in resp or not isinstance(resp["items"], list):
            return {"status": "error", "message": "Invalid polish response format"}
        for item in resp["items"]:
            if not isinstance(item, dict) or "id" not in item or "natural" not in item:
                return {"status": "error", "message": "Invalid polish item format"}
        return {"status": "success", "message": "ok"}

    try:
        resp = ask_gpt(prompt, resp_type="json", valid_def=valid_polish, log_title=log_title)
        id_to_text = {str(it["id"]).strip(): _basic_zh_surface_cleanup(it.get("natural", "")) for it in resp.get("items", [])}
        for c in candidates:
            idx = int(c["id"]) - 1
            new_text = id_to_text.get(c["id"], "").strip()
            if 0 <= idx < len(polished) and new_text:
                polished[idx] = new_text
    except Exception as e:
        console.print(f"[yellow]⚠️ Chinese naturalness polish skipped due to LLM error: {e}[/yellow]")

    return polished


def _postprocess_translation_line(line: str) -> str:
    return _basic_zh_surface_cleanup(line)

def translate_lines(lines, previous_content_prompt, after_cotent_prompt, things_to_note_prompt, summary_prompt, index = 0):
    shared_prompt = generate_shared_prompt(previous_content_prompt, after_cotent_prompt, summary_prompt, things_to_note_prompt)

    # Retry translation if the length of the original text and the translated text are not the same, or if the specified key is missing
    def retry_translation(prompt, length, step_name):
        def valid_faith(response_data):
            return valid_translate_result(response_data, [str(i) for i in range(1, length+1)], ['direct'])
        def valid_express(response_data):
            return valid_translate_result(response_data, [str(i) for i in range(1, length+1)], ['free'])
        for retry in range(3):
            if step_name == 'faithfulness':
                result = ask_gpt(prompt+retry* " ", resp_type='json', valid_def=valid_faith, log_title=f'translate_{step_name}')
            elif step_name == 'expressiveness':
                result = ask_gpt(prompt+retry* " ", resp_type='json', valid_def=valid_express, log_title=f'translate_{step_name}')
            if len(lines.split('\n')) == len(result):
                return result
            if retry != 2:
                console.print(f'[yellow]⚠️ {step_name.capitalize()} translation of block {index} failed, Retry...[/yellow]')
        raise ValueError(f'[red]❌ {step_name.capitalize()} translation of block {index} failed after 3 retries. Please check `output/gpt_log/error.json` for more details.[/red]')

    ## Step 1: Faithful to the Original Text
    prompt1 = get_prompt_faithfulness(lines, shared_prompt)
    faith_result = retry_translation(prompt1, len(lines.split('\n')), 'faithfulness')

    faith_direct_list = [_postprocess_translation_line(faith_result[i]["direct"]) for i in faith_result]
    faith_direct_list = _llm_polish_zh_lines(faith_direct_list, log_title="translate_zh_polish_direct")
    for idx, key in enumerate(faith_result):
        faith_result[key]["direct"] = faith_direct_list[idx]

    # If reflect_translate is False or not set, use faithful translation directly
    reflect_translate = load_key('reflect_translate')
    if not reflect_translate:
        # If reflect_translate is False or not set, use faithful translation directly
        translate_result = "\n".join([faith_result[i]["direct"].strip() for i in faith_result])
        
        table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
        table.add_column("Translations", style="bold")
        for i, key in enumerate(faith_result):
            table.add_row(f"[cyan]Origin:  {faith_result[key]['origin']}[/cyan]")
            table.add_row(f"[magenta]Direct:  {faith_result[key]['direct']}[/magenta]")
            if i < len(faith_result) - 1:
                table.add_row("[yellow]" + "-" * 50 + "[/yellow]")
        
        console.print(table)
        return translate_result, lines

    ## Step 2: Express Smoothly  
    prompt2 = get_prompt_expressiveness(faith_result, lines, shared_prompt)
    express_result = retry_translation(prompt2, len(lines.split('\n')), 'expressiveness')

    table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
    table.add_column("Translations", style="bold")
    for i, key in enumerate(express_result):
        table.add_row(f"[cyan]Origin:  {faith_result[key]['origin']}[/cyan]")
        table.add_row(f"[magenta]Direct:  {faith_result[key]['direct']}[/magenta]")
        table.add_row(f"[green]Free:    {express_result[key]['free']}[/green]")
        if i < len(express_result) - 1:
            table.add_row("[yellow]" + "-" * 50 + "[/yellow]")

    console.print(table)

    free_lines = [_postprocess_translation_line(express_result[i]["free"]) for i in express_result]
    free_lines = _llm_polish_zh_lines(free_lines, log_title="translate_zh_polish_free")
    translate_result = "\n".join(free_lines)

    if len(lines.split('\n')) != len(translate_result.split('\n')):
        console.print(Panel(f'[red]❌ Translation of block {index} failed, Length Mismatch, Please check `output/gpt_log/translate_expressiveness.json`[/red]'))
        raise ValueError(f'Origin ···{lines}···,\nbut got ···{translate_result}···')

    return translate_result, lines


if __name__ == '__main__':
    # test e.g.
    lines = '''All of you know Andrew Ng as a famous computer science professor at Stanford.
He was really early on in the development of neural networks with GPUs.
Of course, a creator of Coursera and popular courses like deeplearning.ai.
Also the founder and creator and early lead of Google Brain.'''
    previous_content_prompt = None
    after_cotent_prompt = None
    things_to_note_prompt = None
    summary_prompt = None
    translate_lines(lines, previous_content_prompt, after_cotent_prompt, things_to_note_prompt, summary_prompt)
