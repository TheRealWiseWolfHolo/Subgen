import pandas as pd
from typing import List, Tuple
import concurrent.futures
import re
from difflib import SequenceMatcher

from core._3_2_split_meaning import split_sentence
from core.prompts import get_align_prompt, get_boundary_rebalance_prompt
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from core.utils import *
from core.utils.models import *
console = Console()

# ! You can modify your own weights here
# Chinese and Japanese 2.5 characters, Korean 2 characters, Thai 1.5 characters, full-width symbols 2 characters, other English-based and half-width symbols 1 character
def calc_len(text: str) -> float:
    text = str(text) # force convert
    def char_weight(char):
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:  # Chinese and Japanese
            return 1.75
        elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:  # Korean
            return 1.5
        elif 0x0E00 <= code <= 0x0E7F:  # Thai
            return 1
        elif 0xFF01 <= code <= 0xFF5E:  # full-width symbols
            return 1.75
        else:  # other characters (e.g. English and half-width symbols)
            return 1

    return sum(char_weight(char) for char in text)


def _safe_load_key(key, default):
    try:
        return load_key(key)
    except Exception:
        return default


def _normalize_compare_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    return text


def _boundary_rebalance_enabled() -> bool:
    return bool(_safe_load_key("subtitle_boundary_rebalance_with_llm", True))


def _boundary_rebalance_max_pairs() -> int:
    return int(_safe_load_key("subtitle_boundary_rebalance_max_pairs", 20))


def _looks_like_bad_boundary(src_1: str, src_2: str, tr_1: str, tr_2: str) -> bool:
    s1 = str(src_1).strip()
    s2 = str(src_2).strip()
    t2 = str(tr_2).strip()
    if not s1 or not s2:
        return False

    # Common case: first block looks unfinished while second is long enough to carry two clauses.
    no_tail_punct = not re.search(r"[.!?。！？…]$|[)\\]\"'”）]$", s1)
    second_has_inner_pause = bool(re.search(r"[,;:，；：]", s2)) or len(s2.split()) >= 8

    # Translation side signal: second translated block clearly contains connective progression.
    tr_connective = bool(re.search(r"(然后|之后|接着|接下来|再|并且|而且|随后)", t2))
    return (no_tail_punct and second_has_inner_pause) or tr_connective


def _validate_rebalanced_pair(orig_pair, cand_pair) -> bool:
    if len(cand_pair) != 4:
        return False
    if any(not str(x).strip() for x in cand_pair):
        return False

    o_src = _normalize_compare_text(orig_pair[0] + orig_pair[1])
    c_src = _normalize_compare_text(cand_pair[0] + cand_pair[1])
    o_tr = _normalize_compare_text(orig_pair[2] + orig_pair[3])
    c_tr = _normalize_compare_text(cand_pair[2] + cand_pair[3])
    if not o_src or not o_tr or not c_src or not c_tr:
        return False

    # Guard rails: keep pair-level meaning and length almost unchanged.
    src_sim = SequenceMatcher(None, o_src, c_src).ratio()
    tr_sim = SequenceMatcher(None, o_tr, c_tr).ratio()
    if src_sim < 0.90 or tr_sim < 0.75:
        return False

    src_len_delta = abs(len(c_src) - len(o_src)) / max(1, len(o_src))
    tr_len_delta = abs(len(c_tr) - len(o_tr)) / max(1, len(o_tr))
    if src_len_delta > 0.12 or tr_len_delta > 0.18:
        return False

    return True


def rebalance_adjacent_boundaries_with_llm(src_lines: List[str], tr_lines: List[str]) -> Tuple[List[str], List[str]]:
    if len(src_lines) < 2 or len(tr_lines) < 2:
        return src_lines, tr_lines
    if not _boundary_rebalance_enabled():
        return src_lines, tr_lines

    max_pairs = max(0, _boundary_rebalance_max_pairs())
    if max_pairs == 0:
        return src_lines, tr_lines

    candidates = []
    for i in range(min(len(src_lines), len(tr_lines)) - 1):
        if _looks_like_bad_boundary(src_lines[i], src_lines[i + 1], tr_lines[i], tr_lines[i + 1]):
            candidates.append(i)
            if len(candidates) >= max_pairs:
                break

    if not candidates:
        return src_lines, tr_lines

    fixed_count = 0
    for i in candidates:
        src_1 = str(src_lines[i]).strip()
        src_2 = str(src_lines[i + 1]).strip()
        tr_1 = str(tr_lines[i]).strip()
        tr_2 = str(tr_lines[i + 1]).strip()

        prompt = get_boundary_rebalance_prompt(src_1, src_2, tr_1, tr_2)

        def valid_rebalance(resp):
            required = ("source_1", "source_2", "target_1", "target_2")
            for key in required:
                if key not in resp:
                    return {"status": "error", "message": f"Missing key: {key}"}
                if not str(resp[key]).strip():
                    return {"status": "error", "message": f"Empty key: {key}"}
            return {"status": "success", "message": "ok"}

        try:
            resp = ask_gpt(
                prompt,
                resp_type='json',
                valid_def=valid_rebalance,
                log_title='subtitle_boundary_rebalance'
            )
            cand_pair = (
                str(resp["source_1"]).strip(),
                str(resp["source_2"]).strip(),
                str(resp["target_1"]).strip(),
                str(resp["target_2"]).strip(),
            )
            orig_pair = (src_1, src_2, tr_1, tr_2)
            if _validate_rebalanced_pair(orig_pair, cand_pair):
                src_lines[i], src_lines[i + 1], tr_lines[i], tr_lines[i + 1] = cand_pair
                fixed_count += 1
        except Exception:
            continue

    if fixed_count:
        console.print(f"[cyan]🧩 Rebalanced subtitle boundaries for {fixed_count} adjacent pair(s).[/cyan]")
    return src_lines, tr_lines

def align_subs(src_sub: str, tr_sub: str, src_part: str) -> Tuple[List[str], List[str], str]:
    align_prompt = get_align_prompt(src_sub, tr_sub, src_part)
    
    def valid_align(response_data):
        if 'align' not in response_data:
            return {"status": "error", "message": "Missing required key: `align`"}
        if len(response_data['align']) < 2:
            return {"status": "error", "message": "Align does not contain more than 1 part as expected!"}
        return {"status": "success", "message": "Align completed"}
    parsed = ask_gpt(align_prompt, resp_type='json', valid_def=valid_align, log_title='align_subs')
    align_data = parsed['align']
    src_parts = src_part.split('\n')
    tr_parts = [item[f'target_part_{i+1}'].strip() for i, item in enumerate(align_data)]
    
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)
    tr_remerged = joiner.join(tr_parts)
    
    table = Table(title="🔗 Aligned parts")
    table.add_column("Language", style="cyan")
    table.add_column("Parts", style="magenta")
    table.add_row("SRC_LANG", "\n".join(src_parts))
    table.add_row("TARGET_LANG", "\n".join(tr_parts))
    console.print(table)
    
    return src_parts, tr_parts, tr_remerged

def split_align_subs(src_lines: List[str], tr_lines: List[str]):
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    remerged_tr_lines = tr_lines.copy()
    
    to_split = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        src, tr = str(src), str(tr)
        if len(src) > MAX_SUB_LENGTH or calc_len(tr) * TARGET_SUB_MULTIPLIER > MAX_SUB_LENGTH:
            to_split.append(i)
            table = Table(title=f"📏 Line {i} needs to be split")
            table.add_column("Type", style="cyan")
            table.add_column("Content", style="magenta")
            table.add_row("Source Line", src)
            table.add_row("Target Line", tr)
            console.print(table)
    
    @except_handler("Error in split_align_subs")
    def process(i):
        split_src = split_sentence(src_lines[i], num_parts=2).strip()
        src_parts, tr_parts, tr_remerged = align_subs(src_lines[i], tr_lines[i], split_src)
        src_lines[i] = src_parts
        tr_lines[i] = tr_parts
        remerged_tr_lines[i] = tr_remerged
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
        executor.map(process, to_split)
    
    # Flatten `src_lines` and `tr_lines`
    src_lines = [item for sublist in src_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    tr_lines = [item for sublist in tr_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    
    return src_lines, tr_lines, remerged_tr_lines

def split_for_sub_main():
    console.print("[bold green]🚀 Start splitting subtitles...[/bold green]")
    
    df = pd.read_excel(_4_2_TRANSLATION)
    src = df['Source'].tolist()
    trans = df['Translation'].tolist()
    
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    
    for attempt in range(3):  # 多次切割
        console.print(Panel(f"🔄 Split attempt {attempt + 1}", expand=False))
        split_src, split_trans, remerged = split_align_subs(src.copy(), trans)
        
        # 检查是否所有字幕都符合长度要求
        if all(len(src) <= MAX_SUB_LENGTH for src in split_src) and \
           all(calc_len(tr) * TARGET_SUB_MULTIPLIER <= MAX_SUB_LENGTH for tr in split_trans):
            break
        
        # 更新源数据继续下一轮分割
        src, trans = split_src, split_trans

    # 确保二者有相同的长度，防止报错
    if len(src) > len(remerged):
        remerged += [None] * (len(src) - len(remerged))
    elif len(remerged) > len(src):
        src += [None] * (len(remerged) - len(src))
    
    split_src, split_trans = rebalance_adjacent_boundaries_with_llm(split_src, split_trans)
    pd.DataFrame({'Source': split_src, 'Translation': split_trans}).to_excel(_5_SPLIT_SUB, index=False)
    pd.DataFrame({'Source': src, 'Translation': remerged}).to_excel(_5_REMERGED, index=False)

if __name__ == '__main__':
    split_for_sub_main()
