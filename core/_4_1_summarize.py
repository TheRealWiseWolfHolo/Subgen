import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from core.prompts import get_summary_prompt, get_person_name_filter_prompt
import pandas as pd
from core.utils import *
from core.utils.models import _3_2_SPLIT_BY_MEANING, _4_1_TERMINOLOGY

CUSTOM_TERMS_PATH = 'custom_terms.xlsx'
TERMINOLOGY_LIBRARY_DIR = Path("history/terminology_profiles")
_SELECTED_PROFILE_PATH = None

def _safe_load_key(key, default):
    try:
        return load_key(key)
    except Exception:
        return default

def reset_selected_profile():
    """Reset cached terminology profile selection in current process."""
    global _SELECTED_PROFILE_PATH
    _SELECTED_PROFILE_PATH = None

def combine_chunks(limit=None):
    """Combine the text chunks identified by whisper into a single long text"""
    with open(_3_2_SPLIT_BY_MEANING, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    cleaned_sentences = [line.strip() for line in sentences]
    combined_text = ' '.join(cleaned_sentences)
    if limit is None:
        return combined_text
    return combined_text[:limit]

def _normalize_terms(terms):
    normalized = []
    for term in terms or []:
        if not isinstance(term, dict):
            continue
        src = str(term.get("src", "")).strip()
        if not src:
            continue
        normalized.append({
            "src": src,
            "tgt": str(term.get("tgt", "")).strip(),
            "note": str(term.get("note", "")).strip(),
        })
    return normalized

def _merge_terms(*term_lists):
    merged = []
    seen = set()
    for terms in term_lists:
        for term in _normalize_terms(terms):
            key = term["src"].lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(term)
    return merged

def _normalize_frequency_map(freq_map):
    normalized = {}
    if not isinstance(freq_map, dict):
        return normalized
    for k, v in freq_map.items():
        key = str(k).strip().lower()
        if not key:
            continue
        try:
            count = int(v)
        except Exception:
            count = 0
        if count > 0:
            normalized[key] = count
    return normalized

def _load_profile(profile_path: Path):
    if not profile_path or not profile_path.exists():
        return [], {}
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            terms = _normalize_terms(payload.get("terms", []))
            freq_map = _normalize_frequency_map(payload.get("term_frequency", {}))
            return terms, freq_map
        if isinstance(payload, list):
            return _normalize_terms(payload), {}
        return [], {}
    except Exception as e:
        rprint(f"[yellow]⚠️ Failed to load terminology profile `{profile_path}`: {e}[/yellow]")
        return [], {}

def _save_profile_terms(profile_path: Path, terms, term_frequency=None):
    if not profile_path:
        return
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "terms": _normalize_terms(terms),
                "term_frequency": _normalize_frequency_map(term_frequency or {}),
            },
            f,
            ensure_ascii=False,
            indent=4,
        )
    rprint(f"💾 Terminology profile saved to → `{profile_path}`")

def _bump_term_frequency(freq_map, terms):
    freq = _normalize_frequency_map(freq_map)
    for term in _normalize_terms(terms):
        key = term["src"].lower()
        if not key:
            continue
        freq[key] = freq.get(key, 0) + 1
    return freq

def _prune_terms_by_frequency(terms, freq_map, keep_top_n):
    if keep_top_n <= 0:
        return _normalize_terms(terms), _normalize_frequency_map(freq_map)

    terms = _normalize_terms(terms)
    if len(terms) <= keep_top_n:
        return terms, _normalize_frequency_map(freq_map)

    freq = _normalize_frequency_map(freq_map)
    sorted_terms = sorted(
        terms,
        key=lambda t: (-freq.get(t["src"].lower(), 0), t["src"].lower())
    )
    kept_terms = sorted_terms[:keep_top_n]
    kept_keys = {t["src"].lower() for t in kept_terms}
    kept_freq = {k: freq.get(k, 0) for k in kept_keys if freq.get(k, 0) > 0}
    rprint(
        f"[cyan]🧹 Pruned terminology profile to top {keep_top_n} terms by frequency "
        f"({len(terms)} -> {len(kept_terms)}).[/cyan]"
    )
    return kept_terms, kept_freq

def _list_profiles():
    TERMINOLOGY_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(TERMINOLOGY_LIBRARY_DIR.glob("*.json"))

def list_terminology_profile_names():
    """List available terminology profile names without extension."""
    return [p.stem for p in _list_profiles()]

def _name_to_profile_path(name: str) -> Path:
    normalized = str(name).strip().replace("/", "_").replace("\\", "_")
    if not normalized:
        raise ValueError("Profile name cannot be empty")
    if not normalized.endswith(".json"):
        normalized += ".json"
    return TERMINOLOGY_LIBRARY_DIR / normalized

def set_selected_profile(profile_name: str):
    """Select (and create if missing) a terminology profile by name."""
    global _SELECTED_PROFILE_PATH
    profile_path = _name_to_profile_path(profile_name)
    if not profile_path.exists():
        _save_profile_terms(profile_path, [])
    _SELECTED_PROFILE_PATH = profile_path
    return _SELECTED_PROFILE_PATH

def _pick_terminology_profile_cli() -> Path:
    while True:
        profiles = _list_profiles()
        print("Pick terminology json:")
        for i, profile in enumerate(profiles, start=1):
            print(f"{i}. {profile.stem}")
        new_idx = len(profiles) + 1
        print(f"{new_idx}. New (Please name)")

        try:
            choice_raw = input("User: ").strip()
        except EOFError:
            # Non-interactive stdin fallback
            default_profile = _name_to_profile_path("default")
            if not default_profile.exists():
                _save_profile_terms(default_profile, [])
            print(f"No interactive input available, fallback to profile: {default_profile.stem}")
            return default_profile
        if not choice_raw.isdigit():
            print("Please enter a valid number.")
            continue
        choice = int(choice_raw)
        if 1 <= choice <= len(profiles):
            return profiles[choice - 1]
        if choice == new_idx:
            while True:
                name = input("Please name the new json: ").strip()
                if not name:
                    print("Name cannot be empty.")
                    continue
                profile_path = _name_to_profile_path(name)
                if profile_path.exists():
                    print("This name already exists. Choose another name.")
                    continue
                _save_profile_terms(profile_path, [])
                return profile_path
        print("Please choose one of the listed options.")

def _load_custom_terms():
    if not os.path.exists(CUSTOM_TERMS_PATH):
        return []
    custom_terms_df = pd.read_excel(CUSTOM_TERMS_PATH)
    return _normalize_terms(
        [
            {
                "src": str(row.iloc[0]),
                "tgt": str(row.iloc[1]),
                "note": str(row.iloc[2]),
            }
            for _, row in custom_terms_df.iterrows()
        ]
    )

def _extract_person_name_candidates(text: str):
    """Heuristic candidate recall for person-like names (high recall, low precision)."""
    if not text:
        return [], set()

    stopwords = {
        "The", "This", "That", "These", "Those", "And", "But", "So", "In", "On", "At",
        "Of", "To", "For", "From", "With", "Without", "It", "Its", "You", "We", "They",
        "He", "She", "I", "A", "An", "As", "By", "Or", "If", "When", "While", "Then",
        "However", "Therefore", "Meanwhile", "Anyway", "Also", "Because", "What", "Who",
        "How", "Where", "Why", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    }
    token = r"[A-Z][a-z]+(?:['-][A-Z]?[a-z]+)?"
    surname_particles = r"(?:[Vv]an|[Vv]on|[Dd]e|[Dd]el|[Dd]a|[Dd]i|[Ll]a|[Ll]e|[Aa]l|[Bb]in)"
    full_name_pattern = re.compile(
        rf"\b{token}(?:\s+(?:{surname_particles}\s+)?{token}){{1,3}}\b"
    )
    titled_name_pattern = re.compile(
        rf"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+{token}(?:\s+{token}){{0,2}}\b"
    )
    single_name_pattern = re.compile(rf"\b{token}\b")

    candidates = []
    high_conf = set()

    for pattern in (titled_name_pattern, full_name_pattern):
        for m in pattern.finditer(text):
            name = re.sub(r"\s+", " ", m.group(0)).strip()
            first = name.replace(".", "").split()[0]
            if first in stopwords:
                continue
            candidates.append(name)
            high_conf.add(name.lower())

    singles = [m.group(0).strip() for m in single_name_pattern.finditer(text)]
    single_counts = Counter(s for s in singles if s and s not in stopwords)
    for name, count in single_counts.items():
        # Include repeated single-token names (common in creator/game reviews).
        if count >= 2 and len(name) >= 3:
            candidates.append(name)
            if count >= 4:
                high_conf.add(name.lower())

    normalized_names = [term["src"] for term in _normalize_terms([{"src": n, "tgt": "", "note": ""} for n in candidates])]
    return normalized_names, high_conf


def _filter_person_names_with_llm(source_text: str, candidates, high_conf):
    """LLM-based strict filtering to reduce false positives in name extraction."""
    if not candidates:
        return []

    max_candidates = 120
    candidates = candidates[:max_candidates]
    prompt_text = source_text[:12000]
    filter_prompt = get_person_name_filter_prompt(prompt_text, candidates)

    def valid_name_filter(resp):
        if not isinstance(resp, dict) or "names" not in resp:
            return {"status": "error", "message": "Invalid name filter response format"}
        if not isinstance(resp["names"], list):
            return {"status": "error", "message": "Invalid `names` type"}
        return {"status": "success", "message": "ok"}

    try:
        filtered = ask_gpt(
            filter_prompt,
            resp_type="json",
            valid_def=valid_name_filter,
            log_title="person_name_filter",
        )
        keep_set = {str(x).strip() for x in filtered.get("names", []) if str(x).strip()}
        keep_set = {x for x in keep_set if x in set(candidates)}
    except Exception as e:
        rprint(f"[yellow]⚠️ Person-name LLM filter failed, fallback to conservative heuristic: {e}[/yellow]")
        keep_set = {name for name in candidates if name.lower() in high_conf}

    terms = [
        {
            "src": name,
            "tgt": name,
            "note": "Person name (LLM-filtered, keep original)"
        }
        for name in candidates
        if name in keep_set
    ]
    return _normalize_terms(terms)


def _extract_person_name_terms(text: str):
    candidates, high_conf = _extract_person_name_candidates(text)
    return _filter_person_names_with_llm(text, candidates, high_conf)

def search_things_to_note_in_prompt(sentence):
    """Search for terms to note in the given sentence"""
    with open(_4_1_TERMINOLOGY, 'r', encoding='utf-8') as file:
        things_to_note = json.load(file)
    things_to_note_list = [term['src'] for term in things_to_note['terms'] if term['src'].lower() in sentence.lower()]
    if things_to_note_list:
        prompt = '\n'.join(
            f'{i+1}. "{term["src"]}": "{term["tgt"]}",'
            f' meaning: {term["note"]}'
            for i, term in enumerate(things_to_note['terms'])
            if term['src'] in things_to_note_list
        )
        return prompt
    else:
        return None

def get_summary(interactive_select=None):
    global _SELECTED_PROFILE_PATH
    full_content = combine_chunks(limit=None)
    src_content = combine_chunks(limit=load_key('summary_length'))
    if interactive_select is None:
        interactive_select = sys.stdin.isatty()

    if interactive_select and _SELECTED_PROFILE_PATH is None:
        _SELECTED_PROFILE_PATH = _pick_terminology_profile_cli()

    profile_terms, profile_freq = _load_profile(_SELECTED_PROFILE_PATH) if _SELECTED_PROFILE_PATH else ([], {})
    custom_terms = _load_custom_terms()
    person_name_terms = _extract_person_name_terms(full_content)
    existing_terms = _merge_terms(profile_terms, custom_terms, person_name_terms)

    if custom_terms:
        rprint(f"📖 Custom Terms Loaded: {len(custom_terms)} terms")
    if _SELECTED_PROFILE_PATH:
        rprint(f"📚 Using terminology profile: `{_SELECTED_PROFILE_PATH}` ({len(profile_terms)} terms)")

    summary_prompt = get_summary_prompt(src_content, {"terms": existing_terms})
    rprint("📝 Summarizing and extracting terminology ...")
    
    def valid_summary(response_data):
        required_keys = {'src', 'tgt', 'note'}
        if 'terms' not in response_data:
            return {"status": "error", "message": "Invalid response format"}
        for term in response_data['terms']:
            if not all(key in term for key in required_keys):
                return {"status": "error", "message": "Invalid response format"}   
        return {"status": "success", "message": "Summary completed"}

    summary = ask_gpt(summary_prompt, resp_type='json', valid_def=valid_summary, log_title='summary')
    summary_terms = _normalize_terms(summary.get("terms", []))
    merged_terms = _merge_terms(profile_terms, summary_terms, custom_terms, person_name_terms)
    summary_output = {
        "theme": summary.get("theme", ""),
        "terms": merged_terms
    }
    
    with open(_4_1_TERMINOLOGY, 'w', encoding='utf-8') as f:
        json.dump(summary_output, f, ensure_ascii=False, indent=4)

    rprint(f'💾 Summary log saved to → `{_4_1_TERMINOLOGY}`')
    if _SELECTED_PROFILE_PATH:
        merged_profile_terms = _merge_terms(profile_terms, summary_terms, person_name_terms)
        updated_profile_freq = _bump_term_frequency(profile_freq, summary_terms + person_name_terms)
        keep_top_n = int(_safe_load_key("terminology_profile_keep_top_n", 0))
        merged_profile_terms, updated_profile_freq = _prune_terms_by_frequency(
            merged_profile_terms, updated_profile_freq, keep_top_n
        )
        _save_profile_terms(_SELECTED_PROFILE_PATH, merged_profile_terms, updated_profile_freq)

if __name__ == '__main__':
    get_summary(interactive_select=True)
