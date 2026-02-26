import json
import os
import sys
from pathlib import Path
from core.prompts import get_summary_prompt
import pandas as pd
from core.utils import *
from core.utils.models import _3_2_SPLIT_BY_MEANING, _4_1_TERMINOLOGY

CUSTOM_TERMS_PATH = 'custom_terms.xlsx'
TERMINOLOGY_LIBRARY_DIR = Path("history/terminology_profiles")
_SELECTED_PROFILE_PATH = None

def reset_selected_profile():
    """Reset cached terminology profile selection in current process."""
    global _SELECTED_PROFILE_PATH
    _SELECTED_PROFILE_PATH = None

def combine_chunks():
    """Combine the text chunks identified by whisper into a single long text"""
    with open(_3_2_SPLIT_BY_MEANING, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    cleaned_sentences = [line.strip() for line in sentences]
    combined_text = ' '.join(cleaned_sentences)
    return combined_text[:load_key('summary_length')]  #! Return only the first x characters

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

def _load_profile_terms(profile_path: Path):
    if not profile_path or not profile_path.exists():
        return []
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return _normalize_terms(payload.get("terms", []))
        if isinstance(payload, list):
            return _normalize_terms(payload)
        return []
    except Exception as e:
        rprint(f"[yellow]⚠️ Failed to load terminology profile `{profile_path}`: {e}[/yellow]")
        return []

def _save_profile_terms(profile_path: Path, terms):
    if not profile_path:
        return
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump({"terms": _normalize_terms(terms)}, f, ensure_ascii=False, indent=4)
    rprint(f"💾 Terminology profile saved to → `{profile_path}`")

def _list_profiles():
    TERMINOLOGY_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(TERMINOLOGY_LIBRARY_DIR.glob("*.json"))

def _name_to_profile_path(name: str) -> Path:
    normalized = str(name).strip().replace("/", "_").replace("\\", "_")
    if not normalized:
        raise ValueError("Profile name cannot be empty")
    if not normalized.endswith(".json"):
        normalized += ".json"
    return TERMINOLOGY_LIBRARY_DIR / normalized

def _pick_terminology_profile_cli() -> Path:
    while True:
        profiles = _list_profiles()
        print("Pick terminology json:")
        for i, profile in enumerate(profiles, start=1):
            print(f"{i}. {profile.stem}")
        new_idx = len(profiles) + 1
        print(f"{new_idx}. New (Please name)")

        choice_raw = input("User: ").strip()
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
    src_content = combine_chunks()
    if interactive_select is None:
        interactive_select = sys.stdin.isatty()

    if interactive_select and _SELECTED_PROFILE_PATH is None:
        _SELECTED_PROFILE_PATH = _pick_terminology_profile_cli()

    profile_terms = _load_profile_terms(_SELECTED_PROFILE_PATH) if _SELECTED_PROFILE_PATH else []
    custom_terms = _load_custom_terms()
    existing_terms = _merge_terms(profile_terms, custom_terms)

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
    merged_terms = _merge_terms(profile_terms, summary_terms, custom_terms)
    summary_output = {
        "theme": summary.get("theme", ""),
        "terms": merged_terms
    }
    
    with open(_4_1_TERMINOLOGY, 'w', encoding='utf-8') as f:
        json.dump(summary_output, f, ensure_ascii=False, indent=4)

    rprint(f'💾 Summary log saved to → `{_4_1_TERMINOLOGY}`')
    if _SELECTED_PROFILE_PATH:
        _save_profile_terms(_SELECTED_PROFILE_PATH, _merge_terms(profile_terms, summary_terms))

if __name__ == '__main__':
    get_summary(interactive_select=True)
