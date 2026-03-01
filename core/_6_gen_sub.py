import pandas as pd
import os
import re
from difflib import SequenceMatcher
from rich.panel import Panel
from rich.console import Console
import autocorrect_py as autocorrect
from core.utils import *
from core.utils.models import *
console = Console()

SUBTITLE_OUTPUT_CONFIGS = [ 
    ('src.srt', ['Source']),
    ('trans.srt', ['Translation']),
    ('src_trans.srt', ['Source', 'Translation']),
    ('trans_src.srt', ['Translation', 'Source'])
]

AUDIO_SUBTITLE_OUTPUT_CONFIGS = [
    ('src_subs_for_audio.srt', ['Source']),
    ('trans_subs_for_audio.srt', ['Translation'])
]

SUSPICIOUS_GAP_SECONDS = 12

def _safe_load_key(key, default):
    try:
        return load_key(key)
    except Exception:
        return default

def convert_to_srt_format(start_time, end_time):
    """Convert time (in seconds) to the format: hours:minutes:seconds,milliseconds"""
    def seconds_to_hmsm(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int(seconds * 1000) % 1000
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

    start_srt = seconds_to_hmsm(start_time)
    end_srt = seconds_to_hmsm(end_time)
    return f"{start_srt} --> {end_srt}"

def remove_punctuation(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def show_difference(str1, str2):
    """Show the difference positions between two strings"""
    min_len = min(len(str1), len(str2))
    diff_positions = []
    
    for i in range(min_len):
        if str1[i] != str2[i]:
            diff_positions.append(i)
    
    if len(str1) != len(str2):
        diff_positions.extend(range(min_len, max(len(str1), len(str2))))
    
    print("Difference positions:")
    print(f"Expected sentence: {str1}")
    print(f"Actual match: {str2}")
    print("Position markers: " + "".join("^" if i in diff_positions else " " for i in range(max(len(str1), len(str2)))))
    print(f"Difference indices: {diff_positions}")

def find_local_fuzzy_match(df_words, clean_words, sentence, start_word_idx,
                           lookahead_words=180, max_sentence_words=80,
                           min_ratio=0.90):
    """Find a near-by fuzzy match to avoid accidental jumps to repeated text later in the video."""
    sentence = sentence.strip()
    if not sentence:
        return None

    n_words = len(clean_words)
    end_search = min(n_words, start_word_idx + lookahead_words)
    sentence_len = len(sentence)
    min_target = max(1, int(sentence_len * 0.8))
    max_target = max(1, int(sentence_len * 1.35))

    best = None  # (ratio, start_idx, end_idx)
    best_ratio = -1.0
    for i in range(max(0, start_word_idx), end_search):
        if not clean_words[i]:
            continue
        candidate = ''
        upper_j = min(n_words, i + max_sentence_words)
        for j in range(i, upper_j):
            candidate += clean_words[j]
            clen = len(candidate)
            if clen < min_target:
                continue
            if clen > max_target:
                break

            ratio = SequenceMatcher(None, sentence, candidate).ratio()
            if ratio > best_ratio:
                best = (ratio, i, j)
                best_ratio = ratio

    if best and best_ratio >= min_ratio:
        _, s_idx, e_idx = best
        return (
            float(df_words['start'][s_idx]),
            float(df_words['end'][e_idx]),
            s_idx,
            e_idx,
            best[0]
        )
    return None

def get_sentence_timestamps(df_words, df_sentences):
    time_stamp_list = []
    
    # Build complete string and position mapping
    full_words_str = ''
    position_to_word_idx = {}
    clean_words = []
    word_end_char_pos = []
    
    for idx, word in enumerate(df_words['text']):
        clean_word = remove_punctuation(word.lower())
        clean_words.append(clean_word)
        start_pos = len(full_words_str)
        full_words_str += clean_word
        for pos in range(start_pos, len(full_words_str)):
            position_to_word_idx[pos] = idx
        word_end_char_pos.append(len(full_words_str) - 1 if clean_word else start_pos - 1)
    
    current_pos = 0
    prev_end_time = None
    prev_end_word_idx = 0
    fallback_count = 0
    for idx, sentence in df_sentences['Source'].items():
        clean_sentence = remove_punctuation(sentence.lower()).replace(" ", "")
        sentence_len = len(clean_sentence)
        
        match_found = False
        while current_pos <= len(full_words_str) - sentence_len:
            if full_words_str[current_pos:current_pos+sentence_len] == clean_sentence:
                start_word_idx = position_to_word_idx[current_pos]
                end_word_idx = position_to_word_idx[current_pos + sentence_len - 1]
                start_time = float(df_words['start'][start_word_idx])
                end_time = float(df_words['end'][end_word_idx])

                # Guard rail: if this introduces a suspiciously large gap, try a local fuzzy rematch first.
                if prev_end_time is not None and start_time - prev_end_time > SUSPICIOUS_GAP_SECONDS:
                    fuzzy = find_local_fuzzy_match(
                        df_words=df_words,
                        clean_words=clean_words,
                        sentence=clean_sentence,
                        start_word_idx=prev_end_word_idx + 1
                    )
                    if fuzzy is not None:
                        fuzzy_start_time, fuzzy_end_time, fuzzy_s_idx, fuzzy_e_idx, score = fuzzy
                        if fuzzy_start_time - prev_end_time < start_time - prev_end_time:
                            print(
                                f"\n⚠️ Large gap detected ({start_time - prev_end_time:.2f}s). "
                                f"Using local fuzzy rematch (score={score:.3f})."
                            )
                            start_word_idx, end_word_idx = fuzzy_s_idx, fuzzy_e_idx
                            start_time, end_time = fuzzy_start_time, fuzzy_end_time
                
                time_stamp_list.append((
                    start_time,
                    end_time
                ))
                
                current_pos = max(current_pos, word_end_char_pos[end_word_idx] + 1)
                prev_end_time = end_time
                prev_end_word_idx = end_word_idx
                match_found = True
                break
            current_pos += 1
            
        if not match_found:
            print(f"\n⚠️ Warning: No exact match found for sentence: {sentence}")
            show_difference(clean_sentence, 
                          full_words_str[current_pos:current_pos+len(clean_sentence)])
            print("\nOriginal sentence:", df_sentences['Source'][idx])
            # Try a wider fuzzy rematch before giving up.
            wide_fuzzy = find_local_fuzzy_match(
                df_words=df_words,
                clean_words=clean_words,
                sentence=clean_sentence,
                start_word_idx=max(prev_end_word_idx, 0),
                lookahead_words=420,
                max_sentence_words=120,
                min_ratio=0.72,
            )
            if wide_fuzzy is not None:
                fuzzy_start_time, fuzzy_end_time, fuzzy_s_idx, fuzzy_e_idx, score = wide_fuzzy
                print(
                    f"⚠️ No exact match, using wide fuzzy rematch "
                    f"(score={score:.3f}, words={fuzzy_s_idx}-{fuzzy_e_idx})."
                )
                time_stamp_list.append((fuzzy_start_time, fuzzy_end_time))
                prev_end_time = fuzzy_end_time
                prev_end_word_idx = fuzzy_e_idx
                current_pos = max(current_pos, word_end_char_pos[fuzzy_e_idx] + 1)
                continue

            # Final fallback: keep timeline monotonic and estimate a short duration.
            fallback_count += 1
            est_words = max(1, len(str(sentence).split()))
            start_word_idx = min(max(prev_end_word_idx + 1, 0), len(df_words) - 1)
            end_word_idx = min(start_word_idx + est_words - 1, len(df_words) - 1)
            fallback_start = float(df_words['start'][start_word_idx])
            fallback_end = float(df_words['end'][end_word_idx])
            if prev_end_time is not None and fallback_start < prev_end_time:
                fallback_start = prev_end_time
            if fallback_end <= fallback_start:
                fallback_end = fallback_start + 0.35
            print(
                f"⚠️ Using timestamp fallback for unmatched sentence #{fallback_count}: "
                f"{fallback_start:.3f}s -> {fallback_end:.3f}s"
            )
            time_stamp_list.append((fallback_start, fallback_end))
            prev_end_time = fallback_end
            prev_end_word_idx = end_word_idx
            current_pos = max(current_pos, word_end_char_pos[end_word_idx] + 1)
    
    if fallback_count:
        console.print(f"[yellow]⚠️ Timestamp fallback used for {fallback_count} sentence(s).[/yellow]")
    return time_stamp_list

def enforce_min_block_duration(timestamps, min_duration_ms=120):
    """
    Enforce minimum subtitle block duration without changing row count.
    Strategy:
    1) Expand into adjacent gaps first.
    2) Borrow boundary time from adjacent blocks that are longer than minimum.
    3) Last resort: extend end time slightly.
    """
    if not timestamps:
        return timestamps, 0

    min_dur = max(0.0, float(min_duration_ms) / 1000.0)
    if min_dur <= 0:
        return timestamps, 0

    ts = [(float(s), float(e)) for s, e in timestamps]
    changed = 0
    n = len(ts)

    for i in range(n):
        start, end = ts[i]
        dur = end - start
        if dur >= min_dur:
            continue
        need = min_dur - dur
        changed += 1

        # 1) Expand to available gaps around current block.
        left_gap = 0.0
        right_gap = 0.0
        if i > 0:
            left_gap = max(0.0, start - ts[i - 1][1])
        if i < n - 1:
            right_gap = max(0.0, ts[i + 1][0] - end)

        take_left = min(left_gap, need * 0.5)
        start -= take_left
        need -= take_left

        take_right = min(right_gap, need)
        end += take_right
        need -= take_right

        # 2) Borrow from the next block by pushing its start later.
        if need > 0 and i < n - 1:
            next_start, next_end = ts[i + 1]
            next_dur = next_end - next_start
            borrow = min(need, max(0.0, next_dur - min_dur))
            if borrow > 0:
                end += borrow
                ts[i + 1] = (next_start + borrow, next_end)
                need -= borrow

        # 3) Borrow from previous block by pulling its end earlier.
        if need > 0 and i > 0:
            prev_start, prev_end = ts[i - 1]
            prev_dur = prev_end - prev_start
            borrow = min(need, max(0.0, prev_dur - min_dur))
            if borrow > 0:
                start -= borrow
                ts[i - 1] = (prev_start, prev_end - borrow)
                need -= borrow

        # 4) Last resort: extend forward.
        if need > 0:
            end += need

        # Keep monotonic order.
        if i > 0 and start < ts[i - 1][1]:
            start = ts[i - 1][1]
        if i < n - 1 and end > ts[i + 1][0]:
            overflow = end - ts[i + 1][0]
            end -= overflow
        if end <= start:
            end = start + 0.05

        ts[i] = (start, end)

    return ts, changed

def infer_speaker_for_timestamp(df_words, start_time, end_time):
    if 'speaker_id' not in df_words.columns:
        return None
    try:
        overlaps = df_words[(df_words['start'] < end_time) & (df_words['end'] > start_time)]
        if overlaps.empty:
            return None
        speakers = overlaps['speaker_id'].dropna()
        if speakers.empty:
            return None
        return str(speakers.mode().iloc[0])
    except Exception:
        return None

def _merge_text_parts(left_text, right_text, different_speaker, max_lines=2):
    left = str(left_text).strip()
    right = str(right_text).strip()
    if not left:
        return right
    if not right:
        return left

    if different_speaker:
        parts = []
        for txt in (left, right):
            parts.extend([p.strip() for p in str(txt).split('\n') if p.strip()])
        if len(parts) <= max_lines:
            return '\n'.join(parts)
        # Exceeds 2 lines: fallback to space-separated text.
        return ' '.join(parts)

    # Same speaker: keep in one line.
    left_flat = ' '.join([p.strip() for p in left.split('\n') if p.strip()])
    right_flat = ' '.join([p.strip() for p in right.split('\n') if p.strip()])
    return f"{left_flat} {right_flat}".strip()

def merge_short_subtitle_blocks(df_trans_time, min_duration_ms=100, max_lines=2):
    """
    Merge subtitle blocks shorter than threshold into adjacent blocks.
    Different-speaker merge prefers newline (max 2 lines), otherwise space.
    """
    if df_trans_time.empty:
        return df_trans_time, 0

    min_dur = max(0.0, float(min_duration_ms) / 1000.0)
    if min_dur <= 0:
        return df_trans_time, 0

    df = df_trans_time.copy().reset_index(drop=True)
    merged_count = 0

    while True:
        df['duration'] = df['timestamp'].apply(lambda x: float(x[1]) - float(x[0]))
        short_indices = [i for i, d in enumerate(df['duration']) if d < min_dur]
        if not short_indices or len(df) <= 1:
            break

        i = short_indices[0]
        start_i, end_i = map(float, df.at[i, 'timestamp'])

        if i == 0:
            target = 1
        elif i == len(df) - 1:
            target = i - 1
        else:
            prev_end = float(df.at[i - 1, 'timestamp'][1])
            next_start = float(df.at[i + 1, 'timestamp'][0])
            prev_gap = abs(start_i - prev_end)
            next_gap = abs(next_start - end_i)
            target = i - 1 if prev_gap <= next_gap else i + 1

        if target < i:
            # Merge current into previous: (prev + curr)
            p_start, p_end = map(float, df.at[target, 'timestamp'])
            c_start, c_end = start_i, end_i
            merged_start = min(p_start, c_start)
            merged_end = max(p_end, c_end)
            sp_prev = df.at[target, '_speaker_id'] if '_speaker_id' in df.columns else None
            sp_curr = df.at[i, '_speaker_id'] if '_speaker_id' in df.columns else None
            diff_spk = bool(sp_prev and sp_curr and sp_prev != sp_curr)

            for col in ('Source', 'Translation'):
                if col in df.columns:
                    df.at[target, col] = _merge_text_parts(df.at[target, col], df.at[i, col], diff_spk, max_lines=max_lines)
            if '_speaker_id' in df.columns and diff_spk:
                df.at[target, '_speaker_id'] = "__multi__"
            df.at[target, 'timestamp'] = (merged_start, merged_end)
            df = df.drop(index=i).reset_index(drop=True)
        else:
            # Merge current into next: (curr + next)
            n_start, n_end = map(float, df.at[target, 'timestamp'])
            c_start, c_end = start_i, end_i
            merged_start = min(c_start, n_start)
            merged_end = max(c_end, n_end)
            sp_next = df.at[target, '_speaker_id'] if '_speaker_id' in df.columns else None
            sp_curr = df.at[i, '_speaker_id'] if '_speaker_id' in df.columns else None
            diff_spk = bool(sp_next and sp_curr and sp_next != sp_curr)

            for col in ('Source', 'Translation'):
                if col in df.columns:
                    df.at[target, col] = _merge_text_parts(df.at[i, col], df.at[target, col], diff_spk, max_lines=max_lines)
            if '_speaker_id' in df.columns and diff_spk:
                df.at[target, '_speaker_id'] = "__multi__"
            df.at[target, 'timestamp'] = (merged_start, merged_end)
            df = df.drop(index=i).reset_index(drop=True)

        merged_count += 1

    df['duration'] = df['timestamp'].apply(lambda x: float(x[1]) - float(x[0]))
    return df, merged_count

def align_timestamp(df_text, df_translate, subtitle_output_configs: list, output_dir: str, for_display: bool = True):
    """Align timestamps and add a new timestamp column to df_translate"""
    df_trans_time = df_translate.copy()

    # Assign an ID to each word in df_text['text'] and create a new DataFrame
    words = df_text['text'].str.split(expand=True).stack().reset_index(level=1, drop=True).reset_index()
    words.columns = ['id', 'word']
    words['id'] = words['id'].astype(int)

    # Process timestamps ⏰
    time_stamp_list = get_sentence_timestamps(df_text, df_translate)
    df_trans_time['timestamp'] = time_stamp_list
    df_trans_time['duration'] = df_trans_time['timestamp'].apply(lambda x: x[1] - x[0])
    df_trans_time['_speaker_id'] = df_trans_time['timestamp'].apply(
        lambda x: infer_speaker_for_timestamp(df_text, float(x[0]), float(x[1]))
    )

    # Remove gaps 🕳️
    for i in range(len(df_trans_time)-1):
        delta_time = df_trans_time.loc[i+1, 'timestamp'][0] - df_trans_time.loc[i, 'timestamp'][1]
        if 0 < delta_time < 1:
            df_trans_time.at[i, 'timestamp'] = (df_trans_time.loc[i, 'timestamp'][0], df_trans_time.loc[i+1, 'timestamp'][0])

    # First, merge tiny blocks into neighbors (preferred behavior).
    min_block_ms = int(_safe_load_key("subtitle_min_duration_ms", 100))
    max_lines = int(_safe_load_key("subtitle_merge_max_lines", 2))
    df_trans_time, merged_count = merge_short_subtitle_blocks(
        df_trans_time,
        min_duration_ms=min_block_ms,
        max_lines=max_lines
    )
    if merged_count > 0:
        console.print(
            f"[yellow]⚠️ Merged {merged_count} subtitle block(s) shorter than {min_block_ms}ms.[/yellow]"
        )

    # Then, enforce minimum duration as a safety net.
    fixed_timestamps, fixed_count = enforce_min_block_duration(
        list(df_trans_time['timestamp']),
        min_duration_ms=min_block_ms
    )
    if fixed_count > 0:
        console.print(
            f"[yellow]⚠️ Adjusted {fixed_count} subtitle block(s) shorter than {min_block_ms}ms.[/yellow]"
        )
    df_trans_time['timestamp'] = fixed_timestamps
    df_trans_time['duration'] = df_trans_time['timestamp'].apply(lambda x: x[1] - x[0])
    if '_speaker_id' in df_trans_time.columns:
        df_trans_time = df_trans_time.drop(columns=['_speaker_id'])

    # Detect suspicious subtitle timeline holes for easier debugging.
    subtitle_gap_threshold = load_key("subtitle_gap_threshold_seconds")
    sub_gap_rows = []
    for i in range(len(df_trans_time) - 1):
        prev_end = float(df_trans_time.loc[i, 'timestamp'][1])
        next_start = float(df_trans_time.loc[i + 1, 'timestamp'][0])
        gap = next_start - prev_end
        if gap > subtitle_gap_threshold:
            sub_gap_rows.append({
                "prev_cue": i + 1,
                "next_cue": i + 2,
                "prev_end": prev_end,
                "next_start": next_start,
                "gap_seconds": round(gap, 3),
                "prev_source": str(df_trans_time.loc[i, "Source"]) if "Source" in df_trans_time.columns else "",
                "next_source": str(df_trans_time.loc[i + 1, "Source"]) if "Source" in df_trans_time.columns else "",
            })

    if output_dir:
        gap_report_path = os.path.join(_OUTPUT_DIR, "log", "sub_gap_report.csv")
        os.makedirs(os.path.dirname(gap_report_path), exist_ok=True)
        pd.DataFrame(sub_gap_rows).to_csv(gap_report_path, index=False, encoding="utf-8")
        if sub_gap_rows:
            console.print(
                f"[yellow]⚠️ Detected {len(sub_gap_rows)} large subtitle gap(s). "
                f"Report saved to `{gap_report_path}`[/yellow]"
            )

    # Convert start and end timestamps to SRT format
    df_trans_time['timestamp'] = df_trans_time['timestamp'].apply(lambda x: convert_to_srt_format(x[0], x[1]))

    # Polish subtitles: replace punctuation in Translation if for_display
    if for_display:
        df_trans_time['Translation'] = df_trans_time['Translation'].apply(lambda x: re.sub(r'[，。]', ' ', x).strip())

    # Output subtitles 📜
    def generate_subtitle_string(df, columns):
        return ''.join([f"{i+1}\n{row['timestamp']}\n{row[columns[0]].strip()}\n{row[columns[1]].strip() if len(columns) > 1 else ''}\n\n" for i, row in df.iterrows()]).strip()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for filename, columns in subtitle_output_configs:
            subtitle_str = generate_subtitle_string(df_trans_time, columns)
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                f.write(subtitle_str)
    
    return df_trans_time

# ✨ Beautify the translation
def _normalize_ellipsis(text: str) -> str:
    """
    Remove excessive ellipsis.
    - Convert runs like "...", "......", "……" to a single space as sentence separator.
    - Then collapse repeated spaces.
    """
    if not text:
        return ""
    # Dot-style ellipsis and unicode ellipsis runs.
    text = re.sub(r'\.{3,}', ' ', text)
    text = re.sub(r'…+', ' ', text)
    # Trim spaces around punctuation after replacement.
    text = re.sub(r'\s+([,，.。!！?？:：;；])', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def clean_translation(x):
    if pd.isna(x):
        return ''
    cleaned = _normalize_ellipsis(str(x))
    cleaned = cleaned.strip('。').strip('，')
    return autocorrect.format(cleaned)

def align_timestamp_main():
    df_text = pd.read_excel(_2_CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    df_translate = pd.read_excel(_5_SPLIT_SUB)
    df_translate['Translation'] = df_translate['Translation'].apply(clean_translation)
    
    align_timestamp(df_text, df_translate, SUBTITLE_OUTPUT_CONFIGS, _OUTPUT_DIR)
    console.print(Panel("[bold green]🎉📝 Subtitles generation completed! Please check in the `output` folder 👀[/bold green]"))

    # for audio
    df_translate_for_audio = pd.read_excel(_5_REMERGED) # use remerged file to avoid unmatched lines when dubbing
    df_translate_for_audio['Translation'] = df_translate_for_audio['Translation'].apply(clean_translation)
    
    align_timestamp(df_text, df_translate_for_audio, AUDIO_SUBTITLE_OUTPUT_CONFIGS, _AUDIO_DIR)
    console.print(Panel(f"[bold green]🎉📝 Audio subtitles generation completed! Please check in the `{_AUDIO_DIR}` folder 👀[/bold green]"))
    

if __name__ == '__main__':
    align_timestamp_main()
