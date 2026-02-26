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
            raise ValueError("❎ No match found for sentence.")
    
    return time_stamp_list

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

    # Remove gaps 🕳️
    for i in range(len(df_trans_time)-1):
        delta_time = df_trans_time.loc[i+1, 'timestamp'][0] - df_trans_time.loc[i, 'timestamp'][1]
        if 0 < delta_time < 1:
            df_trans_time.at[i, 'timestamp'] = (df_trans_time.loc[i, 'timestamp'][0], df_trans_time.loc[i+1, 'timestamp'][0])

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
def clean_translation(x):
    if pd.isna(x):
        return ''
    cleaned = str(x).strip('。').strip('，')
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
