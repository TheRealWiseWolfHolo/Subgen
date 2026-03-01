import os, subprocess
import pandas as pd
from typing import Dict, List, Tuple
from pydub import AudioSegment
from core.utils import *
from core.utils.models import *
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.utils import mediainfo
from rich import print as rprint

def _pick_silence_split_point(
    audio: AudioSegment,
    threshold: float,
    duration: float,
    win: float,
    safe_margin: float
):
    """Find a split point near threshold by adaptively widening the search area and silence threshold."""
    window_candidates = [win, win * 2, win * 3]
    # Relative threshold helps when background noise is high and absolute -30 is too strict.
    rel_thresh = audio.dBFS - 14
    silence_thresh_candidates = [rel_thresh - 3, rel_thresh, -30, -27, -24]
    silence_thresh_candidates = [
        max(-50, min(-12, t)) for t in silence_thresh_candidates if pd.notna(t)
    ]
    min_silence_candidates_ms = [int(safe_margin * 1000), 350, 250]

    best_split = None
    best_distance = float("inf")

    for search_win in window_candidates:
        ws = int(max(0.0, threshold - search_win) * 1000)
        we = int(min(duration, threshold + search_win) * 1000)
        if we <= ws:
            continue

        sub_audio = audio[ws:we]
        region_offset = threshold - search_win
        for min_silence_ms in min_silence_candidates_ms:
            for silence_thresh in silence_thresh_candidates:
                silence_regions = detect_silence(
                    sub_audio,
                    min_silence_len=max(150, min_silence_ms),
                    silence_thresh=silence_thresh
                )
                if not silence_regions:
                    continue

                for s, e in silence_regions:
                    start = s / 1000 + region_offset
                    end = e / 1000 + region_offset
                    if (end - start) < (safe_margin * 2):
                        continue
                    split_at = start + safe_margin
                    if split_at <= 0 or split_at >= duration:
                        continue
                    distance = abs(split_at - threshold)
                    if distance < best_distance:
                        best_distance = distance
                        best_split = split_at

                if best_split is not None:
                    return best_split, search_win, silence_thresh, min_silence_ms

    return None, None, None, None

def normalize_audio_volume(audio_path, output_path, target_db = -20.0, format = "wav"):
    audio = AudioSegment.from_file(audio_path)
    change_in_dBFS = target_db - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_dBFS)
    normalized_audio.export(output_path, format=format)
    rprint(f"[green]✅ Audio normalized from {audio.dBFS:.1f}dB to {target_db:.1f}dB[/green]")
    return output_path

def convert_video_to_audio(video_file: str):
    os.makedirs(_AUDIO_DIR, exist_ok=True)
    if not os.path.exists(_RAW_AUDIO_FILE):
        rprint(f"[blue]🎬➡️🎵 Converting to high quality audio with FFmpeg ......[/blue]")
        subprocess.run([
            'ffmpeg', '-y', '-i', video_file, '-vn',
            '-c:a', 'libmp3lame', '-b:a', '96k',
            '-ar', '16000',
            '-ac', '1', 
            '-metadata', 'encoding=UTF-8', _RAW_AUDIO_FILE
        ], check=True, stderr=subprocess.PIPE)
        rprint(f"[green]🎬➡️🎵 Converted <{video_file}> to <{_RAW_AUDIO_FILE}> with FFmpeg\n[/green]")

def get_audio_duration(audio_file: str) -> float:
    """Get the duration of an audio file using ffmpeg."""
    cmd = ['ffmpeg', '-i', audio_file]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    output = stderr.decode('utf-8', errors='ignore')
    
    try:
        duration_str = [line for line in output.split('\n') if 'Duration' in line][0]
        duration_parts = duration_str.split('Duration: ')[1].split(',')[0].split(':')
        duration = float(duration_parts[0])*3600 + float(duration_parts[1])*60 + float(duration_parts[2])
    except Exception as e:
        print(f"[red]❌ Error: Failed to get audio duration: {e}[/red]")
        duration = 0
    return duration

def split_audio(audio_file: str, target_len: float = 30*60, win: float = 120) -> List[Tuple[float, float]]:
    ## 在 [target_len-win, target_len+win] 区间内用 pydub 检测静默，切分音频
    rprint(f"[blue]🎙️ Starting audio segmentation {audio_file} {target_len} {win}[/blue]")
    audio = AudioSegment.from_file(audio_file)
    duration = float(mediainfo(audio_file)["duration"])
    if duration <= target_len + win:
        return [(0, duration)]
    segments, pos = [], 0.0
    safe_margin = 0.5  # 静默点前后安全边界，单位秒

    while pos < duration:
        if duration - pos <= target_len:
            segments.append((pos, duration)); break

        threshold = pos + target_len
        split_at, used_win, used_thresh, used_min_silence = _pick_silence_split_point(
            audio, threshold, duration, win, safe_margin
        )
        if split_at is None:
            rprint(f"[yellow]⚠️ No valid silence regions found for {audio_file} at {threshold}s, using threshold[/yellow]")
            split_at = threshold
        else:
            rprint(
                f"[green]✅ Split near {threshold:.1f}s using window ±{used_win:.0f}s, "
                f"silence_thresh={used_thresh:.1f}dBFS, min_silence={used_min_silence}ms -> {split_at:.3f}s[/green]"
            )
            
        segments.append((pos, split_at)); pos = split_at

    rprint(f"[green]🎙️ Audio split completed {len(segments)} segments[/green]")
    return segments

def process_transcription(result: Dict) -> pd.DataFrame:
    all_words = []
    for segment in result['segments']:
        # Get speaker_id, if not exists, set to None
        speaker_id = segment.get('speaker_id', None)
        
        for word in segment['words']:
            # Check word length
            if len(word["word"]) > 30:
                rprint(f"[yellow]⚠️ Warning: Detected word longer than 30 characters, skipping: {word['word']}[/yellow]")
                continue
                
            # ! For French, we need to convert guillemets to empty strings
            word["word"] = word["word"].replace('»', '').replace('«', '')
            
            if 'start' not in word and 'end' not in word:
                if all_words:
                    # Assign the end time of the previous word as the start and end time of the current word
                    word_dict = {
                        'text': word["word"],
                        'start': all_words[-1]['end'],
                        'end': all_words[-1]['end'],
                        'speaker_id': speaker_id
                    }
                    all_words.append(word_dict)
                else:
                    # If it's the first word, look next for a timestamp then assign it to the current word
                    next_word = next((w for w in segment['words'] if 'start' in w and 'end' in w), None)
                    if next_word:
                        word_dict = {
                            'text': word["word"],
                            'start': next_word["start"],
                            'end': next_word["end"],
                            'speaker_id': speaker_id
                        }
                        all_words.append(word_dict)
                    else:
                        raise Exception(f"No next word with timestamp found for the current word : {word}")
            else:
                # Normal case, with start and end times
                word_dict = {
                    'text': f'{word["word"]}',
                    'start': word.get('start', all_words[-1]['end'] if all_words else 0),
                    'end': word['end'],
                    'speaker_id': speaker_id
                }
                
                all_words.append(word_dict)
    
    return pd.DataFrame(all_words)

def save_results(df: pd.DataFrame):
    os.makedirs('output/log', exist_ok=True)

    # Remove rows where 'text' is empty
    initial_rows = len(df)
    df = df[df['text'].str.len() > 0]
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        rprint(f"[blue]ℹ️ Removed {removed_rows} row(s) with empty text.[/blue]")
    
    # Check for and remove words longer than 20 characters
    long_words = df[df['text'].str.len() > 30]
    if not long_words.empty:
        rprint(f"[yellow]⚠️ Warning: Detected {len(long_words)} word(s) longer than 30 characters. These will be removed.[/yellow]")
        df = df[df['text'].str.len() <= 30]

    # Detect suspicious no-word spans in ASR output.
    df = df.sort_values("start").reset_index(drop=True)
    gap_threshold = load_key("asr_gap_threshold_seconds")
    gap_rows = []
    for i in range(len(df) - 1):
        prev_end = float(df.loc[i, "end"])
        next_start = float(df.loc[i + 1, "start"])
        gap = next_start - prev_end
        if gap > gap_threshold:
            gap_rows.append({
                "prev_word_idx": i,
                "next_word_idx": i + 1,
                "prev_word": str(df.loc[i, "text"]),
                "next_word": str(df.loc[i + 1, "text"]),
                "prev_end": prev_end,
                "next_start": next_start,
                "gap_seconds": round(gap, 3),
            })

    gap_report_path = "output/log/asr_gap_report.csv"
    pd.DataFrame(gap_rows).to_csv(gap_report_path, index=False, encoding="utf-8")
    if gap_rows:
        rprint(f"[yellow]⚠️ Detected {len(gap_rows)} large ASR gap(s). Report saved to {gap_report_path}[/yellow]")
    
    df['text'] = df['text'].apply(lambda x: f'"{x}"')
    df.to_excel(_2_CLEANED_CHUNKS, index=False)
    rprint(f"[green]📊 Excel file saved to {_2_CLEANED_CHUNKS}[/green]")

def save_language(language: str):
    update_key("whisper.detected_language", language)
