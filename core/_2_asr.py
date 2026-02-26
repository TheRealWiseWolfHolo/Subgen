from core.utils import *
from core.asr_backend.demucs_vl import demucs_audio
from core.asr_backend.audio_preprocess import process_transcription, convert_video_to_audio, split_audio, save_results, normalize_audio_volume
from core._1_ytdlp import find_video_files
from core.utils.models import *
from typing import Dict, List, Tuple


def _extract_words_with_time(result: Dict) -> List[Tuple[float, float]]:
    words = []
    for seg in result.get("segments", []):
        for word in seg.get("words", []):
            if "start" in word and "end" in word:
                words.append((float(word["start"]), float(word["end"])))
    words.sort(key=lambda x: x[0])
    return words


def _detect_asr_gaps(result: Dict, gap_threshold: float, pad_seconds: float) -> List[Tuple[float, float]]:
    words = _extract_words_with_time(result)
    if len(words) < 2:
        return []

    windows = []
    for i in range(len(words) - 1):
        prev_end = words[i][1]
        next_start = words[i + 1][0]
        gap = next_start - prev_end
        if gap > gap_threshold:
            start = max(0.0, prev_end - pad_seconds)
            end = next_start + pad_seconds
            windows.append((start, end))
    return windows


def _recover_asr_gaps(ts, runtime: str, raw_audio_file: str, vocal_audio_file: str, combined_result: Dict):
    if not load_key("asr_gap_recover"):
        return

    gap_threshold = float(load_key("asr_gap_threshold_seconds"))
    pad_seconds = float(load_key("asr_gap_recover_padding_seconds"))
    max_windows = int(load_key("asr_gap_recover_max_windows"))
    windows = _detect_asr_gaps(combined_result, gap_threshold, pad_seconds)
    if not windows:
        rprint("[green]✅ No large ASR gaps detected for recovery.[/green]")
        return

    windows = windows[:max_windows]
    rprint(f"[yellow]⚠️ Trying ASR gap recovery for {len(windows)} window(s)...[/yellow]")

    recovered_segments = []
    for i, (start, end) in enumerate(windows, start=1):
        rprint(f"[cyan]🔁 Recovery window {i}/{len(windows)}: {start:.3f}s -> {end:.3f}s[/cyan]")
        try:
            # First retry: same pipeline settings on a narrowed window.
            retry_result = ts(raw_audio_file, vocal_audio_file, start, end)
            for seg in retry_result.get("segments", []):
                if seg.get("end", 0) > start and seg.get("start", 0) < end:
                    recovered_segments.append(seg)
        except Exception as e:
            rprint(f"[yellow]⚠️ Gap recovery attempt failed: {e}[/yellow]")

        # Local + demucs fallback: retry with vocal audio for both transcribe and align.
        if runtime == "local" and load_key("demucs"):
            try:
                retry_result_vocal = ts(vocal_audio_file, vocal_audio_file, start, end)
                for seg in retry_result_vocal.get("segments", []):
                    if seg.get("end", 0) > start and seg.get("start", 0) < end:
                        recovered_segments.append(seg)
            except Exception as e:
                rprint(f"[yellow]⚠️ Vocal-only recovery attempt failed: {e}[/yellow]")

    if not recovered_segments:
        rprint("[yellow]⚠️ Gap recovery produced no extra segments.[/yellow]")
        return

    combined_result["segments"].extend(recovered_segments)
    combined_result["segments"].sort(key=lambda x: float(x.get("start", 0.0)))
    rprint(f"[green]✅ Gap recovery added {len(recovered_segments)} segment(s).[/green]")

@check_file_exists(_2_CLEANED_CHUNKS)
def transcribe():
    # 1. video to audio
    video_file = find_video_files()
    convert_video_to_audio(video_file)

    # 2. Demucs vocal separation:
    if load_key("demucs"):
        demucs_audio()
        vocal_audio = normalize_audio_volume(_VOCAL_AUDIO_FILE, _VOCAL_AUDIO_FILE, format="mp3")
    else:
        vocal_audio = _RAW_AUDIO_FILE

    # 3. Extract audio
    segments = split_audio(_RAW_AUDIO_FILE)
    
    # 4. Transcribe audio by clips
    all_results = []
    runtime = load_key("whisper.runtime")
    if runtime == "local":
        from core.asr_backend.whisperX_local import transcribe_audio as ts
        rprint("[cyan]🎤 Transcribing audio with local model...[/cyan]")
    elif runtime == "cloud":
        from core.asr_backend.whisperX_302 import transcribe_audio_302 as ts
        rprint("[cyan]🎤 Transcribing audio with 302 API...[/cyan]")
    elif runtime == "elevenlabs":
        from core.asr_backend.elevenlabs_asr import transcribe_audio_elevenlabs as ts
        rprint("[cyan]🎤 Transcribing audio with ElevenLabs API...[/cyan]")

    for start, end in segments:
        result = ts(_RAW_AUDIO_FILE, vocal_audio, start, end)
        all_results.append(result)
    
    # 5. Combine results
    combined_result = {'segments': []}
    for result in all_results:
        combined_result['segments'].extend(result['segments'])

    # 5.1 Recover large no-word spans by re-running ASR on detected windows.
    _recover_asr_gaps(ts, runtime, _RAW_AUDIO_FILE, vocal_audio, combined_result)
    
    # 6. Process df
    df = process_transcription(combined_result)
    save_results(df)
        
if __name__ == "__main__":
    transcribe()
