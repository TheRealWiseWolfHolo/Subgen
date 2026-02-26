import pandas as pd
import json
import concurrent.futures
import re
import os
from core.translate_lines import translate_lines
from core._4_1_summarize import search_things_to_note_in_prompt
from core._8_1_audio_task import check_len_then_trim
from core._6_gen_sub import align_timestamp
from core.utils import *
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from difflib import SequenceMatcher
from core.utils.models import *
console = Console()

# Strict full-line credit/outro overlays. Avoid substring filtering that can remove real speech.
NOISE_LINE_PATTERNS = [
    r'^\s*thanks?\s+for\s+watching[.!?]*\s*$',
    r'^\s*thank\s+you\s+for\s+watching[.!?]*\s*$',
    r'^\s*caption(s)?\s+by\b.*$',
    r'^\s*transcript\s+by\b.*$',
    r'^\s*subtitle(s)?\s+by\b.*$',
    r'^\s*casting\s*words\s*$',
    r'^\s*感谢[收观]看[！!。.]?\s*$',
    r'^\s*字幕[组者]?[：: ]?.*$',
    r'^\s*听译[：: ]?.*$',
]

def is_noise_credit_line(line: str) -> bool:
    text = str(line).strip()
    if not text:
        return False
    lowered = text.lower()
    return any(re.search(p, lowered, flags=re.IGNORECASE) for p in NOISE_LINE_PATTERNS)

def strip_inline_noise_phrases(line: str) -> str:
    """Remove embedded credit/outro phrases from otherwise valid sentences."""
    text = str(line)
    inline_patterns = [
        r'\bthanks?\s+for\s+watching\b',
        r'\bwe\'ll\s+be\s+right\s+back\b',
        r'\btranscription\s+by\s+casting\s*words\b',
        r'感谢[收观]看',
        r'由\s*casting\s*words\s*听译',
        r'听译[:：]?',
        r'字幕[组者]?[:：]?',
    ]
    for p in inline_patterns:
        text = re.sub(p, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip(' ,，。!！?？-—')
    return text

# Function to split text into chunks
def split_chunks_by_chars(chunk_size, max_i): 
    """Split text into chunks based on character count, return a list of multi-line text chunks"""
    with open(_3_2_SPLIT_BY_MEANING, "r", encoding="utf-8") as file:
        sentences = file.read().strip().split('\n')

    raw_count = len(sentences)
    filtered_log_rows = []
    cleaned_sentences = []
    for i, sentence in enumerate(sentences, start=1):
        stripped = strip_inline_noise_phrases(sentence)
        if not stripped:
            filtered_log_rows.append((i, "inline_cleanup_empty", sentence))
            continue
        if is_noise_credit_line(stripped):
            filtered_log_rows.append((i, "strict_credit_match", sentence))
            continue
        cleaned_sentences.append(stripped)
    sentences = cleaned_sentences

    removed_count = raw_count - len(sentences)
    if removed_count > 0:
        console.print(f"[yellow]⚠️ Filtered {removed_count} likely credit/outro line(s) before translation.[/yellow]")
        os.makedirs("output/log", exist_ok=True)
        report_path = "output/log/filtered_lines.txt"
        with open(report_path, "w", encoding="utf-8") as report_file:
            for line_idx, reason, original in filtered_log_rows:
                report_file.write(f"{line_idx}\t{reason}\t{original}\n")
        console.print(f"[yellow]🧾 Filter report saved to {report_path}[/yellow]")

    chunks = []
    chunk = ''
    sentence_count = 0
    for sentence in sentences:
        if len(chunk) + len(sentence + '\n') > chunk_size or sentence_count == max_i:
            chunks.append(chunk.strip())
            chunk = sentence + '\n'
            sentence_count = 1
        else:
            chunk += sentence + '\n'
            sentence_count += 1
    chunks.append(chunk.strip())
    return chunks

# Get context from surrounding chunks
def get_previous_content(chunks, chunk_index):
    return None if chunk_index == 0 else chunks[chunk_index - 1].split('\n')[-3:] # Get last 3 lines
def get_after_content(chunks, chunk_index):
    return None if chunk_index == len(chunks) - 1 else chunks[chunk_index + 1].split('\n')[:2] # Get first 2 lines

# 🔍 Translate a single chunk
def translate_chunk(chunk, chunks, theme_prompt, i):
    things_to_note_prompt = search_things_to_note_in_prompt(chunk)
    previous_content_prompt = get_previous_content(chunks, i)
    after_content_prompt = get_after_content(chunks, i)
    translation, english_result = translate_lines(chunk, previous_content_prompt, after_content_prompt, things_to_note_prompt, theme_prompt, i)
    return i, english_result, translation

# Add similarity calculation function
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# 🚀 Main function to translate all chunks
@check_file_exists(_4_2_TRANSLATION)
def translate_all():
    console.print("[bold green]Start Translating All...[/bold green]")
    chunks = split_chunks_by_chars(chunk_size=600, max_i=10)
    with open(_4_1_TERMINOLOGY, 'r', encoding='utf-8') as file:
        theme_prompt = json.load(file).get('theme')

    # 🔄 Use concurrent execution for translation
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("[cyan]Translating chunks...", total=len(chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                future = executor.submit(translate_chunk, chunk, chunks, theme_prompt, i)
                futures.append(future)
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                progress.update(task, advance=1)

    results.sort(key=lambda x: x[0])  # Sort results based on original order
    
    # 💾 Save results in original chunk order (avoid cross-chunk best-match mispairing)
    src_text, trans_text = [], []
    for i, chunk in enumerate(chunks):
        chunk_lines = chunk.split('\n')
        src_text.extend(chunk_lines)

        result_i = results[i]
        chunk_text = ''.join(chunk_lines).lower()
        expected_text = ''.join(result_i[1].split('\n')).lower()
        ratio = similar(expected_text, chunk_text)
        if ratio < 0.9:
            console.print(f"[yellow]Warning: Chunk {i} source check is low (similarity: {ratio:.3f})[/yellow]")
            raise ValueError(f"Translation source verification failed (chunk {i})")
        elif ratio < 1.0:
            console.print(f"[yellow]Warning: Chunk {i} source check is near match (similarity: {ratio:.3f})[/yellow]")

        trans_text.extend(result_i[2].split('\n'))
    
    # Trim long translation text
    df_text = pd.read_excel(_2_CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    df_translate = pd.DataFrame({'Source': src_text, 'Translation': trans_text})
    subtitle_output_configs = [('trans_subs_for_audio.srt', ['Translation'])]
    df_time = align_timestamp(df_text, df_translate, subtitle_output_configs, output_dir=None, for_display=False)
    console.print(df_time)
    # apply check_len_then_trim to df_time['Translation'], only when duration > MIN_TRIM_DURATION.
    df_time['Translation'] = df_time.apply(lambda x: check_len_then_trim(x['Translation'], x['duration']) if x['duration'] > load_key("min_trim_duration") else x['Translation'], axis=1)
    console.print(df_time)
    
    df_time.to_excel(_4_2_TRANSLATION, index=False)
    console.print("[bold green]✅ Translation completed and results saved.[/bold green]")

if __name__ == '__main__':
    translate_all()
