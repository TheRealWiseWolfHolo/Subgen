import os
from core.st_utils.imports_and_utils import *
from core.utils.onekeycleanup import cleanup
from core.utils import load_key
from core.utils.translation_confirm import wait_for_translation_confirmation_cli
import shutil
from functools import partial
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from core import *

console = Console()

INPUT_DIR = 'batch/input'
OUTPUT_DIR = 'output'
SAVE_DIR = 'batch/output'
ERROR_OUTPUT_DIR = 'batch/output/ERROR'
YTB_RESOLUTION_KEY = "ytb_resolution"

def process_video(file, dubbing=False, is_retry=False):
    if not is_retry:
        prepare_output_folder(OUTPUT_DIR)
    
    text_steps = [
        ("🎥 Processing input file", partial(process_input_file, file)),
        ("🎙️ Transcribing with Whisper", partial(_2_asr.transcribe)),
        ("✂️ Splitting sentences", split_sentences),
        ("📝 Summarizing terminology", summarize_only),
        ("🌐 Translating subtitles", translate_only),
        ("✂️ Splitting translated subtitles", _5_split_sub.split_for_sub_main),
        ("🕒 Aligning timestamps & generating srt", _6_gen_sub.align_timestamp_main),
        ("🎬 Merging subtitles to video", _7_sub_into_vid.merge_subtitles_to_video),
    ]
    
    if dubbing:
        dubbing_steps = [
            ("🔊 Generating audio tasks", gen_audio_tasks),
            ("🎵 Extracting reference audio", _9_refer_audio.extract_refer_audio_main),
            ("🗣️ Generating audio", _10_gen_audio.gen_audio),
            ("🔄 Merging full audio", _11_merge_audio.merge_full_audio),
            ("🎞️ Merging dubbing to video", _12_dub_to_vid.merge_video_audio),
        ]
        text_steps.extend(dubbing_steps)
    
    current_step = ""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
        console=console
    ) as progress:
        task_id = progress.add_task("Starting pipeline...", total=len(text_steps))
        for step_name, step_func in text_steps:
            current_step = step_name
            progress.update(task_id, description=f"{step_name}")
            for attempt in range(3):
                try:
                    if attempt > 0:
                        console.print(f"[cyan]↻ Retrying step: {step_name} ({attempt + 1}/3)[/cyan]")
                    result = step_func()
                    if result is not None:
                        globals().update(result)
                    progress.advance(task_id)
                    break
                except Exception as e:
                    if attempt == 2:
                        error_panel = Panel(
                            f"[bold red]Error in step '{current_step}':[/]\n{str(e)}",
                            border_style="red"
                        )
                        console.print(error_panel)
                        cleanup(ERROR_OUTPUT_DIR)
                        return False, current_step, str(e)
                    console.print(f"[yellow]⚠️ Step failed, retrying ({attempt + 1}/3): {step_name}[/yellow]")
    
    console.print(Panel("[bold green]All steps completed successfully! 🎉[/]", border_style="green"))
    cleanup(SAVE_DIR)
    return True, "", ""

def prepare_output_folder(output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

def process_input_file(file):
    if file.startswith('http'):
        _1_ytdlp.download_video_ytdlp(file, resolution=load_key(YTB_RESOLUTION_KEY))
        video_file = _1_ytdlp.find_video_files()
    else:
        input_file = os.path.join('batch', 'input', file)
        output_file = os.path.join(OUTPUT_DIR, file)
        shutil.copy(input_file, output_file)
        video_file = output_file
    return {'video_file': video_file}

def split_sentences():
    _3_1_split_nlp.split_by_spacy()
    _3_2_split_meaning.split_sentences_by_meaning()

def summarize_only():
    repick_each_video = load_key("batch_terminology_pick_each_video")
    if repick_each_video:
        _4_1_summarize.reset_selected_profile()
        _4_1_summarize.get_summary(interactive_select=True)
    else:
        _4_1_summarize.get_summary()

def translate_only():
    if load_key("pause_before_translate"):
        wait_for_translation_confirmation_cli(
            "⚠️ PAUSE_BEFORE_TRANSLATE is enabled. Review output/log/terminology.json, then press ENTER to continue..."
        )
    _4_2_translate.translate_all()

def gen_audio_tasks():
    _8_1_audio_task.gen_audio_task_main()
    _8_2_dub_chunks.gen_dub_chunks()
