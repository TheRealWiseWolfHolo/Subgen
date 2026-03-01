import streamlit as st
import os, sys
import threading
from core.st_utils.imports_and_utils import *
from core import *
from core.utils.memory_utils import release_runtime_memory
from core.utils.translation_confirm import (
    approve_translation_confirmation,
    clear_translation_confirmation,
    consume_translation_confirmation,
    is_translation_confirmation_pending,
    is_translation_confirmed,
    set_translation_confirmation_pending,
)

# SET PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PATH'] += os.pathsep + current_dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Subgen", page_icon="docs/logo.svg")

SUB_VIDEO = "output/output_sub.mp4"
DUB_VIDEO = "output/output_dub.mp4"

def _safe_load_key(key, default):
    try:
        return load_key(key)
    except Exception:
        return default

def render_terminology_profile_selector():
    enabled_cfg = _safe_load_key("streamlit_terminology_profile_select", True)
    enabled = st.toggle(
        "Terminology Profile Selection / 术语档案选择",
        value=enabled_cfg,
        help="Enable selecting/creating a terminology profile before summary extraction. / 在摘要阶段前启用术语档案选择或新建。"
    )
    if enabled != enabled_cfg:
        update_key("streamlit_terminology_profile_select", enabled)
        st.rerun()

    if not enabled:
        st.session_state.pop("terminology_profile_name", None)
        st.session_state.pop("terminology_profile_active", None)
        return

    # Backward compatibility for previous session key.
    if "terminology_profile_active" not in st.session_state and "terminology_profile_name" in st.session_state:
        st.session_state["terminology_profile_active"] = st.session_state["terminology_profile_name"]

    profile_names = _4_1_summarize.list_terminology_profile_names()
    options = profile_names + ["+ New profile"]
    default_idx = 0
    active_profile = st.session_state.get("terminology_profile_active", "")
    if active_profile:
        current = active_profile
        if current in profile_names:
            default_idx = profile_names.index(current)

    selected = st.selectbox("Terminology Profile / 术语档案", options=options, index=default_idx)
    candidate_profile = ""
    if selected == "+ New profile":
        new_name = st.text_input("New Profile Name / 新档案名称")
        if new_name.strip():
            candidate_profile = new_name.strip()
    else:
        candidate_profile = selected

    confirm_clicked = st.button("Confirm Profile / 确认档案", key="confirm_terminology_profile")
    if confirm_clicked:
        if not candidate_profile:
            st.error("Please select an existing profile or input a new profile name first. / 请先选择已有档案或输入新档案名称。")
        else:
            st.session_state["terminology_profile_active"] = candidate_profile
            st.session_state["terminology_profile_name"] = candidate_profile
            st.success(f"Profile confirmed / 已确认档案: {candidate_profile}")

    active_profile = st.session_state.get("terminology_profile_active", "").strip()
    if active_profile:
        st.info(f"Current profile / 当前档案: {active_profile}")
    else:
        st.warning("No profile confirmed yet. / 尚未确认档案。")


def _start_terminal_confirmation_listener_once():
    if not sys.stdin.isatty():
        return
    if st.session_state.get("terminal_confirm_listener_started"):
        return

    def _worker():
        try:
            input(
                "⚠️ PAUSE_BEFORE_TRANSLATE: review output/log/terminology.json. "
                "Press ENTER here to continue (or confirm in Streamlit)."
            )
            if is_translation_confirmation_pending():
                approve_translation_confirmation("terminal")
        except EOFError:
            return

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    st.session_state["terminal_confirm_listener_started"] = True


def continue_text_after_confirmation():
    try:
        _4_2_translate.translate_all()
        with st.spinner(t("Processing and aligning subtitles...")):
            _5_split_sub.split_for_sub_main()
            _6_gen_sub.align_timestamp_main()
        with st.spinner(t("Merging subtitles to video...")):
            _7_sub_into_vid.merge_subtitles_to_video()
        st.session_state["awaiting_translate_confirmation"] = False
        st.session_state["terminal_confirm_listener_started"] = False
        clear_translation_confirmation()
        st.success(t("Subtitle processing complete! 🎉"))
        st.balloons()
    finally:
        release_runtime_memory("subtitle pipeline")

def text_processing_section():
    st.header(t("b. Translate and Generate Subtitles"))
    with st.container(border=True):
        st.markdown(f"""
        <p style='font-size: 20px;'>
        {t("This stage includes the following steps:")}
        <p style='font-size: 20px;'>
            1. {t("WhisperX word-level transcription")}<br>
            2. {t("Sentence segmentation using NLP and LLM")}<br>
            3. {t("Summarization and multi-step translation")}<br>
            4. {t("Cutting and aligning long subtitles")}<br>
            5. {t("Generating timeline and subtitles")}<br>
            6. {t("Merging subtitles into the video")}
        """, unsafe_allow_html=True)
        if st.session_state.get("awaiting_translate_confirmation", False) or is_translation_confirmation_pending():
            st.warning("Waiting for translation confirmation / 等待翻译确认")
            st.caption("Confirm in Streamlit below, or press ENTER in terminal. / 你可以在下方确认，或在终端按回车确认。")
            if st.button("Continue Translation / 继续翻译", key="continue_translation_button"):
                approve_translation_confirmation("streamlit")
            if is_translation_confirmed():
                consume_translation_confirmation()
                with st.spinner(t("Translating...")):
                    continue_text_after_confirmation()
                st.rerun()
            return

        if not os.path.exists(SUB_VIDEO):
            if st.button(t("Start Processing Subtitles"), key="text_processing_button"):
                process_text()
                st.rerun()
        else:
            if load_key("burn_subtitles"):
                st.video(SUB_VIDEO)
            download_subtitle_zip_button(text=t("Download All Srt Files"))
            
            if st.button(t("Archive to 'history'"), key="cleanup_in_text_processing"):
                cleanup()
                st.rerun()
            return True

def process_text():
    try:
        with st.spinner(t("Using Whisper for transcription...")):
            _2_asr.transcribe()
        with st.spinner(t("Splitting long sentences...")):  
            _3_1_split_nlp.split_by_spacy()
            _3_2_split_meaning.split_sentences_by_meaning()
        with st.spinner(t("Summarizing and translating...")):
            if _safe_load_key("streamlit_terminology_profile_select", True):
                selected_profile = st.session_state.get("terminology_profile_active", "").strip()
                if not selected_profile:
                    st.error("Please confirm a terminology profile in the sidebar first. / 请先在侧边栏确认术语档案。")
                    st.stop()
                _4_1_summarize.set_selected_profile(selected_profile)
                _4_1_summarize.get_summary(interactive_select=False)
            else:
                _4_1_summarize.reset_selected_profile()
                _4_1_summarize.get_summary(interactive_select=False)
            if load_key("pause_before_translate"):
                set_translation_confirmation_pending()
                st.session_state["awaiting_translate_confirmation"] = True
                _start_terminal_confirmation_listener_once()
                st.info(
                    "PAUSE_BEFORE_TRANSLATE is enabled. Please review output/log/terminology.json and confirm in Streamlit or terminal."
                )
                return
            with st.spinner(t("Translating...")):
                continue_text_after_confirmation()
    finally:
        # Covers ASR/split/summary memory even when waiting for manual confirmation.
        release_runtime_memory("text stage")

def audio_processing_section():
    st.header(t("c. Dubbing"))
    with st.container(border=True):
        st.markdown(f"""
        <p style='font-size: 20px;'>
        {t("This stage includes the following steps:")}
        <p style='font-size: 20px;'>
            1. {t("Generate audio tasks and chunks")}<br>
            2. {t("Extract reference audio")}<br>
            3. {t("Generate and merge audio files")}<br>
            4. {t("Merge final audio into video")}
        """, unsafe_allow_html=True)
        if not os.path.exists(DUB_VIDEO):
            if st.button(t("Start Audio Processing"), key="audio_processing_button"):
                process_audio()
                st.rerun()
        else:
            st.success(t("Audio processing is complete! You can check the audio files in the `output` folder."))
            if load_key("burn_subtitles"):
                st.video(DUB_VIDEO) 
            if st.button(t("Delete dubbing files"), key="delete_dubbing_files"):
                delete_dubbing_files()
                st.rerun()
            if st.button(t("Archive to 'history'"), key="cleanup_in_audio_processing"):
                cleanup()
                st.rerun()

def process_audio():
    try:
        with st.spinner(t("Generate audio tasks")): 
            _8_1_audio_task.gen_audio_task_main()
            _8_2_dub_chunks.gen_dub_chunks()
        with st.spinner(t("Extract refer audio")):
            _9_refer_audio.extract_refer_audio_main()
        with st.spinner(t("Generate all audio")):
            _10_gen_audio.gen_audio()
        with st.spinner(t("Merge full audio")):
            _11_merge_audio.merge_full_audio()
        with st.spinner(t("Merge dubbing to the video")):
            _12_dub_to_vid.merge_video_audio()
        
        st.success(t("Audio processing complete! 🎇"))
        st.balloons()
    finally:
        release_runtime_memory("audio stage")

def main():
    st.markdown(button_style, unsafe_allow_html=True)
    welcome_text = "Hello, welcome to Subgen. If you encounter any issues, feel free to get instant answers with our Free QA Agent <a href=\"https://share.fastgpt.in/chat/share?shareId=066w11n3r9aq6879r4z0v9rh\" target=\"_blank\">here</a>!"
    st.markdown(f"<p style='font-size: 20px; color: #808080;'>{welcome_text}</p>", unsafe_allow_html=True)
    # add settings
    with st.sidebar:
        page_setting()
        st.markdown("---")
        render_terminology_profile_selector()
        st.markdown(give_star_button, unsafe_allow_html=True)
    download_video_section()
    text_processing_section()
    audio_processing_section()

if __name__ == "__main__":
    main()
