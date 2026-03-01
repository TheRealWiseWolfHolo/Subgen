# Subgen

Fork of [Huanshere/VideoLingo](https://github.com/Huanshere/VideoLingo) focused on long-term maintainability, safer defaults, and subtitle reliability fixes.

## What Changed In This Fork

### Subtitle gap diagnostics and recovery
- Added ASR no-word gap report: `output/log/asr_gap_report.csv`
- Added subtitle timeline gap report: `output/log/sub_gap_report.csv`
- Added optional ASR gap recovery pass that retries transcription on detected gap windows
- Improved source audio extraction quality for ASR (`96k` instead of `32k`)

### Subtitle preprocessing robustness
- Tightened noise-line filtering so normal spoken lines are less likely to be dropped
- Added filtered-line report: `output/log/filtered_lines.txt`

### Terminology workflow improvements
- Added cross-video terminology profile library under `history/terminology_profiles/`
- Added CLI profile picker:
  - choose existing terminology profile
  - or create a new one by name
- Added batch option to repick terminology profile for every video
- Added Streamlit profile selector toggle + picker in sidebar (`streamlit_terminology_profile_select`)
- Added automatic person-name extraction into `output/log/terminology.json` (names kept as original)

### Local ASR backend updates (Apple Silicon ready)
- Added platform-aware local backend selection:
  - Apple Silicon (`darwin` + `arm64`) defaults to `openai_whisper` + `mps`
  - Other platforms default to `whisperx`
- Added configurable local backend and device:
  - `whisper.local_backend`: `auto | whisperx | openai_whisper`
  - `whisper.local_device`: `auto | cpu | cuda | mps`
- Removed old pinned WhisperX git commit from dependencies

### Translation behavior updates
- Updated prompts so person names default to original English (except globally well-known figures with established exonyms)
- Added configurable terminology extraction limit: `terminology_max_terms`

### Security / repo hygiene
- Added `config.example.yaml` (sanitized template)
- `config.yaml` is ignored to avoid leaking API keys

## Quick Start

1. Clone:
```bash
git clone https://github.com/TheRealWiseWolfHolo/Subgen.git
cd Subgen
```

2. Create environment:
```bash
conda create -n subgen python=3.10.0 -y
conda activate subgen
```

3. Install dependencies (manual / reproducible path):
```bash
python -m pip install --upgrade pip
python -m pip install --upgrade torch torchaudio
python -m pip install -r requirements.txt
python -m pip install --no-deps "demucs[dev] @ git+https://github.com/adefossez/demucs"
python -m pip install -e .
```

4. Alternative installer:
```bash
python install.py
```
`install.py` does the same core steps and includes platform-aware PyTorch install logic.

5. Create local config and fill your keys:
```bash
cp config.example.yaml config.yaml
```

6. Run:
```bash
streamlit run st.py
```

## Dependency Notes

- `requirements.txt` intentionally does not pin `demucs` directly.
- `requirements.txt` includes Demucs runtime dependencies (except `torch`/`torchaudio`, which are managed separately for latest-version installs).
- `demucs` currently depends on old `torchaudio` constraints; to keep newer `torch/torchaudio`, install demucs separately with `--no-deps`.
- If you see `ModuleNotFoundError: No module named 'demucs'`, run:
```bash
python -m pip install --no-deps "demucs[dev] @ git+https://github.com/adefossez/demucs"
```

## Config Notes

Key options introduced in this fork:
- `asr_gap_threshold_seconds`
- `subtitle_gap_threshold_seconds`
- `asr_gap_recover`
- `asr_gap_recover_padding_seconds`
- `asr_gap_recover_max_windows`
- `batch_terminology_pick_each_video`
- `streamlit_terminology_profile_select`
- `terminology_max_terms`
- `whisper.local_backend`
- `whisper.local_device`

Use `config.example.yaml` as the tracked template.

## Terminology Profiles (CLI + Streamlit)

When running in CLI/batch mode, summary stage can prompt:

```text
Pick terminology json:
1. ProfileA
2. ProfileB
3. New (Please name)
```

Selected profile is reused in-process unless repick-per-video is enabled.

In Streamlit mode, use the sidebar selector:
- toggle: `Terminology profile selection`
- choose an existing profile or create `+ New profile`

## License

Apache License 2.0. See [LICENSE](./LICENSE).
