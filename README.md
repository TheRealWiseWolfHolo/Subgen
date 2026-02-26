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

2. Create environment and install:
```bash
conda create -n subgen python=3.10.0 -y
conda activate subgen
python install.py
```

3. Create local config and fill your keys:
```bash
cp config.example.yaml config.yaml
```

4. Run:
```bash
streamlit run st.py
```

## Config Notes

Key options introduced in this fork:
- `asr_gap_threshold_seconds`
- `subtitle_gap_threshold_seconds`
- `asr_gap_recover`
- `asr_gap_recover_padding_seconds`
- `asr_gap_recover_max_windows`
- `batch_terminology_pick_each_video`
- `terminology_max_terms`

Use `config.example.yaml` as the tracked template.

## Terminology Profiles (CLI)

When running in CLI/batch mode, summary stage can prompt:

```text
Pick terminology json:
1. ProfileA
2. ProfileB
3. New (Please name)
```

Selected profile is reused in-process unless repick-per-video is enabled.

## License

Apache License 2.0. See [LICENSE](./LICENSE).
