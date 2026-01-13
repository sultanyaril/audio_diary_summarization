# Audio Diary Summarization System

This project transcribes audio diary files and generates summaries using OpenAI's APIs (Whisper for transcription and GPT for summarization).

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key

1. Create a `.env` file in the project root (copy from `.env.example`):
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

2. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

### 3. Prepare Audio Files

Place your audio files in the `data/` folder with the format: `YYYY-MM-DD.m4a`

Example:
```
data/
├── 2025-12-15.m4a
├── 2025-12-16.m4a
└── 2025-12-17.m4a
```

### 4. Run the Application

```bash
python audio_processor.py
```

## Output

The application creates an `output/` folder containing:

- **all_summaries.json** - All transcripts and summaries in JSON format
- **YYYY-MM-DD_summary.md** - Individual markdown files for each diary entry

Each markdown file contains:
- Full transcript
- AI-generated summary (4-5 paragraphs)
- 2 Insights & Suggestions

## Features

- ✅ Automatic transcription using Whisper API
- ✅ Intelligent summarization using GPT-4
- ✅ Validates filename format (YYYY-MM-DD.m4a)
- ✅ Saves results in multiple formats (JSON + Markdown)
- ✅ Error handling and logging
- ✅ Processes files in chronological order

## Supported Audio Format

- .m4a files (MPEG-4 Audio)

## API Costs

- Whisper: $0.02 per minute of audio
- GPT-4: Pricing varies by model version

For more details, see [OpenAI Pricing](https://openai.com/pricing)

## Troubleshooting

**Error: "No .m4a files found"**
- Check that audio files are in the `data/` folder
- Verify files have `.m4a` extension (lowercase)
- Ensure filenames follow `YYYY-MM-DD.m4a` format

**Error: "OPENAI_API_KEY not found"**
- Create a `.env` file with your API key
- Verify the `.env` file is in the project root

**Error: "Invalid date format"**
- Rename files to match `YYYY-MM-DD.m4a` format
- Example: `2025-12-19.m4a`
