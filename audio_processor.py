"""
Audio Diary Transcription and Summarization System

This module transcribes and summarizes audio files from the data folder.
Audio files should be formatted as YYYY-MM-DD.m4a
Uses LangChain for orchestration and OpenAI APIs for transcription and summarization.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()


class AudioDiarySummarizer:
    """Handles transcription and summarization of audio diary files."""

    def __init__(self, data_folder: str = "data", output_folder: str = "output"):
        """
        Initialize the summarizer.

        Args:
            data_folder: Path to folder containing audio files
            output_folder: Path to folder where results will be saved
        """
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        
        # Initialize OpenAI client for transcription
        self.transcription_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize LangChain LLM for summarization
        self.llm = ChatOpenAI(
            model="gpt-5-mini",
            temperature=1,
            api_key=os.getenv("OPENAI_API_KEY"),
            response_format={"type": "json_object"},
        )
        
        # Create summarization chain using LangChain
        summary_prompt = PromptTemplate(
            input_variables=["transcript"],
            template="""Please provide a detailed summary (4–5 paragraphs) of the following diary entry.
Write the summary in first person, using “I,” describing what I did, what I experienced, and how I felt.

Use the same language as the majority of the provided transcript (do not translate unless explicitly asked).

Return the response as a JSON object with the following structure:
{{
  "Summary": "The detailed summary here (4-5 paragraphs)",
  "Insights & Suggestions": "Two practical, thoughtful insights or suggestions derived from the diary entry. These should be actionable and relevant to my emotional state, habits, or decisions described in the text."
}}

{transcript}"""
        )
        self.summary_chain = summary_prompt | self.llm

        # Create output folders if they don't exist
        self.output_folder.mkdir(exist_ok=True)
        self.meta_folder = self.output_folder / "meta"
        self.meta_folder.mkdir(exist_ok=True)

        if not self.data_folder.exists():
            raise FileNotFoundError(f"Data folder not found: {self.data_folder}")

    def get_audio_files(self) -> list[Path]:
        """
        Get all .m4a audio files from the data folder.

        Returns:
            List of Path objects for audio files formatted as YYYY-MM-DD.m4a
        """
        audio_files = sorted(self.data_folder.glob("*.m4a"))

        if not audio_files:
            print(f"No .m4a files found in {self.data_folder}")
            return []

        # Validate filename format
        valid_files = []
        for file in audio_files:
            try:
                # Check if filename matches YYYY-MM-DD format
                filename = file.stem  # Get filename without extension
                datetime.strptime(filename, "%Y-%m-%d")
                valid_files.append(file)
            except ValueError:
                print(
                    f"Warning: Skipping {file.name} - filename doesn't match YYYY-MM-DD format"
                )

        return valid_files

    def transcribe_audio(self, audio_file: Path) -> Optional[str]:
        """
        Transcribe audio file using OpenAI Whisper API.

        Args:
            audio_file: Path to audio file

        Returns:
            Transcribed text or None if transcription fails
        """
        try:
            print(f"Transcribing {audio_file.name}...")

            with open(audio_file, "rb") as f:
                transcript = self.transcription_client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe", 
                    file=f,
                    prompt="This is a personal diary entry."
                )

            print(f"✓ Transcription complete for {audio_file.name}")
            return transcript.text

        except FileNotFoundError:
            print(f"Error: Audio file not found: {audio_file}")
            return None
        except Exception as e:
            print(f"Error transcribing {audio_file.name}: {str(e)}")
            return None

    def load_transcripts_from_json(self) -> dict:
        """
        Load existing transcripts from the meta folder.

        Returns:
            Dictionary with existing transcripts, or empty dict if no files found
        """
        if not self.meta_folder.exists():
            print(f"No meta folder found at {self.meta_folder}")
            return {}
        
        try:
            meta_files = sorted(self.meta_folder.glob("*_meta.json"))
            if not meta_files:
                print(f"No transcript files found in {self.meta_folder}")
                return {}
            
            data = {}
            for meta_file in meta_files:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                    # Extract date from filename (e.g., "2025-12-19_meta.json" -> "2025-12-19")
                    date = meta_file.stem.replace("_meta", "")
                    data[date] = meta_data
            
            print(f"✓ Loaded {len(data)} transcript(s) from {self.meta_folder}")
            return data
        except Exception as e:
            print(f"Error loading transcripts: {str(e)}")
            return {}

    def summarize_text(self, text: str, date: str) -> Optional[dict]:
        """
        Summarize transcribed text using LangChain and OpenAI GPT.

        Args:
            text: Transcribed text to summarize
            date: Date from filename for context

        Returns:
            Dictionary with 'summary' and 'Insights & Suggestions' keys, or None if summarization fails
        """
        try:
            print(f"Summarizing content from {date}...")

            # Use LangChain chain to generate summary
            result = self.summary_chain.invoke({
                "date": date,
                "transcript": text
            })

            # Parse JSON response
            summary_data = json.loads(result.content)
            print(f"✓ Summarization complete for {date}")
            return summary_data

        except json.JSONDecodeError:
            print(f"Error: Invalid JSON response from summarization for {date}")
            return None
        except Exception as e:
            print(f"Error summarizing content from {date}: {str(e)}")
            return None

    def process_all_files(self) -> dict:
        """
        Process all audio files: transcribe and summarize.

        Returns:
            Dictionary with results for all processed files
        """
        audio_files = self.get_audio_files()

        if not audio_files:
            print("No valid audio files to process.")
            return {}

        results = {}

        for audio_file in audio_files:
            date = audio_file.stem  # YYYY-MM-DD format

            # Transcribe
            transcript = self.transcribe_audio(audio_file)
            if not transcript:
                continue

            # Summarize
            summary = self.summarize_text(transcript, date)
            if not summary:
                continue

            results[date] = {
                "filename": audio_file.name,
                "transcript": transcript,
                "summary": summary,
                "processed_at": datetime.now().isoformat(),
            }

        return results

    def summarize_only(self, specific_date: Optional[str] = None) -> dict:
        """
        Summarize only using existing transcripts from JSON file.
        This skips the transcription step entirely.

        Args:
            specific_date: If provided, only summarize this date (YYYY-MM-DD format)

        Returns:
            Dictionary with updated results including new summaries
        """
        results = self.load_transcripts_from_json()

        if not results:
            print("No transcripts found to summarize.")
            return {}

        # Filter to specific date if provided
        if specific_date:
            if specific_date not in results:
                print(f"Error: No transcript found for date {specific_date}")
                return {}
            results = {specific_date: results[specific_date]}

        print("\nStarting summarization-only mode...")
        updated_results = {}

        for date, data in results.items():
            if "transcript" not in data:
                print(f"⚠ Skipping {date}: No transcript found")
                continue

            # Summarize using existing transcript
            summary = self.summarize_text(data["transcript"], date)
            if not summary:
                continue

            updated_results[date] = {
                "filename": data.get("filename", f"{date}.m4a"),
                "transcript": data["transcript"],
                "summary": summary,
                "processed_at": datetime.now().isoformat(),
            }

        return updated_results

    def process_single_file(self, file_date: str) -> dict:
        """
        Process a single audio file by date.

        Args:
            file_date: Date in YYYY-MM-DD format

        Returns:
            Dictionary with result for the processed file
        """
        audio_files = self.get_audio_files()
        
        # Find the file matching the date
        target_file = None
        for audio_file in audio_files:
            if audio_file.stem == file_date:
                target_file = audio_file
                break
        
        if not target_file:
            print(f"Error: No audio file found for date {file_date}")
            return {}
        
        results = {}
        
        # Transcribe
        transcript = self.transcribe_audio(target_file)
        if not transcript:
            return {}
        
        # Summarize
        summary = self.summarize_text(transcript, file_date)
        if not summary:
            return {}
        
        results[file_date] = {
            "filename": target_file.name,
            "transcript": transcript,
            "summary": summary,
            "processed_at": datetime.now().isoformat(),
        }
        
        return results

    def save_results(self, results: dict) -> None:
        """
        Save results to metadata JSON files and individual markdown files.

        Args:
            results: Dictionary of processing results
        """
        if not results:
            print("No results to save.")
            return

        # Save individual meta JSON and markdown files
        for date, data in results.items():
            # Save metadata JSON
            meta_file = self.meta_folder / f"{date}_meta.json"
            meta_data = {
                "filename": data["filename"],
                "transcript": data["transcript"],
                "summary": data["summary"],
                "processed_at": data["processed_at"]
            }
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Meta JSON saved: {meta_file}")
            
            # Save markdown file
            md_file = self.output_folder / f"{date}_summary.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(f"# Diary Entry - {date}\n\n")
                f.write("## Transcript\n\n")
                f.write(f"{data['transcript']}\n\n")
                f.write("## Summary\n\n")
                f.write(f"{data['summary']['summary']}\n\n")
                f.write("## Insights & Suggestions\n\n")
                f.write(f"{data['summary']['Insights & Suggestions']}\n")
            print(f"✓ Markdown file saved: {md_file}")

    def run(self) -> None:
        """Run the complete pipeline: transcribe and summarize all files."""
        print("=" * 60)
        print("Audio Diary Transcription & Summarization System")
        print("=" * 60)

        results = self.process_all_files()

        if results:
            self.save_results(results)
            print("\n" + "=" * 60)
            print(f"Processing complete! Processed {len(results)} file(s).")
            print("=" * 60)
        else:
            print("\nNo files were successfully processed.")

    def run_summarize_only(self, specific_date: Optional[str] = None) -> None:
        """
        Run summarization only using existing transcripts.
        
        Args:
            specific_date: If provided, only summarize this date (YYYY-MM-DD format)
        """
        print("=" * 60)
        print("Audio Diary Summarization Only (No Transcription)")
        print("=" * 60)

        results = self.summarize_only(specific_date)

        if results:
            self.save_results(results)
            print("\n" + "=" * 60)
            print(f"Summarization complete! Summarized {len(results)} file(s).")
            print("=" * 60)
        else:
            print("\nNo files were successfully summarized.")

    def run_single_file(self, file_date: str) -> None:
        """
        Run full pipeline (transcribe + summarize) for a single file.
        
        Args:
            file_date: Date in YYYY-MM-DD format
        """
        print("=" * 60)
        print(f"Processing Single File: {file_date}")
        print("=" * 60)

        results = self.process_single_file(file_date)

        if results:
            self.save_results(results)
            print("\n" + "=" * 60)
            print(f"Processing complete! Processed 1 file.")
            print("=" * 60)
        else:
            print("\nFile processing failed.")


if __name__ == "__main__":
    import sys
    
    summarizer = AudioDiarySummarizer()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--summarize-only":
            # Optional: specify date after --summarize-only
            date = sys.argv[2] if len(sys.argv) > 2 else None
            summarizer.run_summarize_only(date)
        elif sys.argv[1] == "--file":
            # Requires date argument: python audio_processor.py --file 2025-12-19
            if len(sys.argv) < 3:
                print("Error: --file requires a date argument (YYYY-MM-DD format)")
                print("Usage: python audio_processor.py --file 2025-12-19")
                sys.exit(1)
            summarizer.run_single_file(sys.argv[2])
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("\nUsage:")
            print("  python audio_processor.py                           # Process all files")
            print("  python audio_processor.py --file 2025-12-19         # Process single file")
            print("  python audio_processor.py --summarize-only          # Summarize all existing transcripts")  # summarize-only is mostly used to test prompts without need to transcribe again
            print("  python audio_processor.py --summarize-only 2025-12-19 # Summarize one existing transcript")
            sys.exit(1)
    else:
        summarizer.run()
