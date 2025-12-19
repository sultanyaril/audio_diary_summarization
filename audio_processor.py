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
            model="gpt-4",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create summarization chain using LangChain
        summary_prompt = PromptTemplate(
            input_variables=["date", "transcript"],
            template="""Please provide a concise summary (2-3 paragraphs) of the following diary entry from {date}:

{transcript}

Summary:"""
        )
        
        self.summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt)

        # Create output folder if it doesn't exist
        self.output_folder.mkdir(exist_ok=True)

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
                    model="whisper-1", file=f
                )

            print(f"✓ Transcription complete for {audio_file.name}")
            return transcript.text

        except FileNotFoundError:
            print(f"Error: Audio file not found: {audio_file}")
            return None
        except Exception as e:
            print(f"Error transcribing {audio_file.name}: {str(e)}")
            return None

    def summarize_text(self, text: str, date: str) -> Optional[str]:
        """
        Summarize transcribed text using LangChain and OpenAI GPT.

        Args:
            text: Transcribed text to summarize
            date: Date from filename for context

        Returns:
            Summary or None if summarization fails
        """
        try:
            print(f"Summarizing content from {date}...")

            # Use LangChain chain to generate summary
            result = self.summary_chain.invoke({
                "date": date,
                "transcript": text
            })

            summary = result.get("text", "")
            print(f"✓ Summarization complete for {date}")
            return summary

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

    def save_results(self, results: dict) -> None:
        """
        Save results to JSON and individual markdown files.

        Args:
            results: Dictionary of processing results
        """
        if not results:
            print("No results to save.")
            return

        # Save combined JSON
        json_output = self.output_folder / "all_summaries.json"
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to {json_output}")

        # Save individual markdown files
        for date, data in results.items():
            md_file = self.output_folder / f"{date}_summary.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(f"# Diary Entry - {date}\n\n")
                f.write("## Transcript\n\n")
                f.write(f"{data['transcript']}\n\n")
                f.write("## Summary\n\n")
                f.write(f"{data['summary']}\n")
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


if __name__ == "__main__":
    summarizer = AudioDiarySummarizer()
    summarizer.run()
