import yt_dlp
import assemblyai as aai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
import json
from textwrap import dedent
import os
import subprocess

load_dotenv()

def download_youtube_video(url):
    ydl_opts = {
        'format': 'mp4/bestaudio/best',
        'outtmpl': '%(id)s.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        file_name = ydl.prepare_filename(info_dict)

    return file_name

def get_transcription(video_path):
    config = aai.TranscriptionConfig(
        speaker_labels=True,
        iab_categories=True,
        auto_chapters=True
    )

    config.set_custom_spelling(
    {
        "LangGraph": ["Landgraaf", "Landgraph"],
        "LangChain": ["Lantent"]
    }
    )

    transcript = aai.Transcriber().transcribe(video_path, config)

    for utterance in transcript.utterances:
        print(f"Speaker {utterance.speaker}: {utterance.text}")

    srt = transcript.export_subtitles_srt()
   
    return srt

def extract_highlights_timestamps(srt):
    client = OpenAI()
    MODEL = "gpt-4o-2024-08-06"

    class Clip(BaseModel):
        clip_transcription: str = Field(..., description="Full transcription of the clip")
        start_time: str = Field(..., description="starting timestamp of the highlight clip, should be in HH:MM:SS,SSS format")
        end_time: str = Field(..., description="ending timestamp of the highlight clip, should be in HH:MM:SS,SSS format")

    class Highlight(BaseModel):
        highlight_title: str = Field(..., description="title of the highlight clip that cover the main point")
        highlight_main_points: str = Field(..., description="what are main points of the highlight based on video transcription")
        highlight_clips: list[Clip] = Field(..., description="relevant clips of the main point, each clip should not repeat similar content with each other; in total no more than 60 seconds in total")
        
    class Highlights(BaseModel):
        main_highlights_reasoning: list[str] = Field(..., description="Analyse transcription and think through what main highlights could be, max 3 highlights")
        highlights: list[Highlight] = Field(..., description='core highlight clips extracted from the transcript')

    Timestamp_extraction_prompt = '''
        You are a world class video editor, 
        Based on full video transcript you were given, try to extract main highlights that can be taken out as video short;
        Each highlight should be self explaining & convey core concepts
    '''

    def get_highlights(str):
        response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system", 
                "content": dedent(Timestamp_extraction_prompt)
            },
            {
                "role": "user", 
                "content": json.dumps(str)
            }
        ],
        response_format=Highlights
        )

        highlights_raw = json.loads(response.choices[0].message.content)
        highlight_clips = []

        for highlight in highlights_raw['highlights']:
            title = highlight['highlight_title']
            timestamps = []

            for clip in highlight['highlight_clips']:
                timestamps.append({
                    'start_time': clip['start_time'],
                    'end_time': clip['end_time']
                })
            
            highlight_clips.append({
                'title': title,
                'timestamps': timestamps
            })
        
        return highlight_clips

    return get_highlights(srt)

def trim_highlights(highlights, video_path):
    input_video_path = video_path

    def trim_and_concat_video(input_path, output_path, timestamps):
        temp_files = []
        
        try:
            # Extract clips based on timestamps
            for i, t in enumerate(timestamps):
                start_time = t["start_time"].replace(',', '.')
                end_time = t["end_time"].replace(',', '.')
                temp_output = f"temp_clip_{i}.mp4"
                command = [
                    "ffmpeg",
                    "-i", input_path,
                    "-ss", start_time,
                    "-to", end_time,
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-strict", "experimental",
                    temp_output
                ]
                subprocess.run(command, check=True)
                temp_files.append(temp_output)
            
            # Create a file containing the list of temp files
            with open("temp_file_list.txt", "w") as f:
                for temp_file in temp_files:
                    f.write(f"file '{temp_file}'\n")
            
            # Concatenate clips
            concat_command = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", "temp_file_list.txt",
                "-c", "copy",
                output_path
            ]
            subprocess.run(concat_command, check=True)
        
        finally:
            # Clean up temp files
            for temp_file in temp_files:
                os.remove(temp_file)
            if os.path.exists("temp_file_list.txt"):
                os.remove("temp_file_list.txt")

    for highlight in highlights:
        output_video_path = f"{highlight['title']}.mp4"
        timestamps = highlight['timestamps']
        trim_and_concat_video(input_video_path, output_video_path, timestamps)

url = "https://www.youtube.com/watch?v=IW7jFq3vQbw"
video_path = download_youtube_video(url)
transcription = get_transcription(video_path)
highlights = extract_highlights_timestamps(transcription)

trim_highlights(highlights, video_path)