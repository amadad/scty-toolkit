import google.generativeai as genai
from pytube import YouTube
import os
import requests

SERP_API_KEY = os.getenv("SERPAPI_API_KEY")  # Replace with your SERP API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Replace with your Google API key

genai.configure(api_key=GOOGLE_API_KEY)

def search_youtube_videos(search_term):
    params = {
      "api_key": SERP_API_KEY,
      "engine": "youtube",
      "search_query": search_term,

    }
    response = requests.get("https://serpapi.com/search", params=params)
    results = response.json()
    video_results = results.get('video_results', [])[:5]
    video_links = [result['link'] for result in video_results]
    return video_links

def download_audio(video_url):
    yt = YouTube(video_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_file = "audio.mp3"
    audio_stream.download(filename=audio_file)
    return audio_file

def summarize_video(audio_file):
    your_file = genai.upload_file(audio_file)
    prompt = f"Listen carefully to the following audio file and provide a detailed summary of the content."
    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
    response = model.generate_content([prompt, your_file])
    return response.text

def generate_report(summaries):
    prompt = f"Given the following video summaries:\n\n{summaries}\n\nGenerate a comprehensive report that combines the information from all the summaries. The report should be well-structured and cover the main points discussed in the videos."
    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
    response = model.generate_content(prompt)
    return response.text

def main():
    topic = input("Enter the overall topic: ")
    video_results = search_youtube_videos(topic)

    summaries = []
    for video in video_results:
        print(video)
        audio_file = download_audio(video)
        summary = summarize_video(audio_file)
        summaries.append(summary)
        os.remove(audio_file)  # Remove the downloaded audio file

    report = generate_report('\n\n'.join(summaries))
    print("Final Report:")
    print(report)

main()
     