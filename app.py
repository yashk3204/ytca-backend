from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

from googleapiclient.discovery import build
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=API_KEY)
sia = SentimentIntensityAnalyzer()
english_stopwords = set(stopwords.words('english'))

def get_results(video_id, max_comments=100):
    comments = []
    next_page_token = None

    video_response = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()

    if not video_response["items"]:
        raise ValueError("Video not found")

    snippet = video_response["items"][0]["snippet"]
    title = snippet["title"]
    channel = snippet["channelTitle"]
    thumbnail = snippet["thumbnails"]["medium"]["url"]

    while len(comments) < max_comments:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText"
        ).execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return title, channel, thumbnail, comments

def clean_comment(text):
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove emojis, special chars
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in english_stopwords]
    return " ".join(words)

def get_sentiment(comment):
    score = sia.polarity_scores(comment)['compound']
    if score >= 0.2:
        return 'Positive'
    elif score <= -0.2:
        return 'Negative'
    else:
        return 'Neutral'

@app.route("/analyze", methods=["POST"])

def analyze():
    data = request.get_json()
    video_id = data.get("videoId")
    count = int(data.get("count", 50))

    try:
        title, channel, thumbnail, comments = get_results(video_id, count)
        cleaned = [clean_comment(c) for c in comments]
        results = {"Positive": [], "Neutral": [], "Negative": []}
        for raw, cleaned in zip(comments, cleaned):
            sentiment = get_sentiment(cleaned)
            results[sentiment].append(raw)

        return jsonify({
            "title": title,
            "channel": channel,
            "thumbnail": thumbnail,
            "positive": len(results["Positive"]),
            "neutral": len(results["Neutral"]),
            "negative": len(results["Negative"]),
            "comments": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))