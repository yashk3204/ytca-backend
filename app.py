from flask import Flask, request, jsonify
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
import requests

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
sia = SentimentIntensityAnalyzer()
english_stopwords = set(stopwords.words('english'))

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

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
    
def getSummary(comments, isPositive, title, channel):
    if len(comments) == 0:
        return ""
    sentiment = "positive" if isPositive else "negative"
    prompt = (f"Summarize the following {sentiment} comments from the YouTube video \"{title}\" by \"{channel}\":" + "\n".join(f"- {c}" for c in comments[:30]))
    url = "https://api.perplexity.ai/chat/completions"

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that summarizes YouTube comments."}, 
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error from Perplexity: {response.text}")

# import matplotlib.pyplot as plt
# from collections import Counter

# def plot_sentiments(sentiments):
#     sent_counts = Counter(sentiments)
#     labels = ['Positive', 'Neutral', 'Negative']
#     values = [sent_counts.get(label, 0) for label in labels]

#     plt.figure(figsize=(6, 4))
#     plt.bar(labels, values, color=['green', 'gray', 'red'])
#     plt.title("Sentiment Distribution")
#     plt.ylabel("Number of Comments")
#     plt.tight_layout()
#     os.makedirs('static', exist_ok=True)
#     chart_path = os.path.join('static', 'sentiment_plot.png')
#     plt.savefig(chart_path)
#     plt.close()
#     return chart_path

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
    
@app.route("/summarize", methods=["POST"])

def summarize():
    data = request.get_json()
    posComments = data.get("positive_comments", [])
    negComments = data.get("negative_comments", [])
    title = data.get("title")
    channel = data.get("channel")

    try:
        posSummary = getSummary(posComments, True, title, channel)
    except Exception as e:
        posSummary = f"Error summarizing positive comments: {e}"
    try:
        negSummary = getSummary(negComments, False, title, channel)
    except Exception as e:
        negSummary = f"Error summarizing negative comments: {e}"

    return jsonify({
        "positiveSummary": posSummary,
        "negativeSummary": negSummary
    })

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)