from textblob import TextBlob
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import yfinance as yf
from datetime import datetime, timedelta


def analyze_multiple_transcripts(folder_path):
    """
    Reads all .txt transcripts in the given folder and returns
    average sentiment for each file.
    """
    summary_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            text = load_transcript(file_path)
            if not text:
                continue

            cleaned = clean_transcript(text)
            blob = TextBlob(cleaned)
            analyzer = SentimentIntensityAnalyzer()

            tb_avg = sum(s.sentiment.polarity for s in blob.sentences) / len(blob.sentences)
            vader_avg = sum(analyzer.polarity_scores(str(s))["compound"] for s in blob.sentences) / len(blob.sentences)

            summary_data.append({
                "file": filename,
                "TextBlob_avg": tb_avg,
                "VADER_avg": vader_avg
            })

            # --- Optional: add stock change data if filename has a date ---
        try:
            if "q1" in filename.lower():
                call_date = datetime(2024, 1, 25)
            elif "q2" in filename.lower():
                call_date = datetime(2024, 4, 25)
            elif "q3" in filename.lower():
                call_date = datetime(2024, 7, 25)
            elif "q4" in filename.lower():
                call_date = datetime(2024, 10, 25)
            else:
                call_date = None

            if call_date:
                stock_change = analyze_stock_reaction("AAPL", call_date)
            else:
                stock_change = None
        except Exception as e:
            stock_change = None

        summary_data[-1]["StockChange_%"] = stock_change
    

    return pd.DataFrame(summary_data)

def analyze_stock_reaction(ticker="AAPL", call_date=None, days_after=5):
    """
    Fetches stock data around the earnings call date and calculates
    the % change after the call.
    """
    if call_date is None:
        print("No date provided for stock comparison.")
        return None

    start = call_date - timedelta(days=2)
    end = call_date + timedelta(days=days_after)
    data = yf.download(ticker, start=start, end=end, progress=False)

    if data.empty:
        print("No stock data found for the given range.")
        return None

    before = float(data["Close"].iloc[0])
    after = float(data["Close"].iloc[-1])

    # Calculate percent change safely
    change = ((after - before) / before) * 100

    print(f"\nStock price change for {ticker} ({days_after} days after call): {change:.2f}%")

    return change



# --- Step 1: Function to clean the transcript ---
def clean_transcript(text):
    """
    Removes unwanted elements from the transcript such as 'Applause', 
    blank lines, and any extra whitespace.
    """
    text = re.sub(r'\(Applause\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?\]', '', text)  # remove bracketed text like [Laughter]
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    return text.strip()


# --- Step 2: Load the transcript file ---
def load_transcript(file_path):
    """
    Loads the transcript from a text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print("Transcript loaded successfully.")
        return text
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None


# --- Step 3: Main sentiment analysis logic ---
def main():
    # Load and clean the transcript
    file_path = "data/sample_transcript"
    transcript = load_transcript(file_path)
    if not transcript:
        return

    cleaned_text = clean_transcript(transcript)

    # Split into sentences for TextBlob
    blob = TextBlob(cleaned_text)
    sentences = blob.sentences

    # Initialize VADER
    analyzer = SentimentIntensityAnalyzer()

    print("\nSentiment Analysis Results:\n")

    # Prepare containers for results
    results_data = []
    speaker_data = defaultdict(list)

    # Loop through each sentence and compare both models
    for sentence in sentences:
        text = str(sentence)
        tb_polarity = sentence.sentiment.polarity
        vader_scores = analyzer.polarity_scores(text)

        print(f"Sentence: {text}")
        print(f"  → TextBlob polarity: {tb_polarity:.2f}")
        print(f"  → VADER compound: {vader_scores['compound']:.2f}")
        print(f"     (pos={vader_scores['pos']:.2f}, neu={vader_scores['neu']:.2f}, neg={vader_scores['neg']:.2f})")
        print("-" * 60)

        # Collect sentence-level results
        results_data.append({
            "sentence": text,
            "TextBlob_polarity": tb_polarity,
            "VADER_compound": vader_scores["compound"]
        })

        # Attempt to extract speaker (e.g., "Speaker: ...")
        match = re.match(r"([A-Za-z ]+):", text)
        if match:
            speaker = match.group(1).strip()
            speaker_data[speaker].append((tb_polarity, vader_scores["compound"]))

    # --- Step 4: Store sentence-level results in a DataFrame ---
    results_df = pd.DataFrame(results_data)
    print("\nSentence-level results stored in a DataFrame:\n")
    print(results_df.head())

    # --- Step 5: Aggregate sentiment by speaker ---
    print("\nAverage Sentiment by Speaker:\n")
    speaker_summary = []
    for speaker, values in speaker_data.items():
        if values:
            avg_tb = sum(v[0] for v in values) / len(values)
            avg_vader = sum(v[1] for v in values) / len(values)
            print(f"{speaker}: TextBlob ={avg_tb:.2f}, VADER ={avg_vader:.2f}")
            speaker_summary.append({
                "speaker": speaker,
                "TextBlob_avg": avg_tb,
                "VADER_avg": avg_vader
            })

    # Build and plot speaker summary if any speakers found
    if speaker_summary:
        df_summary = pd.DataFrame(speaker_summary)
        plt.figure(figsize=(8, 5))
        bar_width = 0.35

        # Set positions for bars
        x = range(len(df_summary))
        plt.bar([p - bar_width/2 for p in x], df_summary["TextBlob_avg"], 
                width=bar_width, label="TextBlob", color="#4C72B0")
        plt.bar([p + bar_width/2 for p in x], df_summary["VADER_avg"], 
                width=bar_width, label="VADER", color="#DD8452")

        # Add labels and title
        plt.xticks(x, df_summary["speaker"], rotation=20, ha="right")
        plt.title("Average Sentiment by Speaker", fontsize=14, pad=15)
        plt.xlabel("Speaker")
        plt.ylabel("Sentiment Score")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

        # --- Step 6: Export results to CSV files ---
    try:
        results_df.to_csv("data/sentence_results.csv", index=False)
        if speaker_summary:
            df_summary.to_csv("data/speaker_summary.csv", index=False)
        print("\nSentiment data exported successfully:")
        print(" - data/sentence_results.csv (all sentences)")
        print(" - data/speaker_summary.csv (average by speaker)")
    except Exception as e:
        print(f"Error saving CSV files: {e}")

        # --- Step 7: Visualize sentiment trend across transcript ---
    if not results_df.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(results_df["TextBlob_polarity"], label="TextBlob", color="#4C72B0", marker="o")
        plt.plot(results_df["VADER_compound"], label="VADER", color="#DD8452", marker="o")

        plt.title("Sentiment Trend Across Transcript", fontsize=14, pad=15)
        plt.xlabel("Sentence Number")
        plt.ylabel("Sentiment Score")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

        # --- Step 8: Compare multiple transcripts (optional) ---
    folder_path = "transcripts"
    if os.path.exists(folder_path):
        multi_df = analyze_multiple_transcripts(folder_path)
        print("\nOverall Sentiment by Transcript:\n")
        print(multi_df)

        # Plot comparison
        if not multi_df.empty:
            multi_df.plot(
                kind="bar",
                x="file",
                y=["TextBlob_avg", "VADER_avg"],
                figsize=(8, 5),
                title="Average Sentiment by Transcript"
            )
            plt.ylabel("Average Sentiment")
            plt.tight_layout()
            plt.show()
                # --- Step 10: Visualize sentiment vs stock movement ---
        if "StockChange_%" in multi_df.columns:
            plt.figure(figsize=(8, 5))
            plt.scatter(multi_df["VADER_avg"], multi_df["StockChange_%"], color="#DD8452", label="VADER")
            plt.scatter(multi_df["TextBlob_avg"], multi_df["StockChange_%"], color="#4C72B0", label="TextBlob")

            # Label each point with the transcript filename
            for i, row in multi_df.iterrows():
                plt.text(row["VADER_avg"], row["StockChange_%"], row["file"], fontsize=8, ha="left")

            plt.title("Sentiment vs. Stock Reaction", fontsize=14, pad=15)
            plt.xlabel("Average Sentiment Score")
            plt.ylabel("Stock % Change (5 days after call)")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.show()
            

        # --- Step 9: Compare sentiment to stock reaction ---
    call_date = datetime(2024, 7, 25)  # Example: Apple Q3 2024 earnings call date
    analyze_stock_reaction(ticker="AAPL", call_date=call_date)
    

if __name__ == "__main__":
    main()