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
    
    # Reads all .txt transcripts in the given folder and returns average sentiment for each file.
   
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

            # adds the stock change data to the analysis if filename has a certain date already written in it
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
    
    # Takes the stock data from around the time of earnings call date and calculates change (%) after the call was released
    
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



# Function to clean the transcript of superfluous elements
def clean_transcript(text):
    # Remove bracketed/parenthetical stage directions
    text = re.sub(r'\(Applause\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?\]', '', text)

    # Collapse spaces and tabs, but KEEP line breaks
    text = re.sub(r'[ \t]+', ' ', text)

    # Reduce huge blank gaps but keep paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

  # I might add more cleaning steps later for things like timestamps


# retrieves transcript from a text file
def load_transcript(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print("Transcript loaded successfully.")
        return text
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

# speaker detection works across most transcripts, avoids false hits
def detect_speaker(line: str, last: str) -> str:
    line = line.strip()

    # Case A: "Tim Cook: Thank you..." or "Operator – Good afternoon..."
    m = re.match(
        r"^((?:[A-Z][\w'’\-\.]+(?:\s+[A-Z][\w'’\-\.]+){0,3})|"
        r"Operator|Analyst|Moderator|Host|Participant|Unidentified Analyst)"
        r"\s*[:\-–—]\s+",         # NOTE: punctuation is REQUIRED here
        line
    )
    if m:
        return m.group(1).title()

    # Case B: standalone label line: "Tim Cook" / "Operator" 
    m2 = re.match(
        r"^((?:[A-Z][\w'’\-\.]+(?:\s+[A-Z][\w'’\-\.]+){0,3})|"
        r"Operator|Analyst|Moderator|Host|Participant|Unidentified Analyst)$",
        line
    )
    if m2:
        return m2.group(1).title()

    # No new speaker on this line = keep previous
    return last


from collections import defaultdict

def segment_by_speaker(cleaned_text: str) -> dict:
    current = "Unknown"
    buckets = defaultdict(list)

    for raw_line in cleaned_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        new_speaker = detect_speaker(line, current)

        # If line is only the label itself, switch speaker and skip adding this line as content
        if new_speaker != current:
            current = new_speaker
            if re.fullmatch(rf'{re.escape(current)}', line, flags=re.IGNORECASE) or \
               re.fullmatch(rf'{re.escape(current)}\s*[:\-–—]?', line, flags=re.IGNORECASE):
                continue

        # Otherwise, this is content spoken by `current`
        buckets[current].append(line)

    # Join each speaker’s lines into one block of text
    return {spk: ' '.join(lines) for spk, lines in buckets.items()}


# Main logic for the sentiment analysis 
def main():
    # Load and clean transcript
    file_path = "transcripts/apple_q3_2025.txt"
    transcript = load_transcript(file_path)
    if not transcript:
        return

    cleaned_text = clean_transcript(transcript)
    # Build speaker blocks before sentence tokenization
    speaker_blocks = segment_by_speaker(cleaned_text)


    # Split into single sentences
    blob = TextBlob(cleaned_text)
    sentences = blob.sentences

    # Initializing VADER
    analyzer = SentimentIntensityAnalyzer()

    print("\nSentiment Analysis Results:\n")

    # Containers for results
    results_data = []
    speaker_data = defaultdict(list)

    # Loops through each sentence and compare both models
    for sentence in sentences:
        text = str(sentence)
        tb_polarity = sentence.sentiment.polarity
        vader_scores = analyzer.polarity_scores(text)

        print(f"Sentence: {text}")
        print(f"  → TextBlob polarity: {tb_polarity:.2f}")
        print(f"  → VADER compound: {vader_scores['compound']:.2f}")
        print(f"     (pos={vader_scores['pos']:.2f}, neu={vader_scores['neu']:.2f}, neg={vader_scores['neg']:.2f})")
        print("-" * 60)

        # Collects all sentence-level results
        results_data.append({
            "sentence": text,
            "TextBlob_polarity": tb_polarity,
            "VADER_compound": vader_scores["compound"]
        })

        # Initialize a default speaker if none yet
        if 'last_speaker' not in locals():
            last_speaker = "Unknown"

        # Update current speaker if a name/role appears at the start of the line
        last_speaker = detect_speaker(text, last_speaker)

        # Record this sentence’s sentiment under that speaker
        speaker_data[last_speaker].append((tb_polarity, vader_scores["compound"]))



    # Putting results for individual sentences into DataFrame 
    results_df = pd.DataFrame(results_data)
    print("\nSentence-level results stored in a DataFrame:\n")
    print(results_df.head())

    # Speaker-level averages 
    speaker_summary = []
    vd = SentimentIntensityAnalyzer()

    for speaker, block in speaker_blocks.items():
        if speaker.lower() == "unknown":
            continue
        b = TextBlob(block)
        if len(b.sentences) == 0:
            continue
        tb_avg = sum(s.sentiment.polarity for s in b.sentences) / len(b.sentences)
        vd_avg = sum(vd.polarity_scores(str(s))['compound'] for s in b.sentences) / len(b.sentences)
        speaker_summary.append({"speaker": speaker, "TextBlob_avg": tb_avg, "VADER_avg": vd_avg})

    print("Speakers detected:", [row["speaker"] for row in speaker_summary])

    # Plot (only if speakers were found)
    if speaker_summary:
        df_summary = pd.DataFrame(speaker_summary).groupby("speaker", as_index=False).mean()
        plt.figure(figsize=(9, 5))
        bar_width = 0.35
        x = range(len(df_summary))

        plt.bar([p - bar_width/2 for p in x], df_summary["TextBlob_avg"],
            width=bar_width, label="TextBlob", color="#4C72B0")
        plt.bar([p + bar_width/2 for p in x], df_summary["VADER_avg"],
            width=bar_width, label="VADER", color="#DD8452")

        plt.xticks(x, df_summary["speaker"], rotation=25, ha="right", fontsize=9)
        plt.title("Average Sentiment by Speaker", fontsize=14, pad=15)
        plt.xlabel("Speaker")
        plt.ylabel("Sentiment Score")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()
    else:
        print("No speaker labels detected. Check transcript format or relax detection.")


        # Results export to CSV files
    try:
        results_df.to_csv("data/sentence_results.csv", index=False)
        if speaker_summary:
            df_summary.to_csv("data/speaker_summary.csv", index=False)
        print("\nSentiment data exported successfully:")
        print(" - data/sentence_results.csv (all sentences)")
        print(" - data/speaker_summary.csv (average by speaker)")
    except Exception as e:
        print(f"Error saving CSV files: {e}") # I will probably figure out a way to handle a missing speaker better here

        # Graph overview of sentiment (the trend across entirety of transcript)
    if not results_df.empty:
        plt.figure(figsize=(10, 5))

        x = range(len(results_df))
        
        plt.plot(results_df["TextBlob_polarity"], label="TextBlob", color="#4C72B0", marker="o")
        plt.plot(results_df["VADER_compound"], label="VADER", color="#DD8452", marker="o")

        plt.title("Sentiment Trend Across Transcript", fontsize=14, pad=15)
        plt.xlabel("Sentence Number")
        plt.ylabel("Sentiment Score")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

        # Compares multiple transcripts if there are any
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
                # Sentiment vs stock movement plot
        if "StockChange_%" in multi_df.columns:
            plt.figure(figsize=(8, 5))
            plt.scatter(multi_df["VADER_avg"], multi_df["StockChange_%"], color="#DD8452", label="VADER")
            plt.scatter(multi_df["TextBlob_avg"], multi_df["StockChange_%"], color="#4C72B0", label="TextBlob")

            # Labels each point with the the filename of individual transcript
            for i, row in multi_df.iterrows():
                plt.text(row["VADER_avg"], row["StockChange_%"], row["file"], fontsize=8, ha="left")

            plt.title("Sentiment vs. Stock Reaction", fontsize=14, pad=15)
            plt.xlabel("Average Sentiment Score")
            plt.ylabel("Stock % Change (5 days after call)")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.show()
            

        # Compare sentiment to stock reaction 
    call_date = datetime(2024, 7, 25)  # Example: Apple Q3 2024 earnings call date
    analyze_stock_reaction(ticker="AAPL", call_date=call_date)
    

if __name__ == "__main__":
    main()