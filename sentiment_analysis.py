import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Download VADER Lexicon
nltk.download('vader_lexicon')

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Analyze the sentiment of a given text."""
    sentiment_score = sia.polarity_scores(text)
    compound_score = sentiment_score['compound']
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def analyze_text_file(file_path):
    """Analyze sentiment of text data from a file (CSV or TXT)."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            if 'text' in df.columns:
                df['Sentiment'] = df['text'].apply(analyze_sentiment)
                print(df[['text', 'Sentiment']])
                plt.figure(figsize=(8, 5))
                sns.countplot(x=df['Sentiment'], palette='coolwarm')
                plt.title('Sentiment Distribution')
                plt.xlabel('Sentiment')
                plt.ylabel('Count')
                plt.show()
            else:
                print("CSV file must have a column named 'text'.")
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                sentiment = analyze_sentiment(text)
                print(f'Sentiment: {sentiment}')
        else:
            print("Unsupported file format. Use CSV or TXT.")
    except Exception as e:
        print(f"Error reading file: {e}")

def analyze_multiple_texts(texts):
    """Analyze a list of texts and return a DataFrame."""
    results = pd.DataFrame({'Text': texts, 'Sentiment': [analyze_sentiment(text) for text in texts]})
    print(results)
    plt.figure(figsize=(8, 5))
    sns.countplot(x=results['Sentiment'], palette='viridis')
    plt.title('Batch Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()
    return results

def main():
    while True:
        print("\nSentiment Analysis Tool")
        print("1. Analyze single text")
        print("2. Analyze text file (CSV or TXT)")
        print("3. Analyze multiple texts")
        print("4. Exit")
        choice = input("Choose an option: ")
        
        if choice == '1':
            text = input("Enter text for sentiment analysis: ")
            sentiment = analyze_sentiment(text)
            print(f'Sentiment: {sentiment}')
        elif choice == '2':
            file_path = input("Enter file path: ")
            analyze_text_file(file_path)
        elif choice == '3':
            texts = []
            print("Enter multiple texts (type 'done' to finish):")
            while True:
                text_input = input()
                if text_input.lower() == 'done':
                    break
                texts.append(text_input)
            analyze_multiple_texts(texts)
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()
