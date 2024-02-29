import os
import traceback
import streamlit as st
import requests
import speech_recognition as sr
from transformers import pipeline

# Set Streamlit app layout to wide
st.set_page_config(layout="wide")

# Designing the interface
st.title("🎧 Audio Pulse 🔍: Deciphering Emotions 🎵, Amplifying Insight📈")
st.write("[IBARHIM](https://www.linkedin.com/in/muhammad-ibrahim-qasmi-9876a1297/)")

# Define Streamlit app sidebar
st.sidebar.title("Audio Analysis")
st.sidebar.write("The Audio Analysis app is a powerful tool that allows you to analyze audio files and gain valuable insights from them. "
                 "It combines speech recognition and sentiment analysis techniques to transcribe the audio and determine the sentiment expressed within it.")

# Function to download a sample audio file
def download_sample_audio(url, filepath):
    response = requests.get(url)
    with open(filepath, 'wb') as f:
        f.write(response.content)

# Download a sample audio file from an online source
sample_audio_url = "https://example.com/sample_audio.wav"  # Replace this with the actual URL of your sample audio file
sample_audio_filepath = "sample_audio.wav"  # Local filepath to save the sample audio file

if not os.path.exists(sample_audio_filepath):
    st.sidebar.warning("Downloading sample audio file for testing...")
    download_sample_audio(sample_audio_url, sample_audio_filepath)
    st.sidebar.success("Sample audio file downloaded successfully!")

# Note about it being just beginner-friendly and work in progress for perfection
st.sidebar.title("Note:")
st.sidebar.markdown("Model is predicting sound of audio and audio should be in wav format ")
st.sidebar.markdown("Here's an example audio file: [sample_audio.wav](https://voca.ro/1955dXG84rdK)")

# Upload audio file
st.sidebar.header("Upload Audio")
audio_file = st.sidebar.file_uploader("Upload your audio file or use the default sample", type=["wav"])

def perform_sentiment_analysis(text):
    # Load the sentiment analysis model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_analysis = pipeline("sentiment-analysis", model=model_name)

    # Perform sentiment analysis on the text
    results = sentiment_analysis(text)

    # Extract the sentiment label and score
    sentiment_label = results[0]['label']
    sentiment_score = results[0]['score']

    return sentiment_label, sentiment_score

def transcribe_audio(audio_file):
    r = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)  # Read the entire audio file

    try:
        transcribed_text = r.recognize_google(audio)  # Perform speech recognition
    except sr.UnknownValueError:
        st.error("Error: Unable to transcribe the audio. Please try again with a different audio file.")
        return None
    except sr.RequestError as e:
        st.error(f"Error: Could not request results from Google Speech Recognition service; {e}")
        return None

    return transcribed_text

def main():
    # Perform analysis when audio file is uploaded
    if audio_file is not None:
        try:
            # Perform audio transcription
            transcribed_text = transcribe_audio(audio_file)
            
            if transcribed_text:
                # Perform sentiment analysis
                sentiment_label, sentiment_score = perform_sentiment_analysis(transcribed_text)

                # Display the results
                st.header("Transcribed Text")
                st.text_area("Transcribed Text", transcribed_text, height=200)

                st.header("Sentiment Analysis")

                # Display sentiment labels with icons and scores
                negative_icon = "👎"
                neutral_icon = "😐"
                positive_icon = "👍"

                if sentiment_label == "NEGATIVE":
                    st.write(f"{negative_icon} Negative (Score: {sentiment_score})", unsafe_allow_html=True)
                elif sentiment_label == "NEUTRAL":
                    st.write(f"{neutral_icon} Neutral (Score: {sentiment_score})", unsafe_allow_html=True)
                elif sentiment_label == "POSITIVE":
                    st.write(f"{positive_icon} Positive (Score: {sentiment_score})", unsafe_allow_html=True)
                else:
                    st.error("Error: Unknown sentiment label.")

                # Provide additional information about sentiment score interpretation
                st.info("The sentiment score represents the intensity of positive, negative, or neutral emotions or opinions 📊 "
                        "A higher score indicates a stronger sentiment, while a lower score indicates a weaker sentiment")
        except Exception as ex:
            st.error("Error occurred during audio transcription and sentiment analysis.")
            st.error(str(ex))
            traceback.print_exc()

if __name__ == "__main__":
    main()
