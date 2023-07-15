# Core Pkgs
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

# NLP
import neattext.functions as nfx
import spacy
from spacy import displacy

# Text Downloader
import base64
import time
from urllib.request import urlopen
from bs4 import BeautifulSoup

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Set up Spacy
nlp = spacy.load("en_core_web_sm")

# Load Emotion Classifier Model
pipe_lr = joblib.load(open("model/emotion_classifier_pipe_lr_14_Aug_2022.pkl", "rb"))

# Global Variables
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "happy": "🤗",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮",
}

# Function for cleaning text
def clean_text(raw_text, normalize_case=False, clean_stopwords=False):
    if normalize_case:
        raw_text = raw_text.lower()

    if clean_stopwords:
        stop_words = nfx.available_stopwords()
        raw_text = nfx.remove_stopwords(raw_text, stopwords=stop_words)

    return raw_text


# Function for plotting word cloud
def plot_wordcloud(my_text):
    word_freq = Counter(my_text.split())
    wordcloud = WordCloud().generate_from_frequencies(word_freq)
    fig = plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)


# Function for fetching text from URL
@st.cache
def get_text(raw_url):
    page = urlopen(raw_url)
    soup = BeautifulSoup(page)
    fetched_text = " ".join([p.text for p in soup.find_all("p")])
    return fetched_text


# Function for Sumy Summarization
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = " ".join(summary_list)
    return result


# Function for analyzing text
def analyze_text(text):
    docx = nlp(text)
    allData = [
        (
            token.text,
            token.shape_,
            token.pos_,
            token.tag_,
            token.lemma_,
            token.is_alpha,
            token.is_stop,
        )
        for token in docx
    ]
    df = pd.DataFrame(
        allData,
        columns=["Token", "Shape", "PoS", "Tag", "Lemma", "IsAlpha", "Is_Stopword"],
    )
    return df


# Function for predicting emotions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


# Function for getting prediction probabilities
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


# Function for rendering entities with highlighting
def render_entities(rawtext):
    docx = nlp(rawtext)
    html = displacy.render(docx, style="ent")
    html = html.replace("\n\n", "\n")
    result = HTML_WRAPPER.format(html)
    return result


# Function for getting most common tokens
def get_most_common_tokens(docx, num=10):
    word_freq = Counter(docx.split())
    most_common_tokens = word_freq.most_common(num)
    return dict(most_common_tokens)


# Function for downloading text as a file
def text_downloader(raw_text):
    b64 = base64.b64encode(raw_text.encode()).decode()
    new_filename = "clean_text_result_{}_.txt".format(time.strftime("%Y%m%d-%H%M%S"))
    href = f'<a href="data:file/txt;base64,{b64}" download="{new_filename}">Download</a>'
    st.markdown("### 📥 Download Cleaned Text file ")
    st.markdown(href, unsafe_allow_html=True)


# Function for making dataframe downloadable as CSV
def make_downloadable(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "nlp_result_{}_.csv".format(time.strftime("%Y%m%d-%H%M%S"))
    st.markdown("### 📥 Download CSV file ")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Download</a>'
    st.markdown(href, unsafe_allow_html=True)


def main():
    image = Image.open("logo.png")
    st.image(image, use_column_width=True)
    st.title("Texter")

    menu = ["Text Cleaner", "Emotion Classifier", "Summarizer and Entity Checker", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Text Cleaner":
        st.title("Text Cleaner")
        menu = ["Text Cleaner", "About"]
        choice = st.sidebar.selectbox("Select", menu)

        if choice == "Text Cleaner":
            text_file = st.file_uploader("Upload Txt File", type=["txt"])
            normalize_case = st.sidebar.checkbox("Normalize Case")
            clean_stopwords = st.sidebar.checkbox("Stopwords")

            if text_file is not None:
                file_details = {
                    "Filename": text_file.name,
                    "Filesize": text_file.size,
                    "Filetype": text_file.type,
                }
                st.write(file_details)

                # Decode Text
                raw_text = text_file.read().decode("utf-8")

                col1, col2 = st.columns(2)

                with col1:
                    with st.expander("Original Text"):
                        st.write(raw_text)

                with col2:
                    with st.expander("Processed Text"):
                        raw_text = clean_text(raw_text, normalize_case, clean_stopwords)
                        st.write(raw_text)

                        text_downloader(raw_text)

                with st.expander("Text Analysis"):
                    token_result_df = analyze_text(raw_text)
                    st.dataframe(token_result_df)
                    make_downloadable(token_result_df)

                with st.expander("Plot Wordcloud"):
                    plot_wordcloud(raw_text)

                with st.expander("Plot PoS Tags"):
                    fig = plt.figure()
                    sns.countplot(token_result_df["PoS"])
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

        else:
            st.subheader("About")
            st.markdown("It's an app.")

    if choice == "Emotion Classifier":
        st.title("Emotion Classifier")
        menu = ["Home", "About"]
        choice = st.sidebar.selectbox("Select", menu)

        if choice == "Home":
            st.subheader("Emotions in Text")

            with st.form(key="emotion_clf_form"):
                raw_text = st.text_area("Type Here")
                submit_text = st.form_submit_button(label="Submit")

            if submit_text:
                col1, col2 = st.columns(2)

                # Apply Function Here
                prediction = predict_emotions(raw_text)
                probability = get_prediction_proba(raw_text)

                with col1:
                    st.success("Original Text")
                    st.write(raw_text)

                    st.success("Prediction")
                    emoji_icon = emotions_emoji_dict[prediction]
                    st.write("{}:{}".format(prediction, emoji_icon))
                    st.write("Confidence:{}".format(np.max(probability)))

                with col2:
                    st.success("Prediction Probability")
                    proba_df = pd.DataFrame(
                        probability, columns=pipe_lr.classes_
                    ).T.reset_index()
                    proba_df.columns = ["emotions", "probability"]

                    fig = alt.Chart(proba_df).mark_bar().encode(
                        x="emotions", y="probability", color="emotions"
                    )
                    st.altair_chart(fig, use_container_width=True)

        else:
            st.subheader("About")
            st.markdown("It's an app.")

    if choice == "Summarizer and Entity Checker":
        st.title("Summarizer and Entity Checker")
        menu = ["Summarize", "NER Checker", "NER For URLs", "About"]
        choice = st.sidebar.selectbox("Select", menu)

        if choice == "Summarize":
            st.subheader("Summarize Document")
            raw_text = st.text_area("Enter Text Here", "Type Here")
            summarizer_type = st.selectbox("Summarizer Type", ["Gensim", "Sumy Lex Rank"])
            if st.button("Summarize"):
                if summarizer_type == "Gensim":
                    summary_result = summarize(raw_text)
                elif summarizer_type == "Sumy Lex Rank":
                    summary_result = sumy_summarizer(raw_text)

                st.write(summary_result)

        if choice == "NER Checker":
            st.subheader("Named Entity Recognition with Spacy")
            raw_text = st.text_area("Enter Text Here", "Type Here")
            if st.button("Analyze"):
                docx = analyze_text(raw_text)
                html = displacy.render(docx, style="ent")
                html = html.replace("\n\n", "\n")
                st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

        if choice == "NER For URLs":
            st.subheader("Analysis on Text From URL")
            raw_url = st.text_input("Enter URL Here", "Type here")
            text_preview_length = st.slider("Length to Preview", 50, 100)
            if st.button("Analyze"):
                if raw_url != "Type here":
                    result = get_text(raw_url)
                    len_of_full_text = len(result)
                    len_of_short_text = round(len(result) / text_preview_length)
                    st.success("Length of Full Text: {}".format(len_of_full_text))
                    st.success("Length of Short Text: {}".format(len_of_short_text))
                    st.info(result[:len_of_short_text])
                    summarized_docx = sumy_summarizer(result)
                    docx = analyze_text(summarized_docx)
                    html = displacy.render(docx, style="ent")
                    html = html.replace("\n\n", "\n")
                    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
        else:
            st.subheader("About")
            st.markdown("It's an app.")


if __name__ == "__main__":
    main()
