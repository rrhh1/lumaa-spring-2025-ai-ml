import numpy as np
import pandas as pd

import tiktoken

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Loading function
def load_data(path, column_names, encoding):
    df = pd.read_csv(path, encoding=encoding)
    all_info = df[column_names].values.tolist()
    
    # Adds aggregated column of important features
    for i, info in enumerate(all_info):
        all_info[i] = " ".join(info)
    df["All_Info"] = all_info

    return df


# Cosine similarity function, returns top n indexes and scores
def get_top_N_idx_scores(prompt_vector, info_vector, n):
    similarity = cosine_similarity(prompt_vector, info_vector).flatten()
    idx = np.argsort(similarity)[-n:][::-1]

    return idx, similarity[idx]


def main():
    # Basic Hyperparameters:
    ENCODING = "utf-8" # encoding specific to csv
    TIKTOKEN_ENCODER_MODEL = "r50k_base" # tokenizer model
    TOP_N = 5


    # Load and preprocess data set
    df = load_data(
        "./data/imdb_top250_movies.csv",
        ["Title", "Genre", "Director", "Actors", "Plot", "Language"],
        encoding=ENCODING
    )


    # TF-IDF vectorizer with subword tokenizer
    tokenizer = tiktoken.get_encoding(TIKTOKEN_ENCODER_MODEL).encode
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None, encoding=ENCODING)

    # Fit Vectorizer with movie details and get dataset information vectors
    info_vectors = vectorizer.fit_transform(df["All_Info"]).toarray()


    # Prompting loop
    while True:
        print("Type 'exit' to quit")
        prompt = input("Enter Prompt: ")

        if prompt == 'exit':
            break

        # Vectorize prompt and get similarity scores
        prompt_vector = vectorizer.transform([prompt]).toarray()
        idx, scores = get_top_N_idx_scores(prompt_vector, info_vectors, TOP_N)

        titles = df["Title"].to_numpy()[idx]

        print("Results")
        for i in range(len(titles)):
            print(f"Movie: {titles[i]} --- Similarity: {scores[i]}")
        print("===========================================")


if __name__ == "__main__":
    main()
