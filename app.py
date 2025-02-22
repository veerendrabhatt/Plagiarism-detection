from flask import Flask, render_template, request
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
import os

# Download required NLTK data
nltk.download('punkt')

app = Flask(__name__)

# Load the model and vectorizer with correct paths
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')

model = pickle.load(open(model_path, 'rb'))
tfidf_vectorizer = pickle.load(open(vectorizer_path, 'rb'))

def detect(input_text, reference_text):
    """Detect plagiarism using model prediction and calculate similarity scores."""
    if not input_text.strip() or not reference_text.strip():
        return "Invalid Input: Both fields are required.", 0.0, 0.0

    # Model prediction
    vectorized_text = tfidf_vectorizer.transform([input_text])
    result = model.predict(vectorized_text)
    plagiarism_result = "Plagiarism Detected" if result[0] == 1 else "No Plagiarism Detected"

    # Calculate BLEU score
    bleu_score = calculate_bleu(reference_text, input_text)

    # Calculate Cosine Similarity
    cosine_sim = cosine_similarity_score(reference_text, input_text)

    return plagiarism_result, bleu_score, cosine_sim

def calculate_bleu(reference_text, candidate_text):
    """Calculate BLEU score with unigram weight for short text."""
    reference_tokens = word_tokenize(reference_text)
    candidate_tokens = word_tokenize(candidate_text)
    return sentence_bleu([reference_tokens], candidate_tokens, weights=(1, 0, 0, 0))  # Unigram BLEU

def cosine_similarity_score(text1, text2):
    """Calculate cosine similarity between two texts."""
    vectors = tfidf_vectorizer.transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    input_text = request.form['text']
    reference_text = request.form['reference']
    detection_result, bleu_score, cosine_sim = detect(input_text, reference_text)
    return render_template('index.html', result=detection_result, bleu_score=bleu_score, cosine_sim=cosine_sim)

if __name__ == "__main__":
    app.run(debug=True)
