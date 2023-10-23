import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
import whisper

app = Flask(__name__)

# Load the Whisper ASR model
model = whisper.load_model("medium")

# Load the CSV file containing English words and phonetics
csv_file = "cmudict.csv"
word_data = []

with open(csv_file, "r", encoding="utf-8") as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        word_data.append({"word": row[1], "pronunciation": row[2]})

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    try:
        # Check if the 'audio' file is in the POST request
        if 'audio' not in request.files:
            return jsonify({"error": "Audio file not found"})

        audio_file = request.files['audio']

        # Save the audio data received from Unity to a temporary file
        audio_path = "received_audio.mp3"
        audio_file.save(audio_path)

        # Load and process the audio data
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # Detect the spoken language
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)

        # Decode the audio
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)

        # Get the transcribed text
        transcribed_text = result.text

        # Initialize variables to store the best match and its similarity score
        best_match = None
        best_score = 0.0

        # Tokenize the transcribed text
        transcribed_tokens = transcribed_text.split()

        # Create TF-IDF vectors for the words in the CSV data and the transcribed text
        word_texts = [data["word"] for data in word_data]
        word_texts.append(transcribed_text)
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(word_texts)

        # Calculate similarity scores using cosine similarity
        transcribed_vector = tfidf_matrix[-1]  # Vector for the transcribed text
        word_vectors = tfidf_matrix[:-1]  # Vectors for words in the CSV data
        similarity_scores = cosine_similarity(transcribed_vector, word_vectors)

        # Find the word with the highest similarity score
        best_index = similarity_scores.argmax()
        best_match = word_data[best_index]["word"]
        best_score = similarity_scores[0][best_index]

        # Return the detected language, transcribed text, best match, and similarity score
        return jsonify({"detected_language": detected_language, "transcribed_text": transcribed_text,
                        "best_match": best_match, "similarity_score": best_score})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
