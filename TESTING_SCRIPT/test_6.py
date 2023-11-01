import csv
import os
from deepgram import Deepgram
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEEPGRAM_API_KEY = 'c3fabb0f38a92490dd7fd35ad3195eb828d40c32'
AUDIO_FOLDER = '/home/dhruvi/Desktop/deepgram/testing'
MIMETYPE = 'audio/m4a'

def main():
    # Initialize the Deepgram SDK
    dg_client = Deepgram(DEEPGRAM_API_KEY)
    
    # Create a list to store results for each audio file
    audio_results = []

    # List audio files in the specified folder
    audio_files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith('.m4a')]

    for audio_file in audio_files:
        audio_path = os.path.join(AUDIO_FOLDER, audio_file)

        with open(audio_path, 'rb') as audio:
            source = {'buffer': audio, 'mimetype': MIMETYPE}
            options = {"smart_format": True, "model": "whisper-large", "language": "fr"}
        
            print(f'Requesting transcript for {audio_file}...')
            response = dg_client.transcription.sync_prerecorded(source, options)
            # Inside your for loop for audio files
            try:
              result = response['results']['channels'][0]['alternatives'][0]['transcript']
              result = result.split('.')
              transcribed_text = " ".join(result)
            except (KeyError, IndexError):
              print(f"Failed to get transcript for {audio_file}. Skipping...")
              continue

            
            csv_file = "/home/dhruvi/Desktop/deepgram/fr1 .csv"  # Path to your French words CSV
            word_data = []
            with open(csv_file, "r", encoding="utf-8") as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    if len(row) >= 2:
                        word_data.append({"headword": row[0], "pronunciation": row[1]})
            
            # Tokenize the result text
            result_tokens = transcribed_text.split()  # Assuming space-separated words
            
            # Initialize variables to store the best match and its similarity score
            best_match = None
            best_score = 0.0
            
            # Create TF-IDF vectors for the words in the CSV data and the result text
            word_texts = [data["headword"] for data in word_data]
            word_texts.append(transcribed_text)
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(word_texts)
            
            # Calculate similarity scores using cosine similarity
            result_vector = tfidf_matrix[-1]  # Vector for the result text
            word_vectors = tfidf_matrix[:-1]  # Vectors for words in the CSV data
            similarity_scores = cosine_similarity(result_vector, word_vectors)
            
            # Find the word with the highest similarity score
            best_index = similarity_scores.argmax()
            best_match = word_data[best_index]["headword"]
            best_score = similarity_scores[0][best_index]
            
            audio_results.append({
                "audio_file": audio_file,
                "transcribed_text": transcribed_text,
                "best_match": best_match,
                "similarity_score": best_score
            })

    # Save the results to a CSV file
    csv_output_file = "audio_results.csv_1"
    with open(csv_output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['audio_file', 'transcribed_text', 'best_match', 'similarity_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in audio_results:
            writer.writerow(result)

if __name__ == "__main__":
    main()
