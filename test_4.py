from deepgram import Deepgram
import json
import glob
import pandas as pd

# The API key we created in step 3
DEEPGRAM_API_KEY = 'c3fabb0f38a92490dd7fd35ad3195eb828d40c32'
data = []

def process_audio_file(file_path, stri):
    # Replace with your file path and audio mimetype
    PATH_TO_FILE = file_path
    MIMETYPE = 'audio/mp3'

    # Initializes the Deepgram SDK
    dg_client = Deepgram(DEEPGRAM_API_KEY)
    
    with open(PATH_TO_FILE, 'rb') as audio:
        source = {'buffer': audio, 'mimetype': MIMETYPE}
        options = { "smart_format": True, "model": "enhanced", "language": "fr" }
        print(file_path)

        response = dg_client.transcription.sync_prerecorded(source, options)
        result = response['results']['channels'][0]['alternatives'][0]['transcript']
        result = result.split('\n')  # Split sentences by newline

        for sentence in result:
            data.append({'Word': stri, 'Transcription': sentence.strip()})  # Strip leading/trailing whitespace

def main():
    x = glob.glob("/home/dhruvi/Desktop/deepgram/French_words/*")

    for x1 in x:
        stri = x1.split("/")[-1]  # Extract the word from the file path
        process_audio_file(x1, stri)

    final = pd.DataFrame(data)
    final.to_csv('final_result_enhanced10.csv', index=False)

main()