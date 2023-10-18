
from deepgram import Deepgram
# import json

DEEPGRAM_API_KEY = 'c3fabb0f38a92490dd7fd35ad3195eb828d40c32'

# Replace with your file path and audio mimetype
PATH_TO_FILE = '/home/dhruvi/Desktop/deepgram/Ananas 3 syl.mp3'
MIMETYPE = 'audio/mp3'

def main():
    # Initializes the Deepgram SDK
    dg_client = Deepgram(DEEPGRAM_API_KEY)
    
    with open(PATH_TO_FILE, 'rb') as audio:
        source = {'buffer': audio, 'mimetype': MIMETYPE}
        options = { "smart_format": True, "model": "Whisper Cloud", "language": "fr" }
    
        print('Requesting transcript...')
        
    
        response = dg_client.transcription.sync_prerecorded(source, options)
        # print(json.dumps(response, indent=4))
      
        result = response['results']['channels'][0]['alternatives'][0]['transcript']
        result = result.split('.')
        for sentence in result:
            print(sentence + '.')

main()