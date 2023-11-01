import assemblyai as aai
aai.settings.api_key = f"7f588d42355d4b858e309bb54521b070" 
transcriber = aai.Transcriber()
transcript = transcriber.transcribe("/home/dhruvi/Desktop/deepgram/French_words/Cuisine - 2 syllables.mp3" )
print(transcript.text)

