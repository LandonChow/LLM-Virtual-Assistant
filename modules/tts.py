import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/en/ljspeech/vits").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
# wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
# Text to speech to a file

# my_text = "A chatbot is a computer program designed to simulate conversation with users, often through text or voice interactions. They use natural language processing to understand and respond to user queries in a way that feels conversational."


# tts.tts_to_file(text=my_text, file_path="output.wav")