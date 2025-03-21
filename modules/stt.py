from pywhispercpp.model import Model
import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io
import librosa


model = Model("base.en", n_threads=6)

def record_audio():

    state = st.session_state

    if "text_received" not in state:
        state.text_received = []
    
    audio = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=True,
        use_container_width=True,
        format="wav",
        callback=None,
        args=(),
        kwargs={},
        key=None
    )
    result = ""


    if audio:
        audio_bio = io.BytesIO(audio["bytes"])
        audio_bio.name = 'audio.wav'
        data, sample_rate = librosa.load(audio_bio, sr=audio["sample_rate"], mono=True)
        data = librosa.resample(data, orig_sr=sample_rate, target_sr=16000)
        segments = model.transcribe(data)
        for segment in segments:
            print(segment.text)
            result += segment.text
    return result if result else None



#legacy code
# def save_audio_file(audio, filename):
#     # Extract the audio data and parameters from the dictionary
#     audio_bytes = audio["bytes"]
#     sample_rate = audio["sample_rate"]
#     sample_width = audio["sample_width"]
    
#     # Open a new WAV file in write mode
#     with wave.open(filename, 'wb') as wav_file:
#         # Set the number of channels (1 for mono, 2 for stereo)
#         wav_file.setnchannels(1)  # Assuming mono audio
        
#         # Set the sample width (in bytes)
#         wav_file.setsampwidth(sample_width)
        
#         # Set the sample rate
#         wav_file.setframerate(sample_rate)
        
#         # Write the audio bytes to the file
#         wav_file.writeframes(audio_bytes)


# def audio_bytes_to_numpy(audio):
#     # Use io.BytesIO to treat the raw bytes as a file-like object
#     with io.BytesIO(audio) as wav_file:
#         # Open the WAV file using the wave module
#         with wave.open(wav_file, 'rb') as wav:
#             # Extract WAV file parameters
#             sample_width = wav.getsampwidth()  # Sample width in bytes
#             sample_rate = wav.getframerate()   # Sample rate in Hz
#             num_channels = wav.getnchannels()  # Number of channels (1 for mono, 2 for stereo)
#             num_frames = wav.getnframes()      # Total number of frames (samples per channel)
            
#             # Read all audio frames as bytes
#             raw_data = wav.readframes(num_frames)
            
#             # Determine the data type based on the sample width
#             if sample_width == 2:
#                 dtype = np.int16  # 16-bit audio
#             elif sample_width == 4:
#                 dtype = np.int32  # 32-bit audio
#             else:
#                 raise ValueError("Unsupported sample width. Only 2 (16-bit) or 4 (32-bit) are supported.")
            
#             # Convert the raw bytes to a NumPy array
#             audio_array = np.frombuffer(raw_data, dtype=dtype)
            
#             # Reshape the array if the audio is stereo (2 channels)
#             if num_channels > 1:
#                 audio_array = audio_array.reshape(-1, num_channels)
            
#             return audio_array, sample_rate
    