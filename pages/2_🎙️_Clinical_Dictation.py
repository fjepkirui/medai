# import streamlit as st
#
# ## Set page configuration
# st.set_page_config(page_title="Clinical Dictation", page_icon="🎙️", layout="centered")
#
# ## Title
# st.title("🎙️ Clinical Dictation")
#
# ## Create a sidebar for navigation
# st.sidebar.info(
#     "This is the Clinical Dictation page where you can record and transcribe clinical notes."
# )
#
# st.subheader("Record Your Clinical Note")
#
# ## Use native mic input
# audio_value = st.audio_input(
#     label="Record your voice message", label_visibility="collapsed"
# )
#
# ## If audio is recorded
# if audio_value:
#     st.success("✅ Audio recorded successfully.")
#     st.audio(audio_value)
#
#     # Transcribe button
#     transcribe_btn = st.button("🧠 Transcribe Audio")
#
#     st.subheader("Transcribed Clinical Note")
#     transcription_box = st.empty()
#
#     if transcribe_btn:
#         with st.spinner("Transcribing..."):
#             # Fake transcription (replace with real LLaMA/Whisper call)
#             FAKE_TRANSCRIPTION = (
#                 "This is a simulated transcription of your recorded audio."
#             )
#
#             transcription_box.text_area(
#                 label="Edit Your Clinical Note", value=FAKE_TRANSCRIPTION, height=200
#             )
# else:
#     st.info("🎙️Click the mic icon to start recording a voice message.")

import asyncio
import os
import time
import wave

import numpy as np
import pyaudio
import streamlit as st
import torch
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline

# Avoid RuntimeError from torch.classes
torch.classes.__path__ = []

## Set page configuration
st.set_page_config(page_title="Clinical Dictation", page_icon="🎙️", layout="centered")

## Title
st.title("🎙️ Clinical Dictation")

## Create a sidebar for navigation
st.sidebar.info(
    "This is the Clinical Dictation page where you can record and transcribe clinical notes."
)


# Load ASR model
@st.cache_resource
def load_asr_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        chunk_length_s=30,
        stride_length_s=5,
        return_timestamps=False,
        device=device,
    )


asr_pipe = load_asr_model()


# Helper functions
def transcribe_file(file_path):
    with st.spinner("Transcribing..."):
        start = time.time()
        result = asr_pipe(file_path)
        end = time.time()
        st.success(f"Transcription complete in {round(end - start, 2)}s")
        return result["text"]


def save_audio_file(audio_bytes, filename="temp_audio.wav"):
    os.makedirs("temp", exist_ok=True)
    path = os.path.join("temp", filename)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path


async def real_time_transcription(run_flag, rate, buffer_size):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=rate,
        input=True,
        frames_per_buffer=buffer_size,
    )

    frames = []
    while run_flag():
        data = stream.read(buffer_size, exception_on_overflow=False)
        frames.append(data)

        # Every ~10 seconds, transcribe
        if len(frames) * buffer_size / rate > 10:
            audio_data = b"".join(frames)
            frames.clear()

            temp_path = "temp/temp_rt.wav"
            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(rate)
                wf.writeframes(audio_data)

            with wave.open(temp_path, "rb") as wf:
                raw_audio = wf.readframes(wf.getnframes())
                audio_array = (
                    np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

            try:
                result = asr_pipe(audio_array)
                text = result["text"]
                st.session_state["real_time_text"] += text + " "
                st.write(text)
            except Exception as e:
                st.error(f"Real-time error: {e}")

    stream.stop_stream()
    stream.close()
    p.terminate()


# Initialize session state
st.session_state.setdefault("real_time_text", "")
st.session_state.setdefault("run_rt", False)

# Tabs layout
tab1, tab2, tab3 = st.tabs(
    ["📂 Upload Audio", "🎙️ Record Audio", "⏱️ Real-Time Dictation"]
)

# Upload Tab
with tab1:
    st.header("📂 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file", type=["wav", "mp3", "m4a", "ogg"]
    )

    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Transcribe Uploaded Audio"):
            path = save_audio_file(uploaded_file.getbuffer(), uploaded_file.name)
            transcription = transcribe_file(path)
            st.session_state["upload_transcription"] = transcription

    if "upload_transcription" in st.session_state:
        st.text_area(
            "Transcription", st.session_state["upload_transcription"], height=300
        )

# Record Tab
with tab2:
    st.header("🎙️ Record Audio")
    audio_bytes = audio_recorder(
        text="Click the mic icon to start or stop the recording", icon_name="microphone"
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        if st.button("Transcribe Recorded Audio"):
            path = save_audio_file(audio_bytes, "recorded_audio.wav")
            transcription = transcribe_file(path)
            st.session_state["record_transcription"] = transcription

    if "record_transcription" in st.session_state:
        st.text_area(
            "Transcription", st.session_state["record_transcription"], height=300
        )

# Real-time Tab
with tab3:
    st.header("⏱️ Real-Time Dictation")

    st.sidebar.subheader("Real-Time Settings")
    rate = int(st.sidebar.text_input("Sample Rate", value="16000"))
    buffer_size = int(st.sidebar.text_input("Buffer Size", value="3200"))

    def toggle_rt():
        st.session_state["run_rt"] = not st.session_state["run_rt"]

    if not st.session_state["run_rt"]:
        st.button("Start Real-Time Dictation", on_click=toggle_rt)
    else:
        st.button("Stop Dictation", on_click=toggle_rt)

    if st.session_state["run_rt"]:
        asyncio.run(
            real_time_transcription(
                lambda: st.session_state["run_rt"], rate, buffer_size
            )
        )

    st.text_area("Live Transcription", st.session_state["real_time_text"], height=300)
