# assistant.py

import sounddevice as sd
import numpy as np
import wave
import pyttsx3
import openai
from faster_whisper import WhisperModel

# === Config ===
samplerate = 16000
duration = 5
filename = "temp.wav"
openai.api_key = "API"
engine = pyttsx3.init()

# === Detect Microphones ===
devices = sd.query_devices()
input_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]

if not input_devices:
    print("❌ No microphone input devices found.")
    exit()

print("\n🎙️ Available Microphones:")
for i in input_devices:
    print(f"[{i}] {devices[i]['name']}")

# Set preferred mic
sd.default.device = input_devices[0]  # You can change index if needed
print(f"\n✅ Using mic: {devices[input_devices[0]]['name']}")

# === Load Whisper model ===
model = WhisperModel("tiny.en", device="cpu")
print("✅ Assistant ready. Say 'stop listening' or 'goodbye' to exit.")

# === Start Loop ===
while True:
    try:
        print("\n🎤 Listening... Speak now!")
        audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait(timeout=duration + 2)

        audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio.tobytes())

        segments, info = model.transcribe(filename)
        text = " ".join(segment.text for segment in segments)

        if text.strip():
            print(f"\n🗣️ You said: {text.strip()}")
            engine.say(f"You said: {text.strip()}")
            engine.runAndWait()

            if "stop listening" in text.lower() or "goodbye" in text.lower():
                engine.say("Goodbye! Stopping now.")
                engine.runAndWait()
                print("👋 Assistant stopped.")
                break

            # Smart replies
            if "meal plan" in text.lower():
                prompt = "Make me a healthy vegetarian meal plan for 1 day."
            elif "weather" in text.lower():
                prompt = "What is the weather like today?"
            else:
                prompt = text.strip()

            try:
                print("🤔 Thinking with GPT...")
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful personal assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                gpt_reply = response['choices'][0]['message']['content']
                print(f"\n🤖 Assistant: {gpt_reply}")
                engine.say(gpt_reply)
                engine.runAndWait()

            except Exception as e:
                print(f"\n❌ GPT Error: {e}")
                engine.say("Sorry, I couldn't reach GPT right now.")
                engine.runAndWait()

        else:
            print("😕 Sorry, didn't catch that.")

    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        break
    except Exception as e:
        print(f"\n❌ Mic/audio error: {e}")
        break
