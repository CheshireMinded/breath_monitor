import sounddevice as sd
import soundfile as sf

def record_audio(filename='data/breathing.wav', duration=60, samplerate=44100):
    print("Recording... breathe normally")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    record_audio()
