# Signal Studio

Signal Studio is a small Streamlit app for exploring sound. You can upload a
recording, see what is happening in it, make changes, and download the result.

## Run it

```powershell
python -m pip install -r requirements.txt
python -m streamlit run dashboard.py
```

Use the same Python interpreter for both commands. To run the checks:

```powershell
python -m unittest discover -s tests
```

## What's in the app

**Heart sounds:** Upload a WAV recording to see its waveform and strongest
frequencies. You can filter the sound, listen to the result, and see where the
app found possible S1 and S2 sounds. Beat timing uses up to the first 20 seconds;
the optional animation uses the first five. The timing is an estimate from the
audio, not a medical diagnosis.

**Audio analyzer:** Upload a WAV, FLAC, OGG, or MP3 file, or record a clip with
your microphone. The tools let you see the waveform and spectrogram, filter
frequencies, reduce background noise, adjust bass and treble, try room sounds
and effects, find beats or pitch, quiet a selected part of the spectrogram, and
compare two recordings. Turn off the live display before using the recorder.
Clips must be between 0.1 seconds and five minutes long.

**Audio encryption:** Pick a password to turn an audio file into an encrypted
WAV. Keep that WAV and your password. Upload it in the Decrypt tab to get the
original file back exactly. Converting the encrypted WAV to a lossy format such
as MP3 will stop decryption from working.

**Steganography:** Add a short text message to a WAV file and download the new
recording. Upload that recording later to read the message. The text is hidden
in the audio's frequency data, but it is not password protected; anyone with
the extraction tool can read it.

**Echo detector:** Upload a WAV or MP3 file and choose the delay range to check.
The app looks for sound that repeats after a delay. Repeated music or beats can
look like an echo, so treat those results with care.
