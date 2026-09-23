# Signal Analyzer

The Streamlit dashboard has two pages: the existing heart sound analyzer and a
general audio analyzer. Run it with:

```powershell
pip install -r requirements.txt
streamlit run dashboard.py
```

Run the audio processing checks with `python -m unittest discover -s tests`.

Choose **Audio analyzer** in the sidebar. Upload a WAV, FLAC, OGG, or MP3 file, or
record a clip on the Live tool and select **Microphone recording** as the source.
The live waveform, frequency bars, and spectrogram use browser microphone
access. The other tools run on the selected recording. Each analysis plot has a
PNG download; edited sounds have WAV downloads.

Audio is converted to mono for analysis. Files must be between 0.1 seconds and
5 minutes. The noise remover uses a spectral gate, the pitch tuner analyzes a
short window at the selected time, and beat detection uses spectral onset
peaks. Long spectrograms use a bounded visual preview; edits still process the
full recording. These results are estimates and depend on the recording quality.
