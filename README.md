# Signal Analyzer

The Streamlit dashboard has two pages: the existing heart sound analyzer and a
general audio analyzer. Run it with:

```powershell
pip install -r requirements.txt
streamlit run dashboard.py
```

Run the audio processing checks with `python -m unittest discover -s tests`.

On **Heart sounds**, upload a WAV file. All heart-sound plots and beat timing
use up to the first 20 seconds of the recording (the optional animated GIF uses
five seconds). An eight-second file provides eight seconds of analysis.
The app estimates S1 and S2 from locally consistent alternating cycles. It
reports an estimated BPM after two S1-to-S1 intervals and an interval
coefficient of variation (CV) after three. Ambiguous or insufficient detections
produce no timing estimate. The numbers describe acoustic timing; they do not
diagnose a regular rhythm or an arrhythmia.

Choose **Audio analyzer** in the sidebar. Upload a WAV, FLAC, OGG, or MP3 file,
or record a clip with the microphone control on that page. A successful clip
replaces the recorder with a playback preview; select **Analyze recording** to
open its analysis, or **Record another clip** to replace it. The live waveform,
frequency bars, and spectrogram use browser microphone access for display;
turn that display off before recording. Each analysis plot has a PNG download;
edited sounds have WAV downloads.

Audio is converted to mono for analysis. Files must be between 0.1 seconds and
5 minutes. The noise remover uses a spectral gate, the pitch tuner analyzes a
short window at the selected time, and beat detection uses spectral onset
peaks. Long spectrograms use a bounded visual preview; edits still process the
full recording. These results are estimates and depend on the recording quality.
