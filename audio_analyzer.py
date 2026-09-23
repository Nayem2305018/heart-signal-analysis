"""Streamlit interface for recording, inspecting, and editing general audio."""

from __future__ import annotations

import base64
import io
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import streamlit as st

import audio_processing as dsp
from ui_theme import render_empty_state, render_hero


def _plot(fig, name: str, key: str) -> None:
    st.pyplot(fig)
    image = io.BytesIO()
    fig.savefig(image, format="png", dpi=160, bbox_inches="tight")
    st.download_button("Save visualization as PNG", image.getvalue(),
                       file_name=f"{name}.png", mime="image/png", key=key)
    plt.close(fig)


def _waveform(axis, audio: np.ndarray, sample_rate: int, label: str,
              color: str = "#53ddcb", offset_seconds: float = 0.0) -> None:
    step = max(1, len(audio) // 12000)
    indices = np.arange(0, len(audio), step)
    axis.plot(indices / sample_rate + offset_seconds, audio[::step], linewidth=0.6,
              label=label, color=color)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude")
    axis.legend(loc="upper right")


def _spectrum(axis, audio: np.ndarray, sample_rate: int, label: str,
              color: str = "#53ddcb") -> None:
    frequency, db = dsp.spectrum_db(audio, sample_rate)
    axis.plot(frequency, db, linewidth=0.8, label=label, color=color)
    axis.set_xlim(0, min(10000, sample_rate / 2))
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Power (dB/Hz)")
    axis.legend(loc="upper right")


def _comparison(original: np.ndarray, edited: np.ndarray, sample_rate: int,
                name: str, key: str) -> None:
    peak = float(np.max(np.abs(edited)))
    if peak > 1:
        edited = edited / peak
        st.caption("Processed audio was scaled to avoid clipping in the WAV download.")
    left, right = st.columns(2)
    with left:
        st.caption("Original")
        st.audio(dsp.playback_wav_bytes(original, sample_rate), format="audio/wav")
    with right:
        st.caption("Processed")
        st.audio(dsp.playback_wav_bytes(edited, sample_rate), format="audio/wav")
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), constrained_layout=True)
    _waveform(axes[0], original, sample_rate, "Original")
    _waveform(axes[0], edited, sample_rate, "Processed", "#ff8398")
    _spectrum(axes[1], original, sample_rate, "Original")
    _spectrum(axes[1], edited, sample_rate, "Processed", "#ff8398")
    _plot(fig, name, f"{key}_image")
    st.download_button("Download processed WAV", dsp.wav_bytes(edited, sample_rate),
                       file_name=f"{name}.wav", mime="audio/wav", key=f"{key}_wav")


def _spectrogram(axis, audio: np.ndarray, sample_rate: int):
    if len(audio) / sample_rate > 30:
        frequency, times, magnitude = dsp.spectrogram_preview(audio, sample_rate)
    else:
        frequency, times, spectrum = dsp.stft(audio, sample_rate)
        magnitude = np.abs(spectrum)
    ceiling = min(10000, sample_rate / 2)
    selected = np.flatnonzero(frequency <= ceiling)
    frequency_step = max(1, int(np.ceil(len(selected) / 512)))
    time_step = max(1, int(np.ceil(len(times) / 900)))
    selected = selected[::frequency_step]
    shown_times = times[::time_step]
    db = 20 * np.log10(magnitude[selected, ::time_step] + 1e-7)
    image = axis.pcolormesh(shown_times, frequency[selected], db, shading="auto",
                            cmap="magma", vmin=max(-100, float(np.max(db)) - 80))
    axis.set_xlim(0, len(audio) / sample_rate)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Frequency (Hz)")
    return image


def _live_visualizer() -> None:
    # This display runs in the browser so Streamlit reruns do not interrupt it.
    html = """
    <style>
      html,body{margin:0;background:#08131f;font-family:Arial,sans-serif;color:#e8f3f4}
      .live-card{padding:16px 18px;background:#102233;border:1px solid #294b5b;border-radius:16px}
      .live-toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}
      button{border-radius:9px;padding:9px 14px;font-weight:700;cursor:pointer}
      #start{border:1px solid #6defdc;background:#53ddcb;color:#09222d}
      #stop{border:1px solid #466576;background:#1c3647;color:#e8f3f4}
      button:disabled{opacity:.5;cursor:default}
      #status{margin-left:auto;color:#a7c5ca;font-size:12px}
      .live-label{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#93b9c2;margin:7px 0 4px}
      canvas{display:block;width:100%;border:1px solid #244353;border-radius:8px;background:#101c2a}
    </style>
    <div class="live-card">
      <div class="live-toolbar"><button id="start">Start live display</button>
        <button id="stop" disabled>Stop</button><span id="status">Microphone idle</span></div>
      <div class="live-label">Waveform</div>
      <canvas id="wave" width="850" height="82"></canvas>
      <div class="live-label">Frequency bars</div>
      <canvas id="bars" width="850" height="82"></canvas>
      <div class="live-label">Scrolling spectrogram</div>
      <canvas id="spec" width="850" height="142"></canvas>
    </div>
    <script>
    let stream, context, analyzer, frame;
    const wave=document.getElementById('wave'), bars=document.getElementById('bars'), spec=document.getElementById('spec');
    const wc=wave.getContext('2d'), bc=bars.getContext('2d'), sc=spec.getContext('2d');
    const status=document.getElementById('status');
    function draw(){
      const time=new Uint8Array(analyzer.fftSize), freq=new Uint8Array(analyzer.frequencyBinCount);
      analyzer.getByteTimeDomainData(time); analyzer.getByteFrequencyData(freq);
      wc.fillStyle='#101c2a';wc.fillRect(0,0,wave.width,wave.height);
      wc.strokeStyle='#43d9b8';wc.beginPath();
      for(let x=0;x<wave.width;x++){let i=Math.floor(x*time.length/wave.width),y=time[i]/255*wave.height;
        if(x===0)wc.moveTo(x,y);else wc.lineTo(x,y)}wc.stroke();
      bc.fillStyle='#101c2a';bc.fillRect(0,0,bars.width,bars.height);
      const count=80, band=Math.max(1,Math.floor(freq.length/count));
      for(let i=0;i<count;i++){let v=0;for(let j=0;j<band;j++)v+=freq[i*band+j];v/=band;
        bc.fillStyle=`hsl(${180+i*2},85%,60%)`;
        bc.fillRect(i*bars.width/count,bars.height-v/255*bars.height,bars.width/count-2,v/255*bars.height)}
      sc.drawImage(spec,1,0,spec.width-1,spec.height,0,0,spec.width-1,spec.height);
      for(let y=0;y<spec.height;y++){let i=Math.floor((1-y/spec.height)*Math.min(freq.length-1,400));
        sc.fillStyle=`hsl(${265-freq[i]*0.8},95%,${8+freq[i]*0.27}%)`;sc.fillRect(spec.width-1,y,1,1)}
      frame=requestAnimationFrame(draw);
    }
    document.getElementById('start').onclick=async()=>{
      try {stream=await navigator.mediaDevices.getUserMedia({audio:true});
        context=new AudioContext();analyzer=context.createAnalyser();analyzer.fftSize=2048;
        context.createMediaStreamSource(stream).connect(analyzer);
        status.textContent='Live';document.getElementById('start').disabled=true;
        document.getElementById('stop').disabled=false;draw();
      } catch(e) {status.textContent='Microphone unavailable: '+e.message}
    };
    document.getElementById('stop').onclick=()=>{
      cancelAnimationFrame(frame);if(stream)stream.getTracks().forEach(t=>t.stop());
      if(context)context.close();status.textContent='Stopped';
      document.getElementById('start').disabled=false;document.getElementById('stop').disabled=true;
    };
    window.addEventListener('pagehide',()=>{if(stream)stream.getTracks().forEach(t=>t.stop());});
    </script>
    """
    st.components.v1.html(html, height=480)


def _beat_player(audio: np.ndarray, sample_rate: int, beats: np.ndarray) -> None:
    source = base64.b64encode(dsp.playback_wav_bytes(audio, sample_rate)).decode("ascii")
    times = json.dumps(beats.tolist())
    html = f"""
    <div style="box-sizing:border-box;text-align:center;background:#142239;padding:14px;border-radius:8px;color:white">
      <audio id="beatAudio" controls src="data:audio/wav;base64,{source}" style="display:block;width:100%;margin-bottom:8px"></audio>
      <div id="pulse" style="font-size:46px;line-height:1.15;transition:transform .1s">●</div>
      <div style="line-height:1.5;padding-top:4px">Beat pulse follows playback</div>
    </div>
    <script>
      const beats={times}, player=document.getElementById('beatAudio'), pulse=document.getElementById('pulse');
      function update(){{const t=player.currentTime;let near=false;
        if(!player.paused) for(const b of beats){{if(Math.abs(t-b)<.09){{near=true;break}}}}
        pulse.style.transform=near?'scale(1.5)':'scale(0.8)';
        pulse.style.color=near?'#ff745c':'#48d8c5';requestAnimationFrame(update)}}update();
    </script>"""
    st.components.v1.html(html, height=190)


def _store_recording() -> None:
    recorded = st.session_state.get("general_recording")
    if recorded is None:
        return
    data = recorded.getvalue()
    try:
        dsp.read_audio(data)
    except Exception as exc:
        st.session_state["general_recording_error"] = f"Could not read recording: {exc}"
        return
    st.session_state["general_recorded_bytes"] = data
    st.session_state["general_recorder_open"] = False
    st.session_state.pop("general_recording_error", None)


def _record_another_clip() -> None:
    st.session_state["general_recorder_open"] = True


def _open_recording_analysis() -> None:
    st.session_state["general_source"] = "Microphone recording"
    st.session_state["audio_tool"] = "Analyze"


def render_audio_analyzer() -> None:
    render_hero("audio")
    mode = st.sidebar.radio("Audio tool", ["Live", "Analyze", "Frequency filter", "Noise & EQ",
                             "Rooms & effects", "Beats & pitch",
                             "Spectrogram painter", "Compare"], key="audio_tool")
    live_enabled = False
    if mode == "Live":
        st.subheader("Live sound visualizer")
        live_enabled = st.toggle("Show live display", value=False)
        if live_enabled:
            _live_visualizer()
        st.caption("Turn off the live display before recording. Use the sidebar recorder to capture audio, then open Analyze or another tool.")

    with st.sidebar:
        st.header("Audio source")
        microphone_bytes = st.session_state.get("general_recorded_bytes")
        recorder_open = st.session_state.get("general_recorder_open", microphone_bytes is None)
        if recorder_open:
            st.audio_input("Record from microphone", key="general_recording",
                           on_change=_store_recording)
            if "general_recording_error" in st.session_state:
                st.warning(st.session_state["general_recording_error"])
        if microphone_bytes is not None:
            st.caption("Recording saved. Play it back or choose a tool below.")
            recording_audio, recording_rate = dsp.read_audio(microphone_bytes)
            st.audio(dsp.playback_wav_bytes(recording_audio, recording_rate), format="audio/wav")
            if not recorder_open:
                st.button("Record another clip", on_click=_record_another_clip,
                          key="record_another_clip")
            st.button("Analyze recording", on_click=_open_recording_analysis,
                      key="analyze_recording")
        uploaded = st.file_uploader("Upload WAV, FLAC, OGG, or MP3", type=["wav", "flac", "ogg", "mp3"],
                                    key="general_audio")
        st.caption("Recordings up to 5 minutes")
        source_choice = st.radio("Use", ["Uploaded file", "Microphone recording"],
                                 index=1 if microphone_bytes is not None else 0,
                                 key="general_source")
    if source_choice == "Uploaded file":
        source = uploaded.getvalue() if uploaded is not None else None
    else:
        source = microphone_bytes
    if source is None:
        render_empty_state("Choose an audio source",
                           "Upload a file or record a clip with the microphone control in the sidebar to begin.")
        return
    try:
        audio, sample_rate = dsp.read_audio(source)
    except Exception as exc:
        st.error(f"Could not read audio: {exc}")
        return
    duration = len(audio) / sample_rate

    if mode == "Analyze":
        st.subheader("Audio file analyzer")
        a, b, c = st.columns(3)
        a.metric("Duration", f"{duration:.2f} s")
        b.metric("Sample rate", f"{sample_rate:,} Hz")
        c.metric("Peak", f"{np.max(np.abs(audio)):.2f}")
        st.audio(dsp.playback_wav_bytes(audio, sample_rate), format="audio/wav")
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), constrained_layout=True)
        _waveform(axes[0], audio, sample_rate, "Audio")
        loud_time, loud_db = dsp.loudness(audio, sample_rate)
        axes[1].plot(loud_time, loud_db)
        axes[1].set(title="Loudness over time", xlabel="Time (s)", ylabel="RMS level (dBFS)")
        _spectrum(axes[2], audio, sample_rate, "Audio")
        image = _spectrogram(axes[3], audio, sample_rate)
        fig.colorbar(image, ax=axes[3], label="Magnitude (dB)")
        _plot(fig, "audio_analysis", "analysis_plot")
        st.download_button("Download original WAV", dsp.wav_bytes(audio, sample_rate),
                           file_name="original.wav", mime="audio/wav", key="original_wav")

    if mode == "Frequency filter":
        st.subheader("Frequency filter")
        st.write("Choose which frequencies to keep or remove, then compare the sound and spectrum.")
        highest_hz = int(min(20000, np.floor(sample_rate * 0.475)))
        if highest_hz < 21:
            st.warning("This recording's sample rate is too low for the frequency filter controls.")
        else:
            filter_kind = st.selectbox("Filter type", ["Low pass", "High pass",
                                                       "Band pass", "Band stop"])
            filtered = None
            if filter_kind in ("Low pass", "High pass"):
                default_hz = (min(3000, int(highest_hz * 0.7)) if filter_kind == "Low pass"
                              else min(200, int(highest_hz * 0.25)))
                cutoff_hz = st.slider("Cutoff frequency (Hz)", 10, highest_hz,
                                      max(10, default_hz), 1)
                filtered = dsp.frequency_filter(audio, sample_rate, filter_kind, cutoff_hz)
                st.caption(f"{filter_kind}: {cutoff_hz:,} Hz cutoff")
            else:
                lower = min(highest_hz - 1, max(10, min(80, highest_hz // 4)))
                upper = max(lower + 1, min(4000, int(highest_hz * 0.8)))
                lower_hz, upper_hz = st.slider("Frequency range (Hz)", 10, highest_hz,
                                               (lower, upper), 1)
                if lower_hz == upper_hz:
                    st.info("Move the range handles apart to apply the filter.")
                else:
                    filtered = dsp.frequency_filter(audio, sample_rate, filter_kind,
                                                     lower_hz, upper_hz)
                    st.caption(f"{filter_kind}: {lower_hz:,}–{upper_hz:,} Hz")
            if filtered is not None:
                _comparison(audio, filtered, sample_rate, "frequency_filtered", "frequency_filter")

    if mode == "Noise & EQ":
        st.subheader("Noise remover")
        strength = st.slider("Noise reduction", 0.0, 4.0, 1.5, 0.1)
        cleaned = dsp.denoise(audio, sample_rate, strength)
        _comparison(audio, cleaned, sample_rate, "noise_reduced", "denoise")
        st.subheader("Interactive equalizer")
        bass = st.slider("Bass (dB)", -18, 18, 0)
        mids = st.slider("Mids (dB)", -18, 18, 0)
        treble = st.slider("Treble (dB)", -18, 18, 0)
        eq = dsp.equalize(audio, sample_rate, bass, mids, treble)
        _comparison(audio, eq, sample_rate, "equalized", "eq")

    if mode == "Rooms & effects":
        st.subheader("Room simulator")
        room = st.selectbox("Room", ["Small room", "Hall", "Tunnel"])
        wet = st.slider("Room mix", 0.0, 1.0, 0.5, 0.05)
        room_audio, impulse = dsp.room_simulate(audio, sample_rate, room, wet)
        fig, ax = plt.subplots(figsize=(10, 2.5), constrained_layout=True)
        _waveform(ax, impulse, sample_rate, "Echo tail")
        _plot(fig, "room_echo_tail", "room_tail")
        _comparison(audio, room_audio, sample_rate, "room_simulated", "room")
        st.subheader("Audio effects playground")
        kind = st.selectbox("Effect", ["Echo", "Delay", "Tremolo", "Distortion"])
        amount = st.slider("Effect amount", 0.0, 0.95, 0.5, 0.05)
        delay = st.slider("Delay / tremolo speed", 0.05, 1.0, 0.3, 0.05)
        effected = dsp.effects(audio, sample_rate, kind, amount, delay)
        _comparison(audio, effected, sample_rate, kind.lower(), "effect")

    if mode == "Beats & pitch":
        st.subheader("Beat visualizer")
        beats = dsp.detect_beats(audio, sample_rate)
        st.write(f"Detected beats: {len(beats)}")
        fig, ax = plt.subplots(figsize=(11, 3), constrained_layout=True)
        _waveform(ax, audio, sample_rate, "Audio")
        for beat in beats:
            ax.axvline(beat, color="#ef684c", linewidth=0.8, alpha=0.7)
        _plot(fig, "beats", "beats_plot")
        _beat_player(audio, sample_rate, beats)
        st.subheader("Pitch detector and tuner")
        at = st.slider("Listen at (seconds)", 0.0, float(duration), min(duration / 2, 1.0),
                       0.01)
        result = dsp.detect_pitch(audio, sample_rate, at)
        if result is None:
            st.info("No stable pitch was found at this position. Try a sustained note.")
        else:
            frequency, note, cents = result
            st.metric("Detected note", note, f"{frequency:.1f} Hz")
            st.write(f"{abs(cents):.0f} cents {'sharp' if cents > 5 else 'flat' if cents < -5 else 'in tune'}")

    if mode == "Spectrogram painter":
        st.subheader("Spectrogram painter")
        st.write("Select a time and frequency region to attenuate, then listen to the result.")
        fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
        image = _spectrogram(ax, audio, sample_rate)
        fig.colorbar(image, ax=ax, label="Magnitude (dB)")
        start, end = st.slider("Time region (seconds)", 0.0, float(duration),
                               (0.0, float(min(duration, 1.0))), 0.01)
        ceiling = float(min(10000, sample_rate / 2))
        low, high = st.slider("Frequency region (Hz)", 0.0, ceiling,
                              (0.0, float(min(ceiling, 1000))), 1.0)
        reduction = st.slider("Reduction", 0.0, 1.0, 0.8, 0.05)
        ax.add_patch(Rectangle((start, low), end - start, high - low,
                               fill=False, edgecolor="cyan", linewidth=2))
        _plot(fig, "spectrogram_selection", "paint_plot")
        painted = dsp.paint_spectrogram(audio, sample_rate, start, end, low, high, reduction)
        _comparison(audio, painted, sample_rate, "spectrogram_edited", "paint")

    if mode == "Compare":
        st.subheader("Audio comparison")
        second_file = st.file_uploader("Second WAV, FLAC, OGG, or MP3 recording",
                                       type=["wav", "flac", "ogg", "mp3"], key="second_audio")
        if second_file is not None:
            try:
                second, second_rate = dsp.read_audio(second_file.getvalue())
            except Exception as exc:
                st.error(f"Could not read second recording: {exc}")
            else:
                if second_rate != sample_rate:
                    from scipy import signal
                    from math import gcd
                    divisor = gcd(second_rate, sample_rate)
                    second = signal.resample_poly(second, sample_rate // divisor,
                                                  second_rate // divisor)
                    second = second.astype(np.float32)
                    st.caption("Second recording resampled to match the first recording's sample rate.")
                st.caption(f"Full lengths: first {duration:.2f} s · "
                           f"second {len(second) / sample_rate:.2f} s")
                alignment_mode = st.selectbox("Time alignment", ["Start together", "Automatic",
                                                                   "Manual shift"])
                shift_seconds = 0.0
                if alignment_mode == "Automatic":
                    estimate = dsp.estimate_alignment(audio, second, sample_rate)
                    if estimate is None:
                        st.info("No reliable alignment found within 10 seconds. "
                                "Using the recording starts; try Manual shift.")
                    else:
                        shift_seconds, confidence = estimate
                        st.caption(f"Estimated shift: {shift_seconds:+.2f} s "
                                   f"(envelope correlation {confidence:.2f}). "
                                   "Check the overlaid waveforms for a good match.")
                elif alignment_mode == "Manual shift":
                    shift_limit = float(min(30, max(duration, len(second) / sample_rate)))
                    shift_seconds = st.slider("Shift second recording (seconds)",
                                              -shift_limit, shift_limit, 0.0, 0.01)
                    st.caption("Positive values delay the second recording; negative values move it earlier.")
                shift_samples = round(shift_seconds * sample_rate)
                first_part, second_part, overlap_start = dsp.comparison_overlap(
                    audio, second, shift_samples)
                if len(first_part) < max(2, int(0.1 * sample_rate)):
                    st.warning("The recordings have too little overlap at this shift. Move them closer together.")
                else:
                    shift_seconds = shift_samples / sample_rate
                    rms_difference, correlation, level_difference = dsp.comparison_stats(
                        first_part, second_part)
                    difference = first_part - second_part
                    metric_a, metric_b, metric_c, metric_d = st.columns(4)
                    metric_a.metric("Shared audio", f"{len(first_part) / sample_rate:.2f} s")
                    metric_b.metric("Second shift", f"{shift_seconds:+.2f} s")
                    metric_c.metric("Waveform correlation",
                                    f"{correlation:.3f}" if correlation is not None else "Unavailable")
                    metric_d.metric("Level difference",
                                    f"{level_difference:+.2f} dB"
                                    if level_difference is not None else "Unavailable")
                    st.caption(f"RMS difference over the shared audio: {rms_difference:.5f}. "
                               "Correlation compares waveform shape; level difference is second minus first.")

                    first_col, second_col = st.columns(2)
                    with first_col:
                        st.caption("First recording · shared section")
                        st.audio(dsp.playback_wav_bytes(first_part, sample_rate), format="audio/wav")
                    with second_col:
                        st.caption("Second recording · shared section")
                        st.audio(dsp.playback_wav_bytes(second_part, sample_rate), format="audio/wav")
                    difference_peak = float(np.max(np.abs(difference)))
                    difference_audio = difference / max(1.0, difference_peak)
                    st.caption("Difference track · sound remaining after subtracting the second recording")
                    st.audio(dsp.playback_wav_bytes(difference_audio, sample_rate), format="audio/wav")
                    if difference_peak > 1:
                        st.caption("Difference audio was scaled to avoid clipping.")
                    st.download_button("Download difference WAV", dsp.wav_bytes(difference_audio, sample_rate),
                                       file_name="audio_difference.wav", mime="audio/wav",
                                       key="comparison_difference_wav")

                    fig, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)
                    _waveform(axes[0], audio, sample_rate, "First")
                    _waveform(axes[0], second, sample_rate, "Second", "#ff8398", shift_seconds)
                    axes[0].axvspan(overlap_start / sample_rate,
                                    (overlap_start + len(first_part)) / sample_rate,
                                    color="#b9a1ff", alpha=0.08, label="Shared section")
                    axes[0].legend(loc="upper right")
                    axes[0].set_title("Aligned waveforms")
                    _waveform(axes[1], difference, sample_rate, "Difference", "#b9a1ff",
                              overlap_start / sample_rate)
                    axes[1].set_title("Difference during shared audio")
                    first_freq, first_db = dsp.spectrum_db(first_part, sample_rate)
                    second_freq, second_db = dsp.spectrum_db(second_part, sample_rate)
                    axes[2].plot(first_freq, first_db, label="First", color="#53ddcb")
                    axes[2].plot(second_freq, second_db, label="Second", color="#ff8398")
                    axes[2].fill_between(first_freq, first_db,
                                         np.interp(first_freq, second_freq, second_db),
                                         color="#b9a1ff", alpha=0.18, label="Frequency difference")
                    axes[2].set(xlabel="Frequency (Hz)", ylabel="Power (dB/Hz)",
                                title="Spectrum of shared audio")
                    axes[2].set_xlim(0, min(10000, sample_rate / 2))
                    axes[2].legend(loc="upper right")
                    _plot(fig, "audio_comparison", "comparison_plot")
