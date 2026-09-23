"""An illustrative, rate-driven heart animation for the heart-sound page."""

from __future__ import annotations

import math


def heart_animation_html(bpm: float) -> str:
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("bpm must be positive and finite")
    period = 60 / bpm
    return f"""
    <style>
      * {{ box-sizing: border-box; }}
      body {{ margin: 0; background: transparent; font-family: sans-serif; }}
      .heart-card {{
        display: flex; align-items: center; justify-content: center; gap: 32px;
        min-height: 278px; padding: 12px 28px; overflow: hidden;
        border: 1px solid rgba(251, 113, 133, .22); border-radius: 18px;
        background: radial-gradient(circle at 37% 48%, rgba(218, 72, 98, .16), transparent 170px),
                    linear-gradient(120deg, #192738, #101e2d);
        color: #e8f3f4;
      }}
      .heart-stage {{ position: relative; flex: 0 0 238px; height: 248px; display: grid; place-items: center; }}
      .heart-halo {{
        position: absolute; width: 160px; height: 160px; top: 50px; left: 40px;
        border: 1px solid rgba(255, 117, 135, .4); border-radius: 50%;
        box-shadow: 0 0 36px rgba(235, 71, 101, .13);
        animation: heart-halo {period:.3f}s ease-out infinite;
      }}
      .heart-illustration {{ width: 214px; height: 238px; overflow: visible; }}
      .heart-organ {{
        transform-box: view-box; transform-origin: 50% 57%;
        animation: heart-contract {period:.3f}s cubic-bezier(.36, 0, .2, 1) infinite;
      }}
      .heart-copy {{ min-width: 0; max-width: 290px; }}
      .heart-eyebrow {{ color: #ff9dac; text-transform: uppercase; font-size: 11px;
                        font-weight: 800; letter-spacing: .16em; margin-bottom: 10px; }}
      .heart-title {{ font-size: clamp(20px, 3vw, 28px); line-height: 1.15;
                      font-weight: 750; margin-bottom: 12px; }}
      .heart-rate {{ display: flex; align-items: baseline; gap: 6px; color: #fff; }}
      .heart-rate strong {{ font-size: 38px; line-height: 1; font-variant-numeric: tabular-nums; }}
      .heart-rate span {{ font-size: 13px; color: #b8cbd3; }}
      .heart-note {{ color: #afc3ce; font-size: 12px; line-height: 1.5; margin: 14px 0 0; }}
      @keyframes heart-contract {{
        0%, 100% {{ transform: scale(1) rotate(-1deg); }}
        8% {{ transform: scale(.965, .94) rotate(0deg); }}
        19% {{ transform: scale(1.035, 1.025) rotate(-1deg); }}
        35% {{ transform: scale(1) rotate(-1deg); }}
        43% {{ transform: scale(.985, .975) rotate(-.4deg); }}
        53% {{ transform: scale(1.012) rotate(-1deg); }}
        65% {{ transform: scale(1) rotate(-1deg); }}
      }}
      @keyframes heart-halo {{
        0%, 100% {{ transform: scale(.78); opacity: 0; }}
        12% {{ opacity: .55; }}
        33% {{ transform: scale(1.25); opacity: 0; }}
      }}
      @media (max-width: 560px) {{
        .heart-card {{ gap: 4px; padding: 8px 14px; min-height: 250px; }}
        .heart-stage {{ flex-basis: 175px; height: 220px; }}
        .heart-illustration {{ width: 170px; height: 205px; }}
        .heart-halo {{ left: 17px; top: 43px; width: 140px; height: 140px; }}
        .heart-rate strong {{ font-size: 30px; }}
      }}
      @media (prefers-reduced-motion: reduce) {{
        .heart-organ, .heart-halo {{ animation: none; }}
      }}
    </style>
    <div class="heart-card" role="img" aria-label="Illustration of a beating heart at {bpm:.1f} beats per minute">
      <div class="heart-stage">
        <div class="heart-halo" aria-hidden="true"></div>
        <svg class="heart-illustration" viewBox="0 0 260 280" aria-hidden="true">
          <defs>
            <radialGradient id="muscle" cx="35%" cy="32%" r="76%">
              <stop offset="0" stop-color="#f17b7c"/>
              <stop offset=".45" stop-color="#c43d57"/>
              <stop offset="1" stop-color="#661d37"/>
            </radialGradient>
            <linearGradient id="ventricle" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stop-color="#f07c80" stop-opacity=".68"/>
              <stop offset="1" stop-color="#80243e" stop-opacity=".12"/>
            </linearGradient>
            <linearGradient id="aorta" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stop-color="#ff9991"/>
              <stop offset="1" stop-color="#9c3046"/>
            </linearGradient>
            <filter id="heart-shadow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="8"/>
            </filter>
          </defs>
          <ellipse cx="130" cy="241" rx="64" ry="15" fill="#020711" opacity=".45" filter="url(#heart-shadow)"/>
          <g class="heart-organ" stroke-linecap="round" stroke-linejoin="round">
            <path d="M139 91 C130 69 131 45 145 29 C162 10 191 20 198 43 C202 56 196 73 182 81"
                  fill="none" stroke="#652039" stroke-width="30"/>
            <path d="M139 91 C130 69 131 45 145 29 C162 10 191 20 198 43 C202 56 196 73 182 81"
                  fill="none" stroke="url(#aorta)" stroke-width="22"/>
            <path d="M169 27 L164 12 M187 28 L196 12" fill="none" stroke="#c44c5c" stroke-width="12"/>
            <path d="M111 94 C117 67 104 43 85 34 C66 25 51 33 44 49"
                  fill="none" stroke="#243b62" stroke-width="27"/>
            <path d="M111 94 C117 67 104 43 85 34 C66 25 51 33 44 49"
                  fill="none" stroke="#6b91b7" stroke-width="19"/>
            <path d="M70 75 C56 69 46 71 38 80 M67 106 C51 101 42 107 35 118"
                  fill="none" stroke="#7e3b5c" stroke-width="14"/>
            <path d="M115 95 C99 72 77 68 60 83 C44 98 45 124 59 151 C72 177 78 193 98 218
                     C112 237 124 257 135 254 C148 250 158 230 173 209 C192 182 210 157 210 133
                     C210 110 201 93 185 83 C168 72 146 76 132 91 C126 98 121 101 115 95 Z"
                  fill="url(#muscle)" stroke="#67213a" stroke-width="4"/>
            <path d="M126 101 C148 81 177 87 189 109 C199 127 193 154 177 179
                     C164 202 150 223 136 243 C146 212 148 180 141 151 C137 130 131 111 126 101 Z"
                  fill="url(#ventricle)"/>
            <path d="M116 99 C104 89 88 91 78 105 C68 119 69 137 79 154"
                  fill="none" stroke="#ffb3a2" stroke-opacity=".44" stroke-width="7"/>
            <path d="M120 106 C125 130 138 148 139 173 C140 195 133 222 134 242"
                  fill="none" stroke="#70213d" stroke-width="7" opacity=".72"/>
            <path d="M121 112 C126 136 137 153 138 175"
                  fill="none" stroke="#ff9a92" stroke-width="2.5" opacity=".64"/>
            <path d="M94 153 C110 160 122 169 133 181 M150 142 C164 134 181 134 196 139"
                  fill="none" stroke="#8a2944" stroke-width="5" opacity=".7"/>
            <path d="M84 175 C102 181 113 194 124 213 M149 190 C161 180 177 172 190 170"
                  fill="none" stroke="#f08083" stroke-width="2.5" opacity=".58"/>
          </g>
        </svg>
      </div>
      <div class="heart-copy">
        <div class="heart-eyebrow">Heart motion</div>
        <div class="heart-title">A more natural beat</div>
        <div class="heart-rate"><strong>{bpm:.1f}</strong><span>estimated BPM</span></div>
        <p class="heart-note">Illustration pulses at the estimated average rate. It does not track each sound in the recording.</p>
      </div>
    </div>
    """
