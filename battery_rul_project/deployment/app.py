"""
AI Battery Health Prediction Dashboard — v4 Light
"""

import streamlit as st
import requests
import math

st.set_page_config(
    page_title="AI Battery Health Prediction",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --bg:            #f5f7fa;
    --white:         #ffffff;
    --subtle:        #f0f2f5;
    --border:        #dde3ec;
    --border-strong: #c4cdd8;
    --accent:        #0d9e72;
    --accent-lt:     #e6f7f2;
    --accent-mid:    #b3e8d8;
    --green:         #0d9e72;
    --amber:         #d97706;
    --amber-lt:      #fef3c7;
    --red:           #dc2626;
    --red-lt:        #fee2e2;
    --txt:           #111827;
    --txt2:          #4a5568;
    --txt3:          #8a96a3;
    --code:          #0d7a57;
    --serif:         'DM Serif Display', Georgia, serif;
    --mono:          'DM Mono', monospace;
    --body:          'Outfit', sans-serif;
}

/* Base */
.stApp { background: var(--bg) !important; }
.main .block-container { padding: 2rem 2rem 3rem !important; max-width: 1320px !important; }
div[data-testid="stForm"] { background: transparent !important; border: none !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
body, p, div, span { font-family: var(--body) !important; color: var(--txt) !important; }

/* Page title */
.pg-title {
    font-family: var(--serif) !important;
    font-size: 2rem;
    color: var(--txt) !important;
    margin: 0 0 0.1rem;
    line-height: 1.15;
}
.pg-title em { color: var(--accent) !important; font-style: italic; }
.pg-sub {
    font-family: var(--mono) !important;
    font-size: 0.66rem;
    color: var(--txt3) !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1.8rem;
}
.live-dot {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--accent-lt);
    border: 1px solid var(--accent-mid);
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    padding: 3px 10px;
    border-radius: 20px;
    text-transform: uppercase;
    vertical-align: middle;
    margin-left: 0.8rem;
}
.live-dot::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    display: inline-block;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1}50%{opacity:0.3} }

/* Cards */
.card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.6rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.card-heading {
    font-family: var(--mono) !important;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--txt3) !important;
    margin-bottom: 1.2rem;
    padding-left: 0.7rem;
    border-left: 3px solid var(--accent);
}

/* Inputs */
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    color: var(--txt) !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
}
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus {
    background: var(--white) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(13,158,114,0.12) !important;
    outline: none !important;
}
label[data-testid="stWidgetLabel"] p {
    font-family: var(--mono) !important;
    font-size: 0.67rem !important;
    letter-spacing: 0.08em !important;
    color: var(--txt2) !important;
    text-transform: uppercase !important;
}

/* Button */
.stFormSubmitButton button {
    width: 100% !important;
    background: var(--accent) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-family: var(--body) !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0.75rem !important;
    box-shadow: 0 2px 8px rgba(13,158,114,0.28) !important;
    transition: background 0.2s, box-shadow 0.2s !important;
}
.stFormSubmitButton button:hover {
    background: #0b8a63 !important;
    box-shadow: 0 4px 14px rgba(13,158,114,0.38) !important;
}

/* Metric card */
.mc {
    background: var(--white);
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.2rem 1.2rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    position: relative;
    overflow: hidden;
    margin-bottom: 0.75rem;
}
.mc::before {
    content:''; position:absolute; top:0;left:0;right:0; height:3px; border-radius:12px 12px 0 0;
}
.mc.h::before  { background: var(--green); }
.mc.d::before  { background: var(--amber); }
.mc.c::before  { background: var(--red); }
.mc.r::before  { background: #6366f1; }
.mc.h { border-color: #b3e8d8; }
.mc.d { border-color: #fcd34d; }
.mc.c { border-color: #fca5a5; }
.mc.r { border-color: #c7d2fe; }
.mc-lbl {
    font-family: var(--mono) !important;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--txt3) !important;
    margin-bottom: 0.5rem;
}
.mc-val {
    font-family: var(--serif) !important;
    font-size: 2.8rem;
    line-height: 1;
    margin-bottom: 0.2rem;
    font-weight: 400;
}
.mc-val.h { color: var(--green) !important; }
.mc-val.d { color: var(--amber) !important; }
.mc-val.c { color: var(--red) !important; }
.mc-val.r { color: #4f46e5 !important; }
.mc-unit {
    font-family: var(--mono) !important;
    font-size: 0.68rem;
    color: var(--txt3) !important;
}
.chip {
    display: inline-block;
    margin-top: 0.6rem;
    padding: 2px 12px;
    border-radius: 20px;
    font-family: var(--mono) !important;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.chip.h { background: var(--accent-lt); color: var(--green) !important; }
.chip.d { background: var(--amber-lt);  color: var(--amber) !important; }
.chip.c { background: var(--red-lt);    color: var(--red) !important; }

/* Progress */
.bar-wrap {
    background: var(--subtle);
    border: 1px solid var(--border);
    border-radius: 6px;
    height: 8px;
    overflow: visible;
    position: relative;
    margin: 0.8rem 0 1rem;
}
.bar-fill { height: 100%; border-radius: 6px; }
.bar-eol {
    position:absolute; top:-4px; bottom:-4px; left:70%;
    width:2px; background:var(--red); opacity:0.45; border-radius:1px;
}
.bar-eol-lbl {
    position:absolute; top:-18px; left:4px;
    font-family: var(--mono) !important;
    font-size:0.57rem; color:var(--red) !important;
    letter-spacing:0.05em; white-space:nowrap;
}

/* Alerts */
.alert {
    padding: 0.9rem 1.1rem;
    border-radius: 9px;
    font-family: var(--body) !important;
    font-size: 0.82rem;
    line-height: 1.5;
    margin: 1rem 0;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    border: 1px solid;
}
.alert.ok  { background: var(--accent-lt); border-color: var(--accent-mid); color: #065f46 !important; }
.alert.warn{ background: var(--amber-lt);  border-color: #fcd34d;           color: #92400e !important; }
.alert.err { background: var(--red-lt);    border-color: #fca5a5;           color: #991b1b !important; }

/* Legend */
.leg-row {
    display:flex; align-items:flex-start; gap:9px;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--border);
}
.leg-row:last-child { border-bottom:none; }
.leg-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; margin-top:3px; }
.leg-txt { font-family: var(--body) !important; font-size:0.78rem; color:var(--txt2) !important; line-height:1.4; }

/* Feature grid */
.fg { display:grid; grid-template-columns:1fr 1fr; gap:0.45rem; margin-bottom:1rem; }
.fi {
    background:var(--subtle); border:1px solid var(--border); border-radius:7px;
    padding:0.45rem 0.75rem; display:flex; justify-content:space-between; align-items:center;
}
.fk { font-family:var(--mono) !important; font-size:0.6rem; color:var(--txt3) !important; text-transform:uppercase; letter-spacing:0.06em; }
.fv { font-family:var(--mono) !important; font-size:0.75rem; color:var(--code) !important; font-weight:500; }

/* Expander */
details { background:var(--white) !important; border:1px solid var(--border) !important; border-radius:8px !important; }
summary { font-family:var(--mono) !important; font-size:0.7rem !important; color:var(--txt2) !important; letter-spacing:0.06em !important; padding:0.7rem 1rem !important; }

/* Spinner */
.stSpinner > div > div { border-top-color: var(--accent) !important; }

/* Scrollbar */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:var(--border-strong); border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ── SVG Gauge ────────────────────────────────────────────────────────────────
def gauge_svg(soh_pct, status):
    c = {"Healthy": "#0d9e72", "Degraded": "#d97706", "Critical": "#dc2626"}.get(status, "#0d9e72")
    W, H, cx, cy = 220, 138, 110, 124
    ro, ri = 90, 72
    sa, sw = 205, 260
    pct = max(0, min(100, soh_pct)) / 100

    def pt(a, r):
        rad = math.radians(a)
        return cx + r * math.cos(rad), cy + r * math.sin(rad)

    def arc(ri, ro, a1, a2, fill, op=1):
        x1o,y1o = pt(a1,ro); x2o,y2o = pt(a2,ro)
        x1i,y1i = pt(a1,ri); x2i,y2i = pt(a2,ri)
        lg = 1 if abs(a2-a1)>180 else 0
        return (f'<path d="M{x1o:.1f},{y1o:.1f} A{ro},{ro} 0 {lg},1 {x2o:.1f},{y2o:.1f} '
                f'L{x2i:.1f},{y2i:.1f} A{ri},{ri} 0 {lg},0 {x1i:.1f},{y1i:.1f}Z" '
                f'fill="{fill}" opacity="{op}"/>')

    track = arc(ri, ro, sa, sa+sw, "#e8ecf0")
    fill  = arc(ri, ro, sa, sa + pct*sw, c) if pct > 0 else ""

    ticks = ""
    for val, lbl in [(0,"0%"),(50,"50%"),(100,"100%")]:
        ang = sa + (val/100)*sw
        x1,y1 = pt(ang, ro+4); x2,y2 = pt(ang, ro+11); xl,yl = pt(ang, ro+21)
        ticks += (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                  f'stroke="#c4cdd8" stroke-width="1.5"/>'
                  f'<text x="{xl:.1f}" y="{yl:.1f}" fill="#8a96a3" font-size="9" '
                  f'text-anchor="middle" dominant-baseline="middle" '
                  f'font-family="DM Mono,monospace">{lbl}</text>')

    ex1,ey1 = pt(sa+(70/100)*sw, ri-3); ex2,ey2 = pt(sa+(70/100)*sw, ro+3)
    eol = (f'<line x1="{ex1:.1f}" y1="{ey1:.1f}" x2="{ex2:.1f}" y2="{ey2:.1f}" '
           f'stroke="#dc2626" stroke-width="1.5" stroke-dasharray="3,2" opacity="0.55"/>')

    dot = ""
    if pct > 0.01:
        dx,dy = pt(sa + pct*sw, (ri+ro)/2)
        dot = f'<circle cx="{dx:.1f}" cy="{dy:.1f}" r="5" fill="{c}" stroke="white" stroke-width="2"/>'

    return (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" width="230" height="145">'
            f'{track}{fill}{eol}{ticks}{dot}'
            f'<text x="{cx}" y="{cy-14}" fill="{c}" font-size="28" font-weight="400" '
            f'font-family="DM Serif Display,Georgia,serif" text-anchor="middle" dominant-baseline="middle">'
            f'{soh_pct:.1f}</text>'
            f'<text x="{cx}" y="{cy+10}" fill="#8a96a3" font-size="8.5" font-family="DM Mono,monospace" '
            f'text-anchor="middle" letter-spacing="2">STATE OF HEALTH %</text>'
            f'</svg>')


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.8rem; padding-bottom:1.4rem; border-bottom:1px solid #dde3ec;">
  <div style="display:flex; align-items:center; flex-wrap:wrap; gap:0.5rem;">
    <span class="pg-title">AI Battery Health <em>Prediction</em></span>
    <span class="live-dot">Live Inference</span>
  </div>
  <div class="pg-sub">LSTM v4 · 27-Feature Model · Real-Time Degradation Analysis</div>
</div>
""", unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.6], gap="large")

# ── LEFT: Form ───────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="card"><div class="card-heading">Cycle Sensor Input</div>', unsafe_allow_html=True)
    with st.form("pf"):
        c1, c2 = st.columns(2)
        with c1:
            avg_voltage  = st.number_input("Avg Voltage (V)",  value=3.7,    format="%.4f")
            avg_current  = st.number_input("Avg Current (A)",  value=1.5,    format="%.4f")
            voltage_drop = st.number_input("Voltage Drop (V)", value=0.5,    format="%.4f")
            cycle        = st.number_input("Cycle Number",     value=100,    step=1, format="%d")
        with c2:
            min_voltage  = st.number_input("Min Voltage (V)",  value=3.2,    format="%.4f")
            duration     = st.number_input("Duration (s)",     value=3600.0, format="%.1f")
            temp_change  = st.number_input("Temp Change (°C)", value=10.0,   format="%.2f")

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        # ✅ FIX: Updated default API URL to point to the server's public IP
        api_url = st.text_input("API Endpoint", value="http://43.205.208.227:5000/predict")
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Prediction →", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:1rem;">
      <div class="card-heading">SoH Health Thresholds</div>
      <div class="leg-row">
        <div class="leg-dot" style="background:#0d9e72;"></div>
        <div class="leg-txt"><b style="color:#0d9e72;">Above 80%</b> — Healthy. Optimal performance, no action needed.</div>
      </div>
      <div class="leg-row">
        <div class="leg-dot" style="background:#d97706;"></div>
        <div class="leg-txt"><b style="color:#d97706;">70–80%</b> — Degraded. Plan replacement within predicted RUL window.</div>
      </div>
      <div class="leg-row">
        <div class="leg-dot" style="background:#dc2626;"></div>
        <div class="leg-txt"><b style="color:#dc2626;">Below 70%</b> — Critical. End-of-life threshold breached. Act immediately.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── RIGHT: Results ────────────────────────────────────────────────────────────
with right:
    if not submitted:
        st.markdown("""
        <div class="card" style="display:flex;flex-direction:column;align-items:center;
             justify-content:center;height:400px;text-align:center;gap:1rem;
             background:linear-gradient(135deg,#f7f8fa,#eef6f3);
             border:1.5px dashed #b3e8d8;">
          <div style="font-size:2.5rem;opacity:0.2;">🔋</div>
          <div style="font-family:'DM Serif Display',serif;font-size:1.05rem;
                      color:#4a5568;font-style:italic;">Awaiting sensor data</div>
          <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#8a96a3;
                      max-width:260px;line-height:1.7;">
            Enter cycle measurements and click Run Prediction to begin analysis.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        payload = {
            "avg_voltage":  avg_voltage,
            "min_voltage":  min_voltage,
            "avg_current":  avg_current,
            "duration":     duration,
            "voltage_drop": voltage_drop,
            "temp_change":  temp_change,
            "cycle":        int(cycle),
        }

        with st.spinner("Running LSTM inference..."):
            try:
                resp = requests.post(api_url, json=payload, timeout=15)

                if resp.status_code == 200:
                    data   = resp.json()
                    soh    = float(data["SOH"])
                    rul    = float(data["RUL"])
                    status = data.get("status", "Healthy")
                    s      = status[0].lower()   # h / d / c

                    # Gauge + cards
                    gc, mc = st.columns([1, 1])
                    with gc:
                        st.markdown(
                            f'<div style="display:flex;justify-content:center;padding:0.8rem 0;">'
                            f'{gauge_svg(soh, status)}</div>',
                            unsafe_allow_html=True
                        )
                    with mc:
                        st.markdown(f"""
                        <div class="mc {s}">
                          <div class="mc-lbl">State of Health</div>
                          <div class="mc-val {s}">{soh:.1f}</div>
                          <div class="mc-unit">percent</div>
                          <div class="chip {s}">{status}</div>
                        </div>
                        <div class="mc r">
                          <div class="mc-lbl">Remaining Useful Life</div>
                          <div class="mc-val r">{int(rul):,}</div>
                          <div class="mc-unit">cycles</div>
                          <div class="chip" style="background:#e0e7ff;color:#4338ca;">Estimated</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Progress bar
                    bar = {
                        "Healthy":  "linear-gradient(90deg,#34d399,#0d9e72)",
                        "Degraded": "linear-gradient(90deg,#fbbf24,#d97706)",
                        "Critical": "linear-gradient(90deg,#f87171,#dc2626)",
                    }.get(status, "linear-gradient(90deg,#34d399,#0d9e72)")
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;margin-top:0.8rem;">
                      <span style="font-family:'DM Mono',monospace;font-size:0.63rem;
                                   color:#8a96a3;letter-spacing:0.1em;text-transform:uppercase;">
                        SoH Level</span>
                      <span style="font-family:'DM Mono',monospace;font-size:0.63rem;color:#8a96a3;">
                        {soh:.2f}% / 100%</span>
                    </div>
                    <div class="bar-wrap">
                      <div class="bar-fill" style="width:{soh}%;background:{bar};"></div>
                      <div class="bar-eol"><span class="bar-eol-lbl">EOL 70%</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Alert
                    if status == "Critical":
                        st.markdown(f"""
                        <div class="alert err">
                          <span>🚨</span>
                          <div><b>Critical — End-of-Life Threshold Breached</b><br>
                          SoH has dropped below 70%. Immediate replacement or maintenance required
                          to prevent unexpected failure.</div>
                        </div>""", unsafe_allow_html=True)
                    elif status == "Degraded":
                        st.markdown(f"""
                        <div class="alert warn">
                          <span>⚠️</span>
                          <div><b>Warning — Accelerated Degradation Detected</b><br>
                          Schedule replacement within the remaining
                          <b>{int(rul):,} cycles</b> to avoid critical failure.</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert ok">
                          <span>✅</span>
                          <div><b>Nominal — Operating Within Healthy Parameters</b><br>
                          Battery is healthy with an estimated
                          <b>{int(rul):,} cycles</b> of useful life remaining.</div>
                        </div>""", unsafe_allow_html=True)

                    # Details
                    with st.expander("View Input Details & Raw API Response"):
                        st.markdown('<div class="fg">', unsafe_allow_html=True)
                        for k, v in payload.items():
                            st.markdown(
                                f'<div class="fi"><span class="fk">{k}</span>'
                                f'<span class="fv">{v}</span></div>',
                                unsafe_allow_html=True
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.json(data)

                else:
                    st.markdown(f"""
                    <div class="alert err"><span>🔴</span>
                    <div><b>API Error {resp.status_code}</b><br>{resp.text}</div></div>
                    """, unsafe_allow_html=True)

            except requests.exceptions.ConnectionError:
                st.markdown("""
                <div class="alert err"><span>🔌</span>
                <div><b>Connection Refused</b><br>
                Cannot reach the API. Make sure <code>api.py</code> is running:<br>
                <code>python api.py</code> — listening on port 5000.</div></div>
                """, unsafe_allow_html=True)

            except requests.exceptions.Timeout:
                st.markdown("""
                <div class="alert warn"><span>⏱</span>
                <div><b>Request Timed Out</b><br>
                API did not respond within 15 seconds.</div></div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div class="alert err"><span>❌</span>
                <div><b>Unexpected Error</b><br>{str(e)}</div></div>
                """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2.5rem;padding-top:1.2rem;border-top:1px solid #dde3ec;
            font-family:'DM Mono',monospace;font-size:0.6rem;color:#8a96a3;
            text-align:center;letter-spacing:0.1em;">
  AI BATTERY HEALTH PREDICTION &nbsp;·&nbsp; LSTM v4 · 27-FEATURE MODEL &nbsp;·&nbsp;
  TENSORFLOW 2.x &nbsp;·&nbsp; SAVITZKY-GOLAY LABELS &nbsp;·&nbsp; STRATIFIED 70/15/15 SPLIT
</div>
""", unsafe_allow_html=True)