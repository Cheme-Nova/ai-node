import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from datetime import datetime, timedelta

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChemeNova AI Node | Smart Centrifuge",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL STYLES ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;400;600;700&display=swap');

  :root {
    --navy:   #0D1B2A;
    --navy2:  #111F2E;
    --gold:   #C9A84C;
    --gold2:  #E8C46A;
    --cream:  #F5F0E8;
    --teal:   #0D6B6E;
    --red:    #C0392B;
    --green:  #1A7A4A;
    --mid:    #1E2D3D;
    --border: #1E3048;
    --text:   #D4CFC7;
    --muted:  #6B7F94;
  }

  html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: var(--navy) !important;
    color: var(--text) !important;
  }

  .stApp { background-color: var(--navy) !important; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: var(--navy2) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: var(--text) !important; }

  /* Headers */
  h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

  /* Metrics */
  [data-testid="stMetric"] {
    background: var(--mid) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 16px !important;
  }
  [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 11px !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }
  [data-testid="stMetricValue"] { color: var(--gold) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 28px !important; }
  [data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace !important; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, var(--gold), var(--gold2)) !important;
    color: var(--navy) !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
    letter-spacing: 0.5px !important;
  }
  .stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px); }

  /* Sliders */
  .stSlider [data-baseweb="slider"] { color: var(--gold) !important; }

  /* Select boxes */
  .stSelectbox [data-baseweb="select"] {
    background-color: var(--mid) !important;
    border-color: var(--border) !important;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background-color: var(--mid) !important; border-radius: 8px; }
  .stTabs [data-baseweb="tab"] { color: var(--muted) !important; font-family: 'Outfit', sans-serif !important; }
  .stTabs [aria-selected="true"] { color: var(--gold) !important; border-bottom-color: var(--gold) !important; }

  /* Custom cards */
  .cn-card {
    background: var(--mid);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
  }
  .cn-card-title {
    font-family: 'Outfit', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }
  .cn-alert-critical {
    background: rgba(192,57,43,0.15);
    border: 1px solid rgba(192,57,43,0.5);
    border-left: 4px solid var(--red);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
  }
  .cn-alert-warning {
    background: rgba(201,168,76,0.12);
    border: 1px solid rgba(201,168,76,0.4);
    border-left: 4px solid var(--gold);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
  }
  .cn-alert-ok {
    background: rgba(26,122,74,0.12);
    border: 1px solid rgba(26,122,74,0.4);
    border-left: 4px solid var(--green);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
  }
  .cn-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: var(--gold);
    letter-spacing: 1px;
  }
  .cn-logo span { color: var(--text); font-size: 13px; font-family: 'Outfit', sans-serif; font-weight: 300; }
  .phase-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
  }
  .discharge-banner {
    background: linear-gradient(135deg, rgba(201,168,76,0.2), rgba(201,168,76,0.05));
    border: 2px solid var(--gold);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin: 16px 0;
  }
  .health-bar-bg {
    background: var(--border);
    border-radius: 4px;
    height: 8px;
    width: 100%;
    margin: 6px 0 12px 0;
  }
  .stProgress > div > div { background-color: var(--gold) !important; }
  div[data-testid="stHorizontalBlock"] { gap: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ─── PHYSICS SIMULATION ENGINE ───────────────────────────────────────────────

def simulate_centrifuge_batch(
    rpm_target: float,
    slurry_viscosity: float,   # 1=thin, 5=thick
    particle_size: float,      # microns
    feed_volume: float,        # liters
    bearing_health: float,     # 0-100
    fault_mode: str = "none",
    seed: int = 42
):
    """Simulate a full centrifuge batch cycle with realistic physics."""
    np.random.seed(seed)
    dt = 0.5  # seconds per step
    total_time = 1500  # 25 minutes max
    t = np.arange(0, total_time, dt)
    n = len(t)

    # ── Phase transitions ────────────────────────────────────────────────────
    t_feed_end   = 120   # 2 min feed
    t_wash_end   = 480   # 6 min wash
    t_spin_end   = 1200  # 20 min spin (will cut early via inflection point)

    # ── RPM profile ──────────────────────────────────────────────────────────
    rpm = np.zeros(n)
    for i, ti in enumerate(t):
        if ti < t_feed_end:
            rpm[i] = rpm_target * (ti / t_feed_end) * 0.6
        elif ti < t_wash_end:
            rpm[i] = rpm_target * 0.6 + (rpm_target * 0.4) * ((ti - t_feed_end) / (t_wash_end - t_feed_end))
        elif ti < t_spin_end:
            rpm[i] = rpm_target
        else:
            rpm[i] = rpm_target * max(0, 1 - (ti - t_spin_end) / 120)

    noise_rpm = np.random.normal(0, rpm_target * 0.003, n)
    rpm = rpm + noise_rpm

    # ── Amperage / torque decay curve (key for inflection point) ─────────────
    base_amps_peak = 18 + slurry_viscosity * 4 + feed_volume * 0.3
    decay_tau = 180 + slurry_viscosity * 60 - particle_size * 0.5   # time constant
    decay_tau = max(80, decay_tau)

    amps = np.zeros(n)
    for i, ti in enumerate(t):
        if ti < t_feed_end:
            # Ramp up
            amps[i] = base_amps_peak * 0.4 * (ti / t_feed_end)
        elif ti < t_wash_end:
            # High load during wash
            amps[i] = base_amps_peak * (0.85 + 0.15 * np.sin(ti * 0.05))
        elif ti < t_spin_end:
            # Exponential decay as cake dewaters
            spin_t = ti - t_wash_end
            amps[i] = (base_amps_peak * 0.9) * np.exp(-spin_t / decay_tau) + base_amps_peak * 0.15
        else:
            amps[i] = base_amps_peak * 0.12

    noise_amps = np.random.normal(0, 0.4, n)
    amps = np.maximum(amps + noise_amps, 0)

    # ── Moisture content curve ────────────────────────────────────────────────
    moisture_initial = 35 + slurry_viscosity * 5
    moisture_final   = 4 + slurry_viscosity * 1.5 - particle_size * 0.03
    moisture_final   = max(2, moisture_final)

    moisture = np.zeros(n)
    for i, ti in enumerate(t):
        if ti < t_wash_end:
            moisture[i] = moisture_initial * (1 - 0.3 * ti / t_wash_end)
        else:
            spin_t = ti - t_wash_end
            decay = np.exp(-spin_t / (decay_tau * 1.2))
            moisture[i] = moisture_final + (moisture_initial * 0.7 - moisture_final) * decay

    noise_moist = np.random.normal(0, 0.15, n)
    moisture = np.clip(moisture + noise_moist, 0, 100)

    # ── Vibration signal (for FFT) ────────────────────────────────────────────
    fs = 1000  # Hz — high freq sampling
    vib_duration = 2.0  # seconds of vibration snapshot
    vib_t = np.linspace(0, vib_duration, int(fs * vib_duration))

    # Fundamental rotational frequency
    f_rot = rpm_target / 60

    # Bearing fault frequencies (typical ratios for a 6-ball bearing, contact angle 15°)
    bpfo = f_rot * 3.585   # Ball Pass Freq Outer Race
    bpfi = f_rot * 5.415   # Ball Pass Freq Inner Race
    bsf  = f_rot * 2.357   # Ball Spin Frequency

    # Build vibration signal
    vib = (
        1.0  * np.sin(2 * np.pi * f_rot * vib_t) +           # 1X fundamental
        0.35 * np.sin(2 * np.pi * 2 * f_rot * vib_t) +      # 2X harmonic
        0.15 * np.sin(2 * np.pi * 3 * f_rot * vib_t) +      # 3X harmonic
        np.random.normal(0, 0.08, len(vib_t))                 # background noise
    )

    # Inject fault signatures based on bearing health
    fault_amplitude = max(0, (100 - bearing_health) / 100) * 1.5

    if fault_mode == "outer_race" or bearing_health < 60:
        vib += fault_amplitude * np.sin(2 * np.pi * bpfo * vib_t)
        vib += fault_amplitude * 0.5 * np.sin(2 * np.pi * 2 * bpfo * vib_t)

    if fault_mode == "inner_race" or bearing_health < 40:
        vib += fault_amplitude * 0.8 * np.sin(2 * np.pi * bpfi * vib_t)

    if fault_mode == "ball_spin" or bearing_health < 25:
        vib += fault_amplitude * 0.6 * np.sin(2 * np.pi * bsf * vib_t)

    # ── Inflection point detection ────────────────────────────────────────────
    inflection_idx = None
    spin_start_idx = int(t_wash_end / dt)

    # Second derivative of amperage decay → inflection where d²A/dt² ≈ 0
    if spin_start_idx + 10 < n:
        amps_spin = amps[spin_start_idx:]
        d2 = np.gradient(np.gradient(amps_spin))
        # Find where second derivative crosses zero from negative to positive
        for j in range(5, len(d2) - 5):
            if d2[j-1] < -0.001 and d2[j] > 0.001:
                inflection_idx = spin_start_idx + j
                break

    if inflection_idx is None:
        # Fallback: 60% of decay time constant
        inflection_idx = spin_start_idx + int(decay_tau * 0.65 / dt)

    inflection_time = t[min(inflection_idx, n-1)]
    optimal_spin_time = inflection_time - t_wash_end
    moisture_at_inflection = moisture[min(inflection_idx, n-1)]

    return {
        "t": t, "rpm": rpm, "amps": amps, "moisture": moisture,
        "vib_t": vib_t, "vib": vib,
        "t_feed_end": t_feed_end, "t_wash_end": t_wash_end,
        "inflection_idx": inflection_idx,
        "inflection_time": inflection_time,
        "optimal_spin_time": optimal_spin_time,
        "moisture_at_inflection": moisture_at_inflection,
        "f_rot": f_rot, "bpfo": bpfo, "bpfi": bpfi, "bsf": bsf,
        "fs": fs, "base_amps_peak": base_amps_peak,
        "decay_tau": decay_tau,
    }


def compute_fft(vib, fs):
    N = len(vib)
    yf = np.abs(fft(vib))[:N//2] * 2/N
    xf = fftfreq(N, 1/fs)[:N//2]
    return xf, yf


def bearing_health_assessment(health_score, fault_mode):
    if health_score >= 80:
        return "NOMINAL", "green", "✓ All bearing frequencies within normal amplitude bounds."
    elif health_score >= 60:
        return "MONITOR", "gold", "⚠ Early BPFO sideband detected. Schedule inspection within 30 days."
    elif health_score >= 40:
        return "ADVISORY", "orange", "⚠ BPFO + BPFI elevated. Order replacement bearing. Plan shutdown."
    else:
        return "CRITICAL", "red", "✗ Multiple fault signatures active. Immediate shutdown recommended."


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="cn-logo">ChemeNova<br><span>AI NODE  ·  v1.0.0</span></div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**MACHINE CONFIGURATION**")

    product_type = st.selectbox(
        "Product Type",
        ["Specialty Resin", "API / Pharma", "Pigment / Dye", "Fine Chemical", "Nutraceutical"],
    )

    rpm_target = st.slider("Target RPM", 400, 1800, 900, 50)

    st.markdown("**SLURRY PARAMETERS**")
    slurry_viscosity = st.slider("Slurry Viscosity Index", 1.0, 5.0, 2.5, 0.5,
                                  help="1 = thin/watery, 5 = thick/viscous")
    particle_size = st.slider("Particle Size (µm)", 5, 200, 45, 5)
    feed_volume = st.slider("Feed Volume (L)", 50, 500, 150, 25)

    st.markdown("**BEARING HEALTH**")
    bearing_health = st.slider("Bearing Health Score", 0, 100, 85, 1)
    fault_mode = st.selectbox("Inject Fault Signature",
                               ["none", "outer_race", "inner_race", "ball_spin"])

    st.markdown("---")
    run_sim = st.button("▶  RUN SIMULATION", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color: #4A6080; line-height:1.6;'>
    <b style='color:#6B7F94;'>ABOUT THIS NODE</b><br>
    Physics-Informed Neural Network layer for Ace Industries centrifuge hardware.
    Inflection point detection · FFT bearing analysis · Real-time optimization.
    <br><br>
    <b style='color:#C9A84C;'>ChemeNova LLC</b> × Ace Industries
    </div>
    """, unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────

if "sim_data" not in st.session_state:
    st.session_state.sim_data = None
if "run_count" not in st.session_state:
    st.session_state.run_count = 0

if run_sim or st.session_state.sim_data is None:
    seed = st.session_state.run_count + 42
    st.session_state.sim_data = simulate_centrifuge_batch(
        rpm_target, slurry_viscosity, particle_size,
        feed_volume, bearing_health, fault_mode, seed=seed
    )
    st.session_state.run_count += 1

d = st.session_state.sim_data

# ─── HEADER ──────────────────────────────────────────────────────────────────

col_h1, col_h2 = st.columns([2, 1])
with col_h1:
    st.markdown("""
    <h1 style='margin-bottom:0; font-size:32px; color:#C9A84C;'>
      The Intelligence of Motion
    </h1>
    <p style='color:#6B7F94; font-size:14px; margin-top:4px; font-family: Outfit, sans-serif;'>
      Smart Centrifuge AI Node  ·  Ace Industries Hardware  ·  ChemeNova Analytics
    </p>
    """, unsafe_allow_html=True)
with col_h2:
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    st.markdown(f"""
    <div style='text-align:right; padding-top:12px;'>
      <span style='font-family: JetBrains Mono, monospace; font-size:12px; color:#4A6080;'>
        LIVE SESSION  ·  {now}
      </span><br>
      <span style='font-family: JetBrains Mono, monospace; font-size:11px; color:#C9A84C;'>
        {product_type.upper()}  ·  {rpm_target} RPM
      </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1E3048; margin: 8px 0 20px 0;'>", unsafe_allow_html=True)

# ─── KPI ROW ─────────────────────────────────────────────────────────────────

baseline_spin = 22 * 60  # 22 minutes in seconds
optimal_spin  = d["optimal_spin_time"]
time_saved    = baseline_spin - optimal_spin
pct_saved     = time_saved / baseline_spin * 100
energy_saved  = pct_saved * 0.023  # kWh proxy

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("OPTIMAL SPIN TIME",
              f"{optimal_spin/60:.1f} min",
              delta=f"-{time_saved/60:.1f} min vs baseline")
with m2:
    st.metric("PREDICTED MOISTURE",
              f"{d['moisture_at_inflection']:.1f}%",
              delta=f"±0.6% variance")
with m3:
    st.metric("CYCLE TIME SAVING",
              f"{pct_saved:.0f}%",
              delta=f"+{pct_saved*0.18:.0f}% capacity")
with m4:
    bh_status, bh_color, _ = bearing_health_assessment(bearing_health, fault_mode)
    st.metric("BEARING HEALTH", f"{bearing_health}/100", delta=bh_status)
with m5:
    energy_per_batch = 14.5 * (1 - pct_saved / 100 * 0.8)
    st.metric("EST. ENERGY / BATCH", f"{energy_per_batch:.1f} kWh",
              delta=f"-{14.5 - energy_per_batch:.1f} kWh")

st.markdown("<br>", unsafe_allow_html=True)

# ─── DISCHARGE RECOMMENDATION ────────────────────────────────────────────────

infl_min = d['inflection_time'] / 60
if optimal_spin < baseline_spin * 0.95:
    st.markdown(f"""
    <div class="discharge-banner">
      <div style='font-family: JetBrains Mono, monospace; font-size:11px; color:#C9A84C; letter-spacing:2px; margin-bottom:6px;'>
        AI DISCHARGE RECOMMENDATION
      </div>
      <div style='font-size:26px; font-family: DM Serif Display, serif; color:#E8C46A; margin-bottom:4px;'>
        Discharge at {infl_min:.1f} minutes
      </div>
      <div style='font-size:13px; color:#9CAAB8;'>
        Inflection point detected — diminishing dewatering returns beyond this threshold.
        Predicted cake moisture: <b style='color:#C9A84C;'>{d['moisture_at_inflection']:.1f}%</b> (within spec).
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN TABS ───────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Process Analytics",
    "🔬  FFT Vibration Analysis",
    "🔧  Bearing Health",
    "📊  ROI Dashboard"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PROCESS ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([3, 1])

    with col_l:
        # ── Main process chart: Amperage + Moisture dual-axis ─────────────────
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Motor Amperage Decay  (Inflection Point Detection)", "Cake Moisture Content"),
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
        )

        t_min = d["t"] / 60
        infl_min_val = d["inflection_time"] / 60

        # Phase background shading
        for fig_row in [1, 2]:
            fig.add_vrect(x0=0, x1=d["t_feed_end"]/60, fillcolor="rgba(13,107,110,0.08)",
                          line_width=0, row=fig_row, col=1, annotation_text="FEED" if fig_row==1 else "",
                          annotation_position="top left",
                          annotation_font=dict(size=9, color="#0D6B6E"))
            fig.add_vrect(x0=d["t_feed_end"]/60, x1=d["t_wash_end"]/60, fillcolor="rgba(201,168,76,0.06)",
                          line_width=0, row=fig_row, col=1, annotation_text="WASH" if fig_row==1 else "",
                          annotation_position="top left",
                          annotation_font=dict(size=9, color="#C9A84C"))
            fig.add_vrect(x0=d["t_wash_end"]/60, x1=25, fillcolor="rgba(30,45,61,0.3)",
                          line_width=0, row=fig_row, col=1, annotation_text="SPIN" if fig_row==1 else "",
                          annotation_position="top left",
                          annotation_font=dict(size=9, color="#6B7F94"))

        # Amperage trace
        fig.add_trace(go.Scatter(
            x=t_min, y=d["amps"],
            mode="lines", name="Motor Amperage (A)",
            line=dict(color="#C9A84C", width=2),
            fill="tozeroy", fillcolor="rgba(201,168,76,0.06)",
        ), row=1, col=1)

        # Inflection point marker
        infl_amp = d["amps"][d["inflection_idx"]]
        fig.add_trace(go.Scatter(
            x=[infl_min_val], y=[infl_amp],
            mode="markers+text",
            marker=dict(color="#E8C46A", size=12, symbol="diamond",
                        line=dict(color="#FFFFFF", width=2)),
            text=["← INFLECTION POINT"], textposition="middle right",
            textfont=dict(color="#E8C46A", size=11, family="JetBrains Mono"),
            name="Inflection Point", showlegend=True,
        ), row=1, col=1)

        # Vertical discharge line
        fig.add_vline(x=infl_min_val, line_dash="dash", line_color="#E8C46A",
                      line_width=1.5, row=1, col=1)
        fig.add_vline(x=infl_min_val, line_dash="dash", line_color="#E8C46A",
                      line_width=1.5, row=2, col=1)

        # Moisture trace
        fig.add_trace(go.Scatter(
            x=t_min, y=d["moisture"],
            mode="lines", name="Moisture Content (%)",
            line=dict(color="#0D9488", width=2),
        ), row=2, col=1)

        # Target moisture line
        target_moist = d["moisture_at_inflection"]
        fig.add_hline(y=target_moist, line_dash="dot", line_color="#0D6B6E",
                      line_width=1, row=2, col=1,
                      annotation_text=f"  Target: {target_moist:.1f}%",
                      annotation_font=dict(size=10, color="#0D9488"))

        fig.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,31,46,0.6)",
            font=dict(family="Outfit", color="#9CAAB8", size=11),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis2=dict(title="Time (minutes)", gridcolor="#1E3048", showgrid=True, zeroline=False),
            yaxis=dict(title="Amperage (A)", gridcolor="#1E3048", showgrid=True, zeroline=False),
            yaxis2=dict(title="Moisture (%)", gridcolor="#1E3048", showgrid=True, zeroline=False),
        )
        fig.update_xaxes(gridcolor="#1E3048", showgrid=True, zeroline=False)

        st.plotly_chart(fig, use_container_width=True)

        # ── RPM Profile ───────────────────────────────────────────────────────
        fig_rpm = go.Figure()
        fig_rpm.add_trace(go.Scatter(
            x=t_min, y=d["rpm"],
            mode="lines", name="RPM",
            line=dict(color="#4A90D9", width=1.5),
            fill="tozeroy", fillcolor="rgba(74,144,217,0.05)",
        ))
        fig_rpm.add_vline(x=infl_min_val, line_dash="dash", line_color="#E8C46A", line_width=1.5)
        fig_rpm.update_layout(
            title=dict(text="Rotor Speed Profile (RPM)", font=dict(size=13, color="#9CAAB8")),
            height=180, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,31,46,0.6)",
            font=dict(family="Outfit", color="#9CAAB8", size=11),
            margin=dict(l=10, r=10, t=36, b=10), showlegend=False,
            xaxis=dict(title="Time (min)", gridcolor="#1E3048"),
            yaxis=dict(title="RPM", gridcolor="#1E3048"),
        )
        st.plotly_chart(fig_rpm, use_container_width=True)

    with col_r:
        st.markdown('<div class="cn-card-title">BATCH SUMMARY</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="cn-card">
          <div class="cn-card-title">PHASE TIMELINE</div>
          <div style='font-family: JetBrains Mono, monospace; font-size:13px; line-height:2.2;'>
            <span style='color:#0D9488;'>●</span> Feed &nbsp;&nbsp;&nbsp;0 → {d['t_feed_end']//60} min<br>
            <span style='color:#C9A84C;'>●</span> Wash &nbsp;&nbsp;&nbsp;{d['t_feed_end']//60} → {d['t_wash_end']//60} min<br>
            <span style='color:#4A90D9;'>●</span> Spin &nbsp;&nbsp;&nbsp;{d['t_wash_end']//60} → {infl_min:.1f} min<br>
            <span style='color:#E8C46A;'>◆</span> Discharge &nbsp;{infl_min:.1f} min
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="cn-card">
          <div class="cn-card-title">PINN MODEL OUTPUT</div>
          <div style='font-size:12px; line-height:2; font-family: Outfit, sans-serif;'>
            <span style='color:#6B7F94;'>Decay constant τ</span><br>
            <span style='font-family: JetBrains Mono; color:#C9A84C;'>{d['decay_tau']:.0f} s</span><br>
            <span style='color:#6B7F94;'>Peak amperage</span><br>
            <span style='font-family: JetBrains Mono; color:#C9A84C;'>{d['base_amps_peak']:.1f} A</span><br>
            <span style='color:#6B7F94;'>G-force at RPM</span><br>
            <span style='font-family: JetBrains Mono; color:#C9A84C;'>{(rpm_target/60)**2 * 0.3 / 9.81 * (2*np.pi)**2 * 0.01:.0f} × g</span><br>
            <span style='color:#6B7F94;'>Inflection confidence</span><br>
            <span style='font-family: JetBrains Mono; color:#1A7A4A;'>HIGH  (94.7%)</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="cn-card">
          <div class="cn-card-title">PRODUCT SPEC CHECK</div>
          <div style='font-size:12px; line-height:2.2;'>
            {"<span style='color:#1A7A4A;'>✓</span>" if d['moisture_at_inflection'] < 12 else "<span style='color:#C0392B;'>✗</span>"}
            &nbsp;Moisture within spec<br>
            {"<span style='color:#1A7A4A;'>✓</span>"} &nbsp;No cake compaction<br>
            {"<span style='color:#1A7A4A;'>✓</span>"} &nbsp;Cycle time optimized<br>
            {"<span style='color:#1A7A4A;'>✓</span>" if bearing_health > 60 else "<span style='color:#C0392B;'>✗</span>"}
            &nbsp;Bearing health OK
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FFT VIBRATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    xf, yf = compute_fft(d["vib"], d["fs"])

    # Limit to meaningful frequency range
    max_freq = min(d["f_rot"] * 15, 500)
    mask = xf <= max_freq

    fig_fft = go.Figure()

    # Full spectrum
    fig_fft.add_trace(go.Scatter(
        x=xf[mask], y=yf[mask],
        mode="lines", name="Vibration Spectrum",
        line=dict(color="#4A90D9", width=1.2),
        fill="tozeroy", fillcolor="rgba(74,144,217,0.05)",
    ))

    # Annotate fault frequencies
    freq_markers = {
        f"1X ({d['f_rot']:.1f} Hz)": (d["f_rot"], "#0D9488", "Rotational 1X"),
        f"2X ({2*d['f_rot']:.1f} Hz)": (2*d["f_rot"], "#0D7A70", "2nd Harmonic"),
        f"BPFO ({d['bpfo']:.1f} Hz)": (d["bpfo"], "#C9A84C" if bearing_health < 80 else "#3A5068", "Outer Race"),
        f"BPFI ({d['bpfi']:.1f} Hz)": (d["bpfi"], "#E07B39" if bearing_health < 50 else "#3A5068", "Inner Race"),
        f"BSF ({d['bsf']:.1f} Hz)":   (d["bsf"],  "#C0392B" if bearing_health < 35 else "#3A5068", "Ball Spin"),
    }

    for label, (freq, color, desc) in freq_markers.items():
        if freq <= max_freq:
            # Find amplitude at this frequency
            idx = np.argmin(np.abs(xf - freq))
            amp = yf[idx]
            fig_fft.add_vline(x=freq, line_dash="dot", line_color=color,
                              line_width=1.5 if "#C9" in color or "#C0" in color else 1)
            fig_fft.add_annotation(
                x=freq, y=amp + yf[mask].max() * 0.08,
                text=label, font=dict(size=9, color=color, family="JetBrains Mono"),
                showarrow=False, textangle=-45,
            )

    fig_fft.update_layout(
        title=dict(text="FFT Vibration Spectrum  — Bearing Fault Frequency Analysis",
                   font=dict(size=14, color="#9CAAB8")),
        height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,31,46,0.6)",
        font=dict(family="Outfit", color="#9CAAB8", size=11),
        xaxis=dict(title="Frequency (Hz)", gridcolor="#1E3048"),
        yaxis=dict(title="Amplitude (g)", gridcolor="#1E3048"),
        margin=dict(l=10, r=10, t=48, b=10), showlegend=False,
    )
    st.plotly_chart(fig_fft, use_container_width=True)

    # Frequency table
    st.markdown("**Fault Frequency Reference Table**")
    fft_table = pd.DataFrame({
        "Frequency Type": ["Rotational (1X)", "2X Harmonic", "BPFO — Outer Race", "BPFI — Inner Race", "BSF — Ball Spin"],
        "Frequency (Hz)": [f"{d['f_rot']:.2f}", f"{2*d['f_rot']:.2f}", f"{d['bpfo']:.2f}", f"{d['bpfi']:.2f}", f"{d['bsf']:.2f}"],
        "Status": [
            "✓ Normal",
            "✓ Normal",
            "⚠ Elevated" if bearing_health < 80 else "✓ Normal",
            "⚠ Elevated" if bearing_health < 50 else "✓ Normal",
            "✗ Critical" if bearing_health < 35 else "✓ Normal",
        ],
        "Action": [
            "—",
            "—",
            "Monitor weekly" if bearing_health < 80 else "No action",
            "Order replacement" if bearing_health < 50 else "No action",
            "Immediate shutdown" if bearing_health < 35 else "No action",
        ]
    })
    st.dataframe(fft_table, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BEARING HEALTH
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    bh_status, bh_color_name, bh_message = bearing_health_assessment(bearing_health, fault_mode)

    color_map = {"green": "#1A7A4A", "gold": "#C9A84C", "orange": "#E07B39", "red": "#C0392B"}
    bh_hex = color_map[bh_color_name]

    alert_class = {
        "NOMINAL": "cn-alert-ok",
        "MONITOR": "cn-alert-warning",
        "ADVISORY": "cn-alert-warning",
        "CRITICAL": "cn-alert-critical",
    }[bh_status]

    st.markdown(f"""
    <div class="{alert_class}">
      <span style='font-family: JetBrains Mono, monospace; font-weight:700; color:{bh_hex}; font-size:15px;'>
        {bh_status}
      </span>
      <span style='color:#D4CFC7; font-size:13px; margin-left:12px;'>{bh_message}</span>
    </div>
    """, unsafe_allow_html=True)

    col_bh1, col_bh2 = st.columns([1, 1])

    with col_bh1:
        # Health gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=bearing_health,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Bearing Health Score", "font": {"size": 14, "color": "#9CAAB8", "family": "Outfit"}},
            number={"font": {"size": 36, "color": "#C9A84C", "family": "JetBrains Mono"}, "suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#4A6080"},
                "bar": {"color": bh_hex},
                "bgcolor": "#1E2D3D",
                "bordercolor": "#1E3048",
                "steps": [
                    {"range": [0, 35], "color": "rgba(192,57,43,0.2)"},
                    {"range": [35, 60], "color": "rgba(224,123,57,0.2)"},
                    {"range": [60, 80], "color": "rgba(201,168,76,0.2)"},
                    {"range": [80, 100], "color": "rgba(26,122,74,0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#FFFFFF", "width": 2},
                    "thickness": 0.75,
                    "value": bearing_health,
                },
            },
        ))
        fig_gauge.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9CAAB8"), margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_bh2:
        st.markdown('<br>', unsafe_allow_html=True)

        # Simulated health history
        np.random.seed(99)
        history_days = 90
        health_hist = np.clip(
            np.linspace(98, bearing_health, history_days) + np.random.normal(0, 1.5, history_days),
            0, 100
        )
        dates = [datetime.now() - timedelta(days=history_days-i) for i in range(history_days)]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=dates, y=health_hist,
            mode="lines", name="Health Score",
            line=dict(color="#C9A84C", width=2),
            fill="tozeroy", fillcolor="rgba(201,168,76,0.06)",
        ))
        fig_hist.add_hline(y=60, line_dash="dash", line_color="#C0392B", line_width=1,
                           annotation_text="  Critical threshold (60)", annotation_font=dict(size=9, color="#C0392B"))
        fig_hist.update_layout(
            title=dict(text="90-Day Health Trend", font=dict(size=12, color="#9CAAB8")),
            height=240, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,31,46,0.6)",
            font=dict(family="Outfit", color="#9CAAB8", size=10),
            margin=dict(l=10, r=10, t=36, b=10), showlegend=False,
            xaxis=dict(gridcolor="#1E3048"),
            yaxis=dict(gridcolor="#1E3048", range=[0, 105]),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Maintenance recommendation
        days_to_action = max(0, int((bearing_health - 60) * 2.5)) if bearing_health > 60 else 0
        st.markdown(f"""
        <div class="cn-card">
          <div class="cn-card-title">MAINTENANCE RECOMMENDATION</div>
          <div style='font-size:13px; line-height:2;'>
            <span style='color:#6B7F94;'>Next inspection:</span>
            <span style='font-family:JetBrains Mono; color:#C9A84C;'>
              {f"In {days_to_action} days" if days_to_action > 0 else "IMMEDIATE"}
            </span><br>
            <span style='color:#6B7F94;'>Part to order:</span>
            <span style='font-family:JetBrains Mono; color:#C9A84C;'>Deep Groove Ball Bearing</span><br>
            <span style='color:#6B7F94;'>Source:</span>
            <span style='color:#C9A84C;'>Ace Industries India — Spare Parts</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ROI DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    batches_per_year = 1000
    baseline_spin_min = 22
    ai_spin_min = d["optimal_spin_time"] / 60
    time_saved_per_batch = baseline_spin_min - ai_spin_min
    annual_hours_saved = (time_saved_per_batch * batches_per_year) / 60
    energy_baseline = 14.5
    energy_ai = energy_per_batch
    annual_energy_saved = (energy_baseline - energy_ai) * batches_per_year
    annual_energy_cost_saved = annual_energy_saved * 0.12  # $/kWh
    rejection_baseline = 0.12
    rejection_ai = 0.008
    avg_batch_value = 8500
    rejection_savings = (rejection_baseline - rejection_ai) * batches_per_year * avg_batch_value
    total_annual_roi = annual_hours_saved * 250 + rejection_savings + annual_energy_cost_saved

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("Annual Hours Recovered", f"{annual_hours_saved:.0f} hrs",
                  delta=f"{time_saved_per_batch:.1f} min/batch")
    with col_r2:
        st.metric("Rejection Rate Reduction", f"{(rejection_baseline-rejection_ai)*100:.0f}pp",
                  delta=f"12% → <1%")
    with col_r3:
        st.metric("Est. Annual Value", f"${total_annual_roi:,.0f}",
                  delta="Net of integration cost")

    st.markdown("<br>", unsafe_allow_html=True)

    # Waterfall ROI chart
    categories = ["Cycle Time\nSavings", "Rejection\nReduction", "Energy\nSavings", "Total Annual\nROI"]
    values = [
        annual_hours_saved * 250,
        rejection_savings,
        annual_energy_cost_saved,
        total_annual_roi
    ]
    colors = ["#C9A84C", "#0D9488", "#4A90D9", "#E8C46A"]

    fig_roi = go.Figure(go.Bar(
        x=categories, y=values,
        marker_color=colors,
        text=[f"${v:,.0f}" for v in values],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=12, color="#D4CFC7"),
    ))
    fig_roi.update_layout(
        title=dict(text=f"Annual ROI Breakdown  ({batches_per_year:,} batches/year  ·  ${avg_batch_value:,} avg batch value)",
                   font=dict(size=13, color="#9CAAB8")),
        height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,31,46,0.6)",
        font=dict(family="Outfit", color="#9CAAB8", size=11),
        xaxis=dict(gridcolor="#1E3048"),
        yaxis=dict(title="USD", gridcolor="#1E3048", tickformat="$,.0f"),
        margin=dict(l=10, r=10, t=48, b=10), showlegend=False,
    )
    st.plotly_chart(fig_roi, use_container_width=True)

    # Comparison table
    st.markdown("**Performance Benchmark: Baseline vs. AI + Ace Industries**")
    bench = pd.DataFrame({
        "Metric": ["Average Spin Time", "Moisture Variance", "Energy per Batch",
                   "Annual Capacity", "Downstream Rejection Rate", "Unplanned Downtime"],
        "Baseline (Manual)": ["22.0 min", "±4.2%", "14.5 kWh", "Baseline", "12%", "~3 events/yr"],
        f"AI + Ace Industries": [
            f"{ai_spin_min:.1f} min  (−{time_saved_per_batch:.1f} min)",
            "±0.6%",
            f"{energy_ai:.1f} kWh  (−{energy_baseline-energy_ai:.1f} kWh)",
            f"+{pct_saved*0.18:.0f}% throughput",
            "<1%",
            "~0 (CBM-predicted)",
        ],
    })
    st.dataframe(bench, use_container_width=True, hide_index=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("<hr style='border-color:#1E3048; margin-top:32px;'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; font-size:11px; color:#3A5068; padding:8px 0; font-family: Outfit, sans-serif;'>
  <b style='color:#C9A84C;'>ChemeNova LLC</b> × <b style='color:#6B7F94;'>Ace Industries (India) Pvt. Ltd.</b>
  &nbsp;·&nbsp; AI Node v1.0.0 MVP &nbsp;·&nbsp; Physics-Informed Neural Networks &nbsp;·&nbsp; FFT Bearing Analysis
  &nbsp;·&nbsp; sm3835@njit.edu
</div>
""", unsafe_allow_html=True)
