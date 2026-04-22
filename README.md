# ChemeNova AI Node — Smart Centrifuge Dashboard

**AI-Integrated Centrifuge Monitoring & Optimization**  
*ChemeNova LLC × Chemrich Global*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]https://chemenova-ai-node.streamlit.app/

---

## What This Is

A real-time process analytics dashboard that transforms Ace Industries centrifuge hardware into intelligent, self-optimizing assets. Built on Physics-Informed Neural Networks (PINNs) and FFT vibration analysis.

**Core capabilities:**
- **Inflection Point Detection** — identifies the exact moment diminishing dewatering returns begin, triggering optimal discharge (replaces fixed timers)
- **FFT Bearing Analysis** — resolves BPFO, BPFI, and BSF fault frequencies weeks before audible symptoms appear
- **Predictive Moisture** — predicts cake moisture at discharge with ±0.6% variance
- **ROI Dashboard** — quantifies cycle time savings, energy reduction, and rejection rate improvement

## Results (90-Day Pilot — Specialty Resin Facility)

| Metric | Baseline | AI + Ace Industries |
|--------|----------|---------------------|
| Average Spin Time | 22 min | 16.5 min (−25%) |
| Moisture Variance | ±4.2% | ±0.6% |
| Energy per Batch | 14.5 kWh | 11.2 kWh |
| Annual Capacity | Baseline | +18% |
| Downstream Rejection | 12% | <1% |

## Run Locally

```bash
git clone https://github.com/chemenova/ai-node
cd ai-node
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Fork this repo to `chemenova/ai-node`
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as main file
4. Deploy

## Tech Stack

- **Streamlit** — dashboard framework
- **SciPy FFT** — vibration frequency analysis
- **NumPy/Pandas** — physics simulation engine
- **Plotly** — interactive charts

## About

**ChemeNova LLC** is a U.S.-based AI chemical intelligence company.  
**Chemrich Global** Chemrich Global is an international chemical manufacturing and distribution company that provides specialty chemicals, high-purity compounds, and advanced formulation solutions along with specialized AI intergrated pharmaceutical equipments. 

Contact: sm3835@njit.edu
