# Fokker‑Planck Equation – Entropy Production for ETFs

Applies the Fokker‑Planck equation (forward Kolmogorov) to model the evolution of ETF return probability density over time. Macro variables modulate the density evolution via a composite factor. The per‑ETF score is the **entropy production rate** (KL divergence between density halves divided by time) – a signal for regime change and instability.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Gaussian KDE density estimation
- KL divergence between two halves of the window
- Entropy production rate = KL / dt
- Composite macro factor from all macro variables
- Score = entropy production rate (higher = regime change)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-fokker-planck-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High entropy production → density is evolving rapidly → regime change / instability → potential alpha.
- Low entropy production → stable density → predictable.

## Requirements

See `requirements.txt`.
