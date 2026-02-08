import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime

# ======================================================
# CONFIG (MATCHES QC)
# ======================================================
BOND_TICKERS = ["VGIT", "VCIT", "HYG"]
RISK_TICKERS = ["SPXL", "TQQQ"]
CASH_TICKER = "SHV"

LB_SHORT = 10
LB_MID   = 21
LB_LONG  = 63

WINRATE_LOOKBACK = 63
VOL_LOOKBACK = 21
GROUP_VOL_TARGET = 0.50  # annualized

HOLD_BARS = 1

LOOKBACK = max(253 + VOL_LOOKBACK, LB_LONG) + 5


# ======================================================
# DATA FETCH
# ======================================================
def fetch_prices(tickers, lookback, adjusted=False):
    data = yf.download(
        tickers,
        period=f"{lookback + 10}d",
        interval="1d",
        auto_adjust=adjusted,
        progress=False
    )
    return data["Close"].dropna(how="all")


# ======================================================
# BOND REGIME (UNADJUSTED, PINE-EXACT)
# ======================================================
def avg_cum_momentum(px):
    try:
        r10 = px.iloc[-1] / px.iloc[-1 - LB_SHORT] - 1
        r21 = px.iloc[-1] / px.iloc[-1 - LB_MID]   - 1
        r63 = px.iloc[-1] / px.iloc[-1 - LB_LONG]  - 1
        return (r10 + r21 + r63) / 3.0
    except:
        return np.nan


def compute_bond_regime(closes, prev_hold=0):
    mT = avg_cum_momentum(closes["VGIT"])
    mI = avg_cum_momentum(closes["VCIT"])
    mH = avg_cum_momentum(closes["HYG"])

    if any(math.isnan(x) for x in [mT, mI, mH]):
        return None

    hyHardRiskOff = mH <= 0
    treasAboveBoth = (mT > mI) and (mT > mH)

    riskNow = hyHardRiskOff or treasAboveBoth

    if riskNow:
        hold = HOLD_BARS
    elif prev_hold > 0:
        hold = prev_hold - 1
    else:
        hold = 0

    isRiskOff = riskNow or hold > 0

    reason = (
        "HY<0" if hyHardRiskOff else
        "VGIT>VCIT&HYG" if treasAboveBoth else
        "HOLD" if hold > 0 else
        "—"
    )

    return isRiskOff, riskNow, reason, mT, mI, mH, hold


# ======================================================
# ZEPHYR-STYLE SIZING
# ======================================================
def compute_group_momentum(px):
    return float(np.mean([
        px.iloc[-1] / px.iloc[-22]  - 1,
        px.iloc[-1] / px.iloc[-64]  - 1,
        px.iloc[-1] / px.iloc[-127] - 1,
        px.iloc[-1] / px.iloc[-190] - 1,
        px.iloc[-1] / px.iloc[-253] - 1,
    ]))


def compute_weights(risk_closes):
    edges, vols = {}, {}

    for sym in RISK_TICKERS:
        px = risk_closes[sym].dropna()
        rets = px.pct_change().dropna()

        if len(rets) < WINRATE_LOOKBACK:
            continue

        p_win = float(np.mean(rets.tail(WINRATE_LOOKBACK) > 0))

        group_mom = compute_group_momentum(px)
        mom_std = float(np.std(rets.tail(VOL_LOOKBACK)))

        confidence = abs(group_mom) / (mom_std + 1e-6)
        confidence = np.clip(confidence, 0.0, 2.0)

        scale = max(0.1, 1.0 + confidence * np.sign(group_mom))
        edge = p_win * scale

        g_vol = float(
            np.std(np.log1p(rets.tail(VOL_LOOKBACK))) * np.sqrt(252)
        )

        if g_vol <= 0:
            continue

        edges[sym] = edge
        vols[sym] = g_vol

    if not edges:
        return {"SPXL": 0.0, "TQQQ": 0.0, "SHV": 1.0}

    edge_eff = {k: v + 0.01 for k, v in edges.items()}
    total_edge = sum(edge_eff.values())

    w_raw = {k: v / total_edge for k, v in edge_eff.items()}
    w_scaled = {
        k: w_raw[k] * min(1.0, GROUP_VOL_TARGET / vols[k])
        for k in w_raw
    }

    risk_weight = sum(w_scaled.values())
    cash_weight = max(0.0, 1.0 - risk_weight)

    return {
        "SPXL": w_scaled.get("SPXL", 0.0),
        "TQQQ": w_scaled.get("TQQQ", 0.0),
        "SHV": cash_weight
    }


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    bond_closes = fetch_prices(BOND_TICKERS, LOOKBACK, adjusted=False)
    risk_closes = fetch_prices(RISK_TICKERS, LOOKBACK, adjusted=True)

    regime = compute_bond_regime(bond_closes)

    if regime is None:
        raise RuntimeError("Insufficient bond data")

    isRiskOff, riskNow, reason, mT, mI, mH, hold = regime

    if isRiskOff:
        weights = {"SPXL": 0.0, "TQQQ": 0.0, "SHV": 1.0}
    else:
        weights = compute_weights(risk_closes)

    print("\n=== PORTFOLIO SIGNAL ===")
    print(f"Date        : {bond_closes.index[-1].date()}")
    print(f"Risk NOW    : {int(riskNow)}")
    print(f"Risk OFF    : {int(isRiskOff)}")
    print(f"Reason      : {reason}")
    print(f"VGIT mom    : {mT:.4f}")
    print(f"VCIT mom    : {mI:.4f}")
    print(f"HYG mom     : {mH:.4f}")
    print("\n--- WEIGHTS ---")
    for k, v in weights.items():
        print(f"{k:5s}: {v:.3f}")
