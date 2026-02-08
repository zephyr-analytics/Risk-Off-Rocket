# region imports
from AlgorithmImports import *
import numpy as np
import pandas as pd
import math
# endregion

class RiskOffRocket(QCAlgorithm):
    """
    Trades SPXL and TQQQ as separate competing sleeves using Zephyr-style
    conviction + persistence + vol targeting.
    Bond regime ONLY gates participation.
    Excess risk always goes to SHV.
    """

    def Initialize(self):
        self.SetStartDate(2011, 1, 1)
        self.SetCash(1_000_000)

        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.CASH)
        self.SetSecurityInitializer(lambda s: s.SetFeeModel(ConstantFeeModel(0)))

        # -----------------------------
        # Regime params (UNCHANGED)
        # -----------------------------
        self.hold_bars = 0
        self.lb_short = 3
        self.lb_mid   = 12
        self.lb_long  = 84

        self.risk_off_hold_counter = 0
        self.is_risk_off = False
        self.last_rebalance_date = None

        # -----------------------------
        # Zephyr-style sizing params
        # -----------------------------
        self.winrate_lookback = 63
        self.vol_lookback = 21
        self.group_vol_target = 0.50  # annualized target per sleeve

        # -----------------------------
        # Signal assets (bond regime)
        # -----------------------------
        self.sym_vgit = self.AddEquity("VGIT", Resolution.Daily).Symbol
        self.sym_vcit = self.AddEquity("VCIT", Resolution.Daily).Symbol
        self.sym_hyg  = self.AddEquity("HYG",  Resolution.Daily).Symbol

        # -----------------------------
        # Trade assets (sleeves)
        # -----------------------------
        self.sym_spxl = self.AddEquity("SPXL", Resolution.Daily).Symbol
        self.sym_tqqq = self.AddEquity("TQQQ", Resolution.Daily).Symbol
        self.sym_shv  = self.AddEquity(
            "SHV",
            Resolution.Daily,
            dataNormalizationMode=DataNormalizationMode.TOTAL_RETURN
        ).Symbol

        # -----------------------------
        # Warmup
        # -----------------------------
        self.SetWarmUp(self.lb_long + self.vol_lookback + 5, Resolution.Daily)

        # -----------------------------
        # Schedule
        # -----------------------------
        self.Schedule.On(
            self.DateRules.EveryDay(self.sym_spxl),
            self.TimeRules.BeforeMarketClose(self.sym_spxl, 5),
            self.Rebalance
        )

    # ==================================================
    # History helpers
    # ==================================================
    def _history_closes(self, symbols, bars, norm_mode):
        hist = self.History(
            symbols,
            bars,
            Resolution.Daily,
            dataNormalizationMode=norm_mode
        )
        if hist.empty:
            return None
        return hist["close"].unstack(0).dropna(how="all")

    # ==================================================
    # Bond regime (UNCHANGED, Pine-exact)
    # ==================================================
    def _avg_cum_mom_last(self, symbol):
        bars = self.lb_long + 1
        closes = self._history_closes([symbol], bars, norm_mode=DataNormalizationMode.SCALED_RAW)
        # closes = self._history_closes([symbol], bars)
        if closes is None or symbol not in closes:
            return np.nan

        px = closes[symbol]
        try:
            r10 = px.iloc[-1] / px.iloc[-(self.lb_short + 1)] - 1
            r21 = px.iloc[-1] / px.iloc[-(self.lb_mid + 1)]   - 1
            r63 = px.iloc[-1] / px.iloc[-(self.lb_long + 1)]  - 1
            return (r10 + r21 + r63) / 3.0
        except:
            return np.nan

    def _compute_regime_pine_exact(self):
        mT = self._avg_cum_mom_last(self.sym_vgit)
        mI = self._avg_cum_mom_last(self.sym_vcit)
        mH = self._avg_cum_mom_last(self.sym_hyg)

        if any(math.isnan(x) for x in [mT, mI, mH]):
            return None

        hyHardRiskOff = (mH <= 0)
        treasAboveBoth = (mT > mI) and (mT > mH)
        riskNow = hyHardRiskOff or treasAboveBoth

        if riskNow:
            self.risk_off_hold_counter = self.hold_bars
        elif self.risk_off_hold_counter > 0:
            self.risk_off_hold_counter -= 1

        self.is_risk_off = riskNow or (self.risk_off_hold_counter > 0)

        reason = (
            "HY<0" if hyHardRiskOff else
            "VGIT>VCIT&HYG" if treasAboveBoth else
            "HOLD" if self.risk_off_hold_counter > 0 else
            "—"
        )

        return self.is_risk_off, riskNow, reason, mT, mI, mH, self.risk_off_hold_counter

    # ==================================================
    # Zephyr-style group momentum
    # ==================================================
    def _compute_group_momentum(self, symbol, closes):
        if symbol not in closes:
            return 0.0
        px = closes[symbol]
        if len(px) < 253:
            return 0.0
        return float(np.mean([
            px.iloc[-1] / px.iloc[-22]  - 1,
            px.iloc[-1] / px.iloc[-64]  - 1,
            px.iloc[-1] / px.iloc[-127] - 1,
            px.iloc[-1] / px.iloc[-190] - 1,
            px.iloc[-1] / px.iloc[-253] - 1,
        ]))

    # ==================================================
    # Rebalance
    # ==================================================
    def Rebalance(self):
        if self.IsWarmingUp:
            return
        if self.last_rebalance_date == self.Time.date():
            return
        self.last_rebalance_date = self.Time.date()

        out = self._compute_regime_pine_exact()
        if out is None:
            return

        isRiskOff, riskNow, reason, mT, mI, mH, hold_left = out

        # -----------------------------
        # Risk OFF → 100% SHV
        # -----------------------------
        if isRiskOff:
            self.SetHoldings(self.sym_spxl, 0.0)
            self.SetHoldings(self.sym_tqqq, 0.0)
            self.SetHoldings(self.sym_shv,  1.0)
            return

        # -----------------------------
        # Zephyr-style sizing block
        # -----------------------------
        risk_groups = {
            "spxl": self.sym_spxl,
            "tqqq": self.sym_tqqq
        }

        closes = self._history_closes(
            list(risk_groups.values()),
            self.vol_lookback + 253,
            norm_mode=DataNormalizationMode.TOTAL_RETURN
        )
        if closes is None:
            return

        edges, vols = {}, {}

        for g, sym in risk_groups.items():
            if sym not in closes:
                continue

            g_rets = closes[sym].pct_change().dropna()
            if len(g_rets) < self.winrate_lookback:
                continue

            log_group = np.log1p(g_rets)
            p_win = float(np.mean(log_group.tail(self.winrate_lookback) > 0))

            group_mom = self._compute_group_momentum(sym, closes)

            if not np.isfinite(group_mom) or group_mom <= 0:
                continue

            g_vol = float(
                np.std(log_group.tail(self.vol_lookback)) * np.sqrt(252)
            )

            if not np.isfinite(g_vol) or g_vol <= 0:
                continue

            vols[g] = g_vol

            # Pattern Match: Confidence calculation
            confidence = group_mom / (g_vol + 1e-6)
            edge = p_win * (1.0 + confidence)

            edges[g] = edge

        if not edges:
            self.SetHoldings(self.sym_shv, 1.0)
            return

        # -----------------------------
        # Normalize + vol targeting
        # -----------------------------
        edge_eff = {g: e + 0.01 for g, e in edges.items()}
        total_edge = sum(edge_eff.values())

        w_raw = {g: e / total_edge for g, e in edge_eff.items()}

        w_scaled = {
            g: w_raw[g] * min(1.0, self.group_vol_target / vols[g])
            for g in w_raw
        }

        risk_weight = sum(w_scaled.values())
        cash_weight = max(0.0, 1.0 - risk_weight)

        # -----------------------------
        # Allocate
        # -----------------------------
        self.SetHoldings(self.sym_spxl, w_scaled.get("spxl", 0.0))
        self.SetHoldings(self.sym_tqqq, w_scaled.get("tqqq", 0.0))
        self.SetHoldings(self.sym_shv,  cash_weight)

        # -----------------------------
        # Debug
        # -----------------------------
        self.Debug(
            f"{self.Time.date()} "
            f"riskNow={int(riskNow)} isRiskOff={int(isRiskOff)} reason={reason} "
            f"SPXL={w_scaled.get('spxl',0):.3f} "
            f"TQQQ={w_scaled.get('tqqq',0):.3f} "
            f"CASH={cash_weight:.3f}"
        )
