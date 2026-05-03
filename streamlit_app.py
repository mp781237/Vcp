"""Streamlit dashboard for Taiwan VCP + CANSLIM-lite system.

Run locally:
    streamlit run streamlit_app.py

Deploy to Streamlit Cloud:
    Point at this repo + this file. requirements.txt at repo root handles deps.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import streamlit as st

from vcp.backtest import BacktestConfig
from vcp.backtest import run as run_backtest
from vcp.data import load, load_universe
from vcp.indicators import sma
from vcp.metrics import summary
from vcp.screener import ScreenerConfig, explain, screen_at

st.set_page_config(page_title="台股 VCP + CANSLIM 儀表板", layout="wide")
st.title("台股 VCP + CANSLIM-lite Dashboard")
st.caption(
    "Mark Minervini's Trend Template + VCP × William O'Neil's CANSLIM (lite). "
    "資料來源 yfinance，Universe 130 檔（手動策劃）。"
)


@st.cache_data(ttl=3600, show_spinner="抓資料 + 跑 screener 中（首次約 2-5 分鐘）...")
def cached_screen(date_iso: str, min_vcp: float, all_canslim: bool) -> pd.DataFrame:
    cfg = ScreenerConfig(min_vcp_score=min_vcp, require_all_canslim=all_canslim)
    return screen_at(date_iso, cfg=cfg)


@st.cache_data(ttl=3600, show_spinner="計算診斷中...")
def cached_explain(ticker: str, date_iso: str) -> dict:
    return explain(ticker, date_iso)


@st.cache_data(ttl=3600, show_spinner="下載 K 線資料...")
def cached_load(ticker: str, start: str, end: str) -> pd.DataFrame:
    return load(ticker, start, end)


@st.cache_data(ttl=3600, show_spinner="跑回測中（首次較久）...")
def cached_backtest(
    start_iso: str,
    end_iso: str,
    initial_capital: float,
    max_positions: int,
    stop_loss_pct: float,
    profit_target_pct: float,
    max_hold_days: int,
    min_vcp_score: float,
) -> tuple[pd.Series, pd.DataFrame]:
    cfg = BacktestConfig(
        initial_capital=initial_capital,
        max_positions=max_positions,
        stop_loss_pct=stop_loss_pct,
        profit_target_pct=profit_target_pct,
        max_hold_days=max_hold_days,
        min_vcp_score=min_vcp_score,
    )
    result = run_backtest(start_iso, end_iso, cfg=cfg)
    return result.equity_curve, result.trades


with st.sidebar:
    st.markdown("### Universe")
    try:
        u = load_universe()
        st.write(f"共 **{len(u)}** 檔")
        st.write(
            f"TWSE: {(u['market'] == 'TWSE').sum()} ／ "
            f"OTC: {(u['market'] == 'TWO').sum()}"
        )
        with st.expander("ticker 清單"):
            st.dataframe(u[["symbol", "name", "market"]], hide_index=True, height=300)
    except Exception as e:
        st.error(f"universe 載入失敗：{e}")

    st.markdown("---")
    st.markdown("### 注意")
    st.caption(
        "首次跑會下載 130 檔歷史資料（~2-5 分鐘），結果快取 1 小時。"
        " Streamlit Cloud 上的快取會在容器重啟後消失。"
    )


tab_screen, tab_detail, tab_backtest = st.tabs(
    ["📋 Screener", "🔍 Stock Detail", "📈 Backtest"]
)


with tab_screen:
    st.subheader("跑 screener")
    c1, c2, c3 = st.columns([1, 1, 1])
    s_date = c1.date_input(
        "As-of 日期",
        pd.Timestamp("2024-12-31").date(),
        max_value=pd.Timestamp.today().date(),
    )
    s_min_vcp = c2.slider("最低 VCP score", 0.0, 1.0, 0.6, 0.05)
    s_all_canslim = c3.checkbox("L+S+N+M 全要過", value=False)

    if st.button("Run screener", type="primary", key="run_screen"):
        df = cached_screen(s_date.isoformat(), s_min_vcp, s_all_canslim)
        st.session_state["screen_result"] = df

    df = st.session_state.get("screen_result")
    if df is None:
        st.info("點上面的按鈕開始")
    elif df.empty:
        st.warning("沒有符合條件的股票，試著放寬 min VCP 或選別的日期")
    else:
        st.success(f"{len(df)} 檔通過")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "下載 CSV",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"screen_{s_date.isoformat()}.csv",
            mime="text/csv",
        )


with tab_detail:
    st.subheader("單檔診斷 + K 線")
    c1, c2 = st.columns([1, 1])
    d_ticker = c1.text_input(
        "Ticker（含 .TW / .TWO 後綴）",
        value="2330.TW",
        help="範例：2330.TW（台積電）、6531.TWO（愛普）",
    )
    d_date = c2.date_input(
        "日期",
        pd.Timestamp("2024-12-31").date(),
        max_value=pd.Timestamp.today().date(),
        key="detail_date",
    )

    if st.button("顯示診斷", key="show_detail"):
        diag = cached_explain(d_ticker, d_date.isoformat())
        if "error" in diag:
            st.error(diag["error"])
        else:
            st.session_state["detail_diag"] = diag

    diag = st.session_state.get("detail_diag")
    if diag and "error" not in diag:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Close", f'{diag["close"]:.2f}')
        rs = diag.get("rs_rating")
        c2.metric("RS Rating", f"{rs:.0f}" if rs is not None else "N/A")
        c3.metric("VCP Score", f'{diag["vcp"]["vcp_score"]:.2f}')
        c4.metric("Extended", "⚠️ Yes" if diag["vcp"]["extended"] else "✅ No")

        col_tt, col_cs, col_vp = st.columns(3)
        with col_tt:
            st.markdown("**Trend Template (Minervini)**")
            tt = diag["trend_template"]
            for k, v in tt.items():
                if k == "passed":
                    continue
                st.write(("✅" if v else "❌") + f" {k}")
            st.markdown(f"**Overall: {'✅ PASS' if tt['passed'] else '❌ FAIL'}**")

        with col_cs:
            st.markdown("**CANSLIM-lite**")
            cs = diag["canslim_lite"]
            for k in ["L_leader", "S_supply_demand", "N_new_high", "M_market_uptrend"]:
                st.write(("✅" if cs[k] else "❌") + f" {k}")
            st.markdown(
                f"**Core (L+M): {'✅' if cs['canslim_core'] else '❌'} ／ "
                f"All: {'✅' if cs['canslim_all'] else '❌'}**"
            )

        with col_vp:
            st.markdown("**VCP Pattern**")
            vp = diag["vcp"]
            st.write(f"valid: {'✅' if vp['valid'] else '❌'}")
            st.write(f"score: {vp['vcp_score']:.2f}")
            st.write(f"reason: `{vp['reason']}`")
            if vp["contractions"]:
                contractions_str = " → ".join(f"{c:.1%}" for c in vp["contractions"])
                st.write(f"contractions: {contractions_str}")
            st.write(f"extended: {'⚠️ yes' if vp['extended'] else 'no'}")

        st.markdown("---")
        st.markdown("**K 線（最近 180 個交易日）**")
        try:
            end_str = d_date.isoformat()
            start_str = (pd.Timestamp(d_date) - pd.Timedelta(days=400)).date().isoformat()
            df = cached_load(d_ticker, start_str, end_str)
            if df.empty:
                st.warning("無資料")
            else:
                df_plot = df.iloc[-180:].copy()
                df_plot.columns = [c.capitalize() for c in df_plot.columns]
                addplots = []
                if len(df_plot) >= 50:
                    addplots.append(mpf.make_addplot(sma(df_plot["Close"], 50), color="blue"))
                if len(df_plot) >= 150:
                    addplots.append(mpf.make_addplot(sma(df_plot["Close"], 150), color="orange"))
                if len(df_plot) >= 200:
                    addplots.append(mpf.make_addplot(sma(df_plot["Close"], 200), color="red"))
                fig, _ = mpf.plot(
                    df_plot,
                    type="candle",
                    volume=True,
                    addplot=addplots if addplots else None,
                    style="yahoo",
                    figsize=(14, 7),
                    returnfig=True,
                    title=f"{d_ticker} — SMA50 (blue) / SMA150 (orange) / SMA200 (red)",
                )
                st.pyplot(fig)
                plt.close(fig)
        except Exception as e:
            st.error(f"K 線繪圖失敗：{e}")

        with st.expander("完整 JSON 診斷"):
            st.json(diag)


with tab_backtest:
    st.subheader("簡易 backtest")
    c1, c2 = st.columns(2)
    bt_start = c1.date_input(
        "起始日", pd.Timestamp("2022-01-01").date(), key="bt_start"
    )
    bt_end = c2.date_input("結束日", pd.Timestamp("2024-12-31").date(), key="bt_end")

    c3, c4, c5, c6 = st.columns(4)
    bt_capital = c3.number_input(
        "初始資金", min_value=100_000, max_value=10_000_000, value=1_000_000, step=100_000
    )
    bt_max_pos = c4.number_input("最大同時持倉", min_value=1, max_value=20, value=5)
    bt_min_vcp = c5.slider("Min VCP score", 0.0, 1.0, 0.6, 0.05, key="bt_min_vcp")
    bt_hold = c6.number_input("Time stop（bar）", min_value=10, max_value=250, value=60)

    c7, c8 = st.columns(2)
    bt_stop = c7.slider("停損 %", 0.03, 0.15, 0.07, 0.005)
    bt_target = c8.slider("停利 %", 0.05, 0.50, 0.20, 0.01)

    if st.button("Run backtest", type="primary", key="run_bt"):
        if bt_end <= bt_start:
            st.error("結束日必須晚於起始日")
        else:
            equity, trades = cached_backtest(
                bt_start.isoformat(),
                bt_end.isoformat(),
                float(bt_capital),
                int(bt_max_pos),
                float(bt_stop),
                float(bt_target),
                int(bt_hold),
                float(bt_min_vcp),
            )
            st.session_state["bt_equity"] = equity
            st.session_state["bt_trades"] = trades

    equity = st.session_state.get("bt_equity")
    trades = st.session_state.get("bt_trades")
    if equity is not None and trades is not None:
        s = summary(equity, trades)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CAGR", f'{s["cagr"]:.2%}')
        m2.metric("Max DD", f'{s["max_drawdown"]:.2%}')
        m3.metric("Sharpe", f'{s["sharpe"]:.2f}')
        m4.metric("Final Equity", f'{s["final_equity"]:,.0f}')

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("交易筆數", s["n_trades"])
        m6.metric("Win Rate", f'{s["win_rate"]:.2%}')
        m7.metric("Expectancy", f'{s["expectancy"]:.2%}')
        pf = s["profit_factor"]
        m8.metric("Profit Factor", "∞" if pf == float("inf") else f"{pf:.2f}")

        st.markdown("**Equity curve**")
        fig, ax = plt.subplots(figsize=(14, 4))
        equity.plot(ax=ax)
        ax.grid(True)
        ax.set_ylabel("equity")
        st.pyplot(fig)
        plt.close(fig)

        if not trades.empty:
            st.markdown("**Trade log**")
            st.dataframe(trades, use_container_width=True, hide_index=True)
            st.download_button(
                "下載 trades CSV",
                trades.to_csv(index=False).encode("utf-8-sig"),
                file_name="trades.csv",
                mime="text/csv",
            )

            st.markdown("**單筆報酬分布**")
            fig, ax = plt.subplots(figsize=(10, 4))
            trades["return_pct"].hist(bins=30, ax=ax)
            ax.set_xlabel("return %")
            ax.set_ylabel("trade count")
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)
