# Vcp — 台股 VCP + CANSLIM-lite 最小實驗

實作 Mark Minervini 的 **VCP + Trend Template** 與 William O'Neil 的 **CANSLIM**（價量+市場面子集），套用在台股市場。第一版範圍是「最小實驗」：螢幕器 + 簡易回測 + 視覺化 notebook。

## 安裝

```bash
pip install -e ".[dev]"
```

需要 Python 3.10+。資料來源是 yfinance（免費，無需 API key），快取在 `data/cache/*.parquet`。

## 使用

### 螢幕器（單日）

```bash
# 找出 2024-12-31 通過所有條件的股票
python -m scripts.screen --date 2024-12-31

# 寫入 CSV
python -m scripts.screen --date 2024-12-31 --output out/screen.csv

# 嚴格模式：要求 L+S+N+M 全過
python -m scripts.screen --date 2024-12-31 --all-canslim

# 對單檔做完整診斷（看每個條件 pass/fail）
python -m scripts.screen --date 2024-04-30 --explain 6531.TW
```

### 回測

```bash
python -m scripts.backtest \
    --start 2020-01-01 \
    --end 2024-12-31 \
    --trades-out out/trades.csv \
    --equity-out out/equity.csv
```

### 視覺化

```bash
jupyter lab notebooks/01_explore.ipynb
```

## 設計決策

- **243 交易日**（不是 252）—— 台股實際年交易日
- **靜態 universe**（`data/universe.csv`，~130 檔）—— 第一版手動維護，不爬 TWSE
- **CANSLIM-lite** —— 跳過 C / A / I（需要財報），只實作 L / S / N / M
- **進場規則**：訊號隔日開盤，等權，最多 5 個部位
- **出場規則**：-7% 停損 / +20% 停利 / 60 日 time stop（先到先出）
- **交易成本**：往返 0.4%

完整設計理由見 `/root/.claude/plans/claude-code-fluttering-hopper.md`。

## 測試

```bash
pytest
```

## 模組結構

```
src/vcp/
├── data.py            # yfinance + parquet cache
├── indicators.py      # SMA, ATR, 52w high/low, RS rating（cross-sectional）
├── trend_template.py  # Minervini 8 criteria
├── vcp_pattern.py     # contraction 偵測 + score
├── canslim.py         # L/S/N/M
├── screener.py        # 組合所有檢查
├── backtest.py        # bar-by-bar 模擬
└── metrics.py         # CAGR, MaxDD, expectancy, profit factor
```

## 不在第一版內

留待 baseline 通過後再加：
- 每日 cron / Notion 推送
- 處置股、警示股自動排除
- 籌碼面（外資、投信、融資）
- 走步式參數最佳化
- 美股
- CANSLIM 的 C / A / I（需要 fundamentals）
- Trailing stop、加減碼
