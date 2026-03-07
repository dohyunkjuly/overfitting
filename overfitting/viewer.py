from __future__ import annotations

import os, sys, tempfile
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from overfitting import Strategy
from overfitting.data import MultiCurrency

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QAbstractItemView, QPushButton, QButtonGroup,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import QColor

_BG        = "#0a0a0a"
_BG2       = "#0d0d0d"
_BG3       = "#161616"
_BORDER    = "#222222"
_AMBER     = "#ff9900"
_AMBER_DIM = "#995c00"
_AMBER_BG  = "#1a1000"
_WHITE     = "#d8d8d8"
_DIM       = "#555555"
_GREEN     = "#00cc66"
_RED       = "#ff3333"
_CYAN      = "#00bcd4"
_FONT      = "Courier New, Courier, monospace"

_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
_TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


class StrategyAdapter:
    """Extracts symbol metadata from a Strategy instance."""

    def __init__(self, strategy):
        self._strategy = strategy

    def get_symbols(self) -> list[str]:
        if isinstance(self._strategy.data, MultiCurrency):
            return list(self._strategy.data.symbols)
        return ["DEFAULT"]


class OHLCBuilder:
    """Builds and resamples OHLC DataFrames from strategy data."""

    @staticmethod
    def detect_base_tf_minutes(price_df: pd.DataFrame) -> int:
        """Infer the bar interval of the DataFrame in minutes."""
        if len(price_df) < 2:
            return 60
        diffs = price_df["timestamp"].diff().dropna()
        median_sec = diffs.median().total_seconds()
        return max(1, int(round(median_sec / 60)))

    @staticmethod
    def resample(price_df: pd.DataFrame, target_minutes: int, base_minutes: int) -> pd.DataFrame:
        """Resample a tidy OHLC DataFrame to a coarser timeframe."""
        if price_df is None or price_df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])

        if target_minutes <= base_minutes:
            return price_df.copy().reset_index(drop=True)

        df = price_df.set_index("timestamp")
        rs = df.resample(f"{target_minutes}min", label="left", closed="left").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        ).dropna(subset=["open"])
        return rs.reset_index()

    @classmethod
    def from_strategy(cls, strategy, symbol: str) -> pd.DataFrame:
        """Extract a tidy OHLC DataFrame for *symbol* from a strategy instance."""
        try:
            d = strategy.data[symbol] if isinstance(strategy.data, MultiCurrency) else strategy.data
        except KeyError:
            d = strategy.data

        return pd.DataFrame({
            "timestamp": pd.to_datetime(d.timestamp),
            "open":  d.open,
            "high":  d.high,
            "low":   d.low,
            "close": d.close,
        }).sort_values("timestamp").reset_index(drop=True)


class TradeBuilder:
    """Builds trade and marker DataFrames from a strategy's broker trade log."""

    COLUMNS = [
        "id", "created_at", "executed_at", "symbol", "side",
        "qty", "price", "theoretical_price", "executed_price", "type",
        "status", "stop_price", "is_triggered", "commission",
        "pnl", "realized_pnl", "label", "reason",
    ]

    @classmethod
    def from_strategy(cls, strategy, symbol: str | None = None) -> pd.DataFrame:
        """Return the full trade log as a normalised DataFrame."""
        raw = strategy.broker.trades
        if not raw:
            return pd.DataFrame(columns=cls.COLUMNS)

        df = pd.DataFrame(raw)
        for col in ("created_at", "executed_at"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        if isinstance(strategy.data, MultiCurrency) and symbol and "symbol" in df.columns:
            df = df[df["symbol"] == symbol]

        for c in cls.COLUMNS:
            if c not in df.columns:
                df[c] = None

        return (
            df.sort_values(["executed_at", "created_at"], ascending=True, na_position="last")
              .reset_index(drop=True)
        )

    @staticmethod
    def to_markers(trades_df: pd.DataFrame) -> pd.DataFrame:
        """Convert a filled-trades DataFrame into chart marker rows."""
        empty = pd.DataFrame(columns=["x", "y", "side", "text"])
        if trades_df.empty:
            return empty

        filled = trades_df[
            (trades_df["status"] == "FILLED") &
            trades_df["executed_at"].notna() &
            trades_df["executed_price"].notna()
        ].copy()

        if filled.empty:
            return empty

        def _fmt(row) -> str:
            color = "#00e5ff" if row.get("side") == "LONG" else "#ff00cc"
            pnl = row.get("realized_pnl", "")
            pnl_s = f"{float(pnl):+.4f}" if pnl not in ("", None) else "—"
            ts = pd.to_datetime(row.get("executed_at", ""), errors="coerce")
            ts_s = ts.strftime("%Y-%m-%d %H:%M") if not pd.isna(ts) else "—"
            return (
                f"<span style='color:{color};font-weight:bold'>{row.get('side','')}</span>"
                f"  {row.get('type','')}  "
                f"<span style='color:#888'>{ts_s}</span><br>"
                f"qty: {row.get('qty','')}<br>"
                f"price: {row.get('executed_price','')}<br>"
                f"pnl: {pnl_s}<br>"
                f"label: {row.get('label','') or '—'}"
            )

        return pd.DataFrame({
            "x": filled["executed_at"].values,
            "y": filled["executed_price"].values,
            "side": filled["side"].values,
            "executed_at": filled["executed_at"].values,
            "text": filled.apply(_fmt, axis=1).values,
        })


class CandleChartWidget(QWidget):
    # Fixed chart window size
    _MAX_CANDLES = 20_000
    _HALF_WINDOW = _MAX_CANDLES // 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._price_df = None      # base (finest) OHLC
        self._markers_df = None    # all filled trade markers
        self._title = ""
        self._tmp_file = None
        self._base_tf_min = 60
        self._current_tf = "1h"

        self._view = QWebEngineView(self)
        self._view.setStyleSheet(
            f"QWebEngineView {{ background:{_BG}; border:none; margin:0; padding:0; }}"
        )
        self._view.page().setBackgroundColor(QColor(_BG))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._view)

    def set_data(self, price_df, markers_df=None, title="", base_tf_min: int | None = None):
        self._price_df = price_df.copy() if price_df is not None else None
        self._markers_df = markers_df.copy() if markers_df is not None else None
        self._title = title
        if self._price_df is not None:
            self._base_tf_min = base_tf_min or OHLCBuilder.detect_base_tf_minutes(self._price_df)
        self._render()

    def set_timeframe(self, tf: str):
        self._current_tf = tf
        self._render()

    def zoom_to_trade(self, executed_at):
        if self._price_df is None or self._price_df.empty:
            return
        ts = pd.to_datetime(executed_at, errors="coerce")
        if pd.isna(ts):
            return
        self._render(center_time=ts, selected_time=ts)

    def _get_display_df(self, center_time=None) -> pd.DataFrame:
        """
        Return chart OHLC for the current timeframe.

        - Initial render: first 2000 base rows
        - Trade click: base rows around selected trade timestamp, approx n-1000:n+1000
        """
        if self._price_df is None or self._price_df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])

        base_df = self._price_df.reset_index(drop=True)

        if center_time is None:
            src = base_df.iloc[:self._MAX_CANDLES].copy()
        else:
            ts = pd.to_datetime(center_time, errors="coerce")
            if pd.isna(ts):
                src = base_df.iloc[:self._MAX_CANDLES].copy()
            else:
                idx = (base_df["timestamp"] - ts).abs().idxmin()
                left = max(0, idx - self._HALF_WINDOW)
                right = min(len(base_df), idx + self._HALF_WINDOW)
                src = base_df.iloc[left:right].copy()

        tf_min = _TF_MINUTES.get(self._current_tf, self._base_tf_min)
        df = OHLCBuilder.resample(src, tf_min, self._base_tf_min).reset_index(drop=True)

        return df

    def _y_range_for_window(self, df: pd.DataFrame):
        if df is None or df.empty:
            return None
        lo = float(df["low"].min())
        hi = float(df["high"].max())
        pad = (hi - lo) * 0.07 if hi > lo else max(abs(hi) * 0.01, 1.0)
        return [lo - pad, hi + pad]

    def _render(self, center_time=None, selected_time=None):
        if self._price_df is None or self._price_df.empty:
            return

        display_df = self._get_display_df(center_time=center_time)
        if display_df.empty:
            return

        display_x_start = display_df["timestamp"].min()
        display_x_end = display_df["timestamp"].max()

        fig = go.Figure()

        # ── candlesticks ──────────────────────────────────────────────────────
        fig.add_trace(go.Candlestick(
            x=display_df["timestamp"],
            open=display_df["open"], high=display_df["high"],
            low=display_df["low"], close=display_df["close"],
            name="OHLC",
            increasing=dict(line=dict(color=_GREEN, width=1), fillcolor=_GREEN),
            decreasing=dict(line=dict(color=_RED, width=1), fillcolor=_RED),
            hoverlabel=dict(bgcolor=_BG3),
            hovertemplate=(
                "<span style='color:#888'>%{x|%Y-%m-%d %H:%M}</span><br>"
                "O: %{open:.2f}  H: %{high:.2f}<br>"
                "L: %{low:.2f}  C: %{close:.2f}"
                "<extra></extra>"
            ),
        ))

        # ── markers only inside visible candle window ────────────────────────
        mdf = pd.DataFrame()
        if self._markers_df is not None and not self._markers_df.empty:
            mdf = self._markers_df.copy()
            mdf["x"] = pd.to_datetime(mdf["x"], errors="coerce")
            mdf = mdf.dropna(subset=["x"]).sort_values("x").reset_index(drop=True)

            mdf = mdf[
                (mdf["x"] >= display_x_start) &
                (mdf["x"] <= display_x_end)
            ].reset_index(drop=True)

            for _, row in mdf.iterrows():
                side = row["side"]
                color = "#00e5ff" if side == "LONG" else "#ff00cc"
                sym = "triangle-up" if side == "LONG" else "triangle-down"

                y_val = pd.to_numeric(pd.Series([row["y"]]), errors="coerce").iloc[0]
                if pd.isna(y_val):
                    continue

                nudge = y_val * (0.0012 if side == "LONG" else -0.0012)

                fig.add_trace(go.Scatter(
                    x=[row["x"]],
                    y=[y_val - nudge],
                    mode="markers",
                    name=side,
                    text=[row["text"]],
                    hovertemplate="%{text}<extra></extra>",
                    marker=dict(
                        symbol=sym,
                        size=10,
                        color=color,
                        line=dict(width=1, color="rgba(255,255,255,0.12)"),
                    ),
                    showlegend=False,
                ))

        # ── highlight selected trade if visible ──────────────────────────────
        if selected_time is not None and not mdf.empty:
            ts_series = pd.to_datetime(mdf["x"], errors="coerce")
            if ts_series.notna().any():
                nearest = (ts_series - pd.to_datetime(selected_time)).abs().idxmin()
                row = mdf.loc[[nearest]]

                highlight_y = pd.to_numeric(row["y"], errors="coerce")
                if not highlight_y.empty and pd.notna(highlight_y.iloc[0]):
                    fig.add_trace(go.Scatter(
                        x=row["x"],
                        y=highlight_y,
                        mode="markers",
                        hoverinfo="skip",
                        marker=dict(
                            symbol="circle-open",
                            size=22,
                            color=_AMBER,
                            line=dict(width=2, color=_AMBER),
                        ),
                        showlegend=False,
                        name="selected-trade",
                    ))


        y_axis_cfg: dict = dict(
            title="",
            gridcolor=_BORDER,
            linecolor=_BORDER,
            tickfont=dict(color=_AMBER_DIM, size=10, family=_FONT),
            tickformat=",.0f",
            side="right",
            zeroline=False,
            fixedrange=False,
            autorange=True,
        )

        x_axis_cfg: dict = dict(
            title="",
            rangeslider_visible=False,
            gridcolor=_BORDER,
            linecolor=_BORDER,
            tickfont=dict(color=_AMBER_DIM, size=10, family=_FONT),
            zeroline=False,
            fixedrange=False,
            range=[str(display_x_start), str(display_x_end)],
        )

        y_rng = self._y_range_for_window(display_df)
        if y_rng:
            y_axis_cfg["range"] = y_rng
            y_axis_cfg["autorange"] = False

        fig.update_layout(
            title=dict(
                text=f"<b>{self._title}</b>  "
                     f"<span style='color:{_AMBER_DIM};font-size:11px'>{self._current_tf.upper()}</span>",
                font=dict(color=_AMBER, size=13, family=_FONT),
                x=0.005, xanchor="left", y=0.98, yanchor="top",
            ),
            paper_bgcolor=_BG,
            plot_bgcolor=_BG2,
            font=dict(color=_AMBER_DIM, family=_FONT, size=11),
            xaxis=x_axis_cfg,
            yaxis=y_axis_cfg,
            legend=dict(
                orientation="h",
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=_DIM, size=10, family=_FONT),
                x=0, y=1.0,
            ),
            margin=dict(l=0, r=56, t=32, b=0),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor=_BG3,
                bordercolor=_AMBER_DIM,
                font=dict(color=_AMBER, family=_FONT, size=11),
            ),
            dragmode="pan",
            uirevision="constant",
        )

        extra_head = (
            "<style>"
            "html,body{margin:0;padding:0;background:" + _BG + ";overflow:hidden}"
            ".js-plotly-plot,.plot-container{width:100%!important;height:100vh!important}"
            "body{user-select:none;}"
            "</style>"
        )

        html = fig.to_html(
            include_plotlyjs=True,
            full_html=True,
            config={
                "scrollZoom": True,
                "displayModeBar": False,
                "doubleClick": "reset",
            },
        )

        custom_js = r"""
        <script>
        document.addEventListener("DOMContentLoaded", function () {
            const gd = document.querySelector('.js-plotly-plot');
            if (!gd) return;

            let isDraggingYAxis = false;
            let startY = 0;
            let startRange = null;

            function getYAxisRange() {
                const layout = gd._fullLayout;
                if (!layout || !layout.yaxis) return null;
                const r = layout.yaxis.range;
                if (!r || r.length !== 2) return null;
                return [Number(r[0]), Number(r[1])];
            }

            function isNearRightAxis(ev) {
                const rect = gd.getBoundingClientRect();
                const fullLayout = gd._fullLayout;
                if (!fullLayout || !fullLayout.yaxis) return false;

                // Plotly's plotting area ends before the right margin.
                // We use the right side of the graph div as the y-axis interaction zone.
                const zoneWidth = 60;  // pixels near the right edge
                const x = ev.clientX - rect.left;

                return x >= (rect.width - zoneWidth) && x <= rect.width;
            }

            function scaleRangeFromDrag(startRange, dy, plotHeight) {
                const y0 = startRange[0];
                const y1 = startRange[1];
                const center = (y0 + y1) / 2;
                const span = Math.abs(y1 - y0);

                if (!isFinite(span) || span === 0 || !isFinite(plotHeight) || plotHeight <= 0) {
                    return startRange;
                }

                // Drag up => compress range, drag down => expand range
                const sensitivity = 1.25;
                const factor = Math.exp((dy / plotHeight) * sensitivity);

                const newSpan = span * factor;
                return [center - newSpan / 2, center + newSpan / 2];
            }

            gd.addEventListener('mousedown', function (ev) {
                if (ev.button !== 0) return;
                if (!isNearRightAxis(ev)) return;

                const range = getYAxisRange();
                if (!range) return;

                isDraggingYAxis = true;
                startY = ev.clientY;
                startRange = range;

                ev.preventDefault();
                ev.stopPropagation();
            }, true);

            window.addEventListener('mousemove', function (ev) {
                if (!isDraggingYAxis || !startRange) return;

                const fullLayout = gd._fullLayout;
                const plotHeight = fullLayout && fullLayout._size ? fullLayout._size.h : gd.clientHeight;
                const dy = ev.clientY - startY;

                const newRange = scaleRangeFromDrag(startRange, dy, plotHeight);

                Plotly.relayout(gd, {
                    'yaxis.autorange': false,
                    'yaxis.range': newRange
                });

                ev.preventDefault();
            }, true);

            window.addEventListener('mouseup', function () {
                isDraggingYAxis = false;
                startRange = null;
            }, true);

            // Cursor feedback when hovering near right axis
            gd.addEventListener('mousemove', function (ev) {
                if (isDraggingYAxis) {
                    gd.style.cursor = 'ns-resize';
                    return;
                }
                gd.style.cursor = isNearRightAxis(ev) ? 'ns-resize' : '';
            });

            gd.addEventListener('mouseleave', function () {
                if (!isDraggingYAxis) {
                    gd.style.cursor = '';
                }
            });
        });
        </script>
        """

        html = html.replace("<head>", "<head>" + extra_head, 1)
        html = html.replace("</body>", custom_js + "</body>", 1)

        if not self._tmp_file:
            fd, path = tempfile.mkstemp(suffix=".html")
            os.close(fd)
            self._tmp_file = path

        with open(self._tmp_file, "w", encoding="utf-8") as f:
            f.write(html)

        self._view.load(QUrl.fromLocalFile(self._tmp_file))


class TimeframeBar(QFrame):
    """Row of TF buttons; disables those finer than the base data interval."""

    tf_changed = Signal(str)

    _BTN_STYLE = """
        QPushButton {{
            background: {bg};
            color: {fg};
            border: 1px solid {border};
            border-radius: 2px;
            padding: 2px 10px;
            font-family: {font};
            font-size: 11px;
            font-weight: bold;
            min-width: 32px;
        }}
        QPushButton:hover  {{ background: {hover};  color: {amber}; border-color: {amber}; }}
        QPushButton:checked {{ background: {amber_bg}; color: {amber}; border-color: {amber}; }}
        QPushButton:disabled {{ color: {dim}; border-color: {border}; }}
    """.format(
        bg=_BG3, fg=_DIM, border=_BORDER,
        hover=_BG3, amber=_AMBER, amber_bg=_AMBER_BG,
        dim=_DIM, font=_FONT,
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.setStyleSheet(f"QFrame {{ background:{_BG}; border:none; }}")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(4)

        tf_lbl = QLabel("TF:")
        tf_lbl.setStyleSheet(
            f"color:{_DIM}; font-family:{_FONT}; font-size:10px; background:transparent;"
        )
        lay.addWidget(tf_lbl)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}

        for tf in _TIMEFRAMES:
            btn = QPushButton(tf.upper())
            btn.setCheckable(True)
            btn.setStyleSheet(self._BTN_STYLE)
            btn.setFocusPolicy(Qt.NoFocus)
            lay.addWidget(btn)
            self._group.addButton(btn)
            self._buttons[tf] = btn

        lay.addStretch()
        self._group.buttonClicked.connect(self._on_click)

    def set_base_tf(self, base_minutes: int, active_tf: str):
        """Enable only TFs that are >= the base data interval."""
        for tf, btn in self._buttons.items():
            tf_min = _TF_MINUTES[tf]
            enabled = tf_min >= base_minutes
            btn.setEnabled(enabled)
            if tf == active_tf and enabled:
                btn.setChecked(True)
            elif not enabled:
                btn.setChecked(False)

    def _on_click(self, btn: QPushButton):
        for tf, b in self._buttons.items():
            if b is btn:
                self.tf_changed.emit(tf)
                return


class TradeTableWidget(QWidget):
    trade_selected = Signal(dict)

    COLUMNS = [
        ("executed_at",       "EXECUTED AT"),
        ("created_at",        "CREATED AT"),
        ("symbol",            "SYMBOL"),
        ("side",              "SIDE"),
        ("qty",               "QTY"),
        ("price",             "ORDER PRICE"),
        ("theoretical_price", "THEOR. PRICE"),
        ("executed_price",    "EXEC PRICE"),
        ("type",              "TYPE"),
        ("status",            "STATUS"),
        ("stop_price",        "STOP PRICE"),
        ("is_triggered",      "TRIGGERED"),
        ("commission",        "COMMISSION"),
        ("pnl",               "PNL"),
        ("realized_pnl",      "REAL. PNL"),
        ("label",             "LABEL"),
        ("reason",            "REASON"),
        ("id",                "ORDER ID"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df = pd.DataFrame()
        self._keys = [k for k, _ in self.COLUMNS]

        self._table = QTableWidget(0, len(self._keys), self)
        self._table.setHorizontalHeaderLabels([l for _, l in self.COLUMNS])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSortingEnabled(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.cellClicked.connect(self._on_click)
        self._apply_style()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._table)

    def set_data(self, df: pd.DataFrame):
        self._df = df.reset_index(drop=True) if df is not None and not df.empty else pd.DataFrame()
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)

        if self._df.empty:
            return

        self._table.setRowCount(len(self._df))
        for r, (_, row) in enumerate(self._df.iterrows()):
            for c, key in enumerate(self._keys):
                raw = row.get(key, "")
                if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                    val = ""
                elif isinstance(raw, float):
                    val = f"{raw:.6f}".rstrip("0").rstrip(".")
                else:
                    val = str(raw)

                item = QTableWidgetItem(val)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setData(Qt.UserRole, r)

                if key in ("realized_pnl", "pnl") and val:
                    try:
                        item.setForeground(QColor(_GREEN if float(val) >= 0 else _RED))
                    except ValueError:
                        pass
                elif key == "side":
                    item.setForeground(QColor("#00e5ff" if val == "LONG" else "#ff00cc"))
                elif key == "status":
                    item.setForeground(QColor({
                        "FILLED": _GREEN,
                        "CANCELLED": _DIM,
                        "REJECTED": _RED
                    }.get(val, _WHITE)))
                elif key == "type":
                    item.setForeground(QColor(_CYAN))
                elif key == "id":
                    item.setForeground(QColor(_DIM))

                self._table.setItem(r, c, item)

        self._table.setSortingEnabled(True)
        self._table.sortByColumn(0, Qt.AscendingOrder)

    def _on_click(self, row: int, _col: int):
        if self._df.empty:
            return

        item = self._table.item(row, 0)
        if item is None:
            return

        original_idx = item.data(Qt.UserRole)
        self.trade_selected.emit(self._df.iloc[original_idx].to_dict())

    def _apply_style(self):
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {_BG2};
                alternate-background-color: {_BG3};
                color: {_WHITE};
                gridline-color: {_BORDER};
                border: none;
                font-family: {_FONT};
                font-size: 11px;
            }}
            QHeaderView::section {{
                background-color: {_BG};
                color: {_AMBER};
                border: none;
                border-right: 1px solid {_BORDER};
                border-bottom: 2px solid {_AMBER_DIM};
                padding: 3px 6px;
                font-family: {_FONT};
                font-size: 10px;
                font-weight: bold;
            }}
            QTableWidget::item {{ padding: 2px 6px; }}
            QTableWidget::item:selected {{ background-color:{_AMBER_BG}; color:{_AMBER}; }}
            QScrollBar:horizontal, QScrollBar:vertical {{
                background:{_BG}; height:5px; width:5px;
            }}
            QScrollBar::handle:horizontal, QScrollBar::handle:vertical {{
                background:{_AMBER_DIM}; border-radius:2px;
            }}
            QScrollBar::add-line, QScrollBar::sub-line {{ background:none; border:none; }}
        """)


class HeaderBar(QFrame):
    def __init__(self, symbols: list[str], parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setStyleSheet(
            f"QFrame {{ background:{_BG}; border-bottom:2px solid {_AMBER}; }}"
        )

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 12, 0)

        title = QLabel("OVERFITTING  //  BACKTEST VIEWER")
        title.setStyleSheet(
            f"color:{_AMBER}; font-family:{_FONT}; font-size:12px; "
            f"font-weight:bold; background:transparent;"
        )
        lay.addWidget(title)
        lay.addStretch()

        sym_lbl = QLabel("SYMBOL:")
        sym_lbl.setStyleSheet(
            f"color:{_DIM}; font-family:{_FONT}; font-size:10px; background:transparent;"
        )
        lay.addWidget(sym_lbl)

        self.combo = QComboBox()
        self.combo.addItems(symbols)
        self.combo.setFixedWidth(150)
        self.combo.setStyleSheet(f"""
            QComboBox {{
                background:{_BG3}; color:{_AMBER};
                border:1px solid {_AMBER_DIM}; border-radius:2px;
                padding:2px 24px 2px 8px;
                font-family:{_FONT}; font-size:11px;
            }}
            QComboBox::drop-down {{ border:none; width:20px; }}
            QComboBox QAbstractItemView {{
                background:{_BG3}; color:{_AMBER};
                selection-background-color:{_BG};
                border:1px solid {_AMBER_DIM};
                font-family:{_FONT};
            }}
        """)
        lay.addWidget(self.combo)


class BacktestViewerWindow(QMainWindow):
    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy
        self.symbols = StrategyAdapter(strategy).get_symbols()
        self.current_symbol = self.symbols[0]

        self.setWindowTitle("OVERFITTING — BACKTEST VIEWER")
        self.resize(1600, 960)
        self.setStyleSheet(
            f"QMainWindow, QWidget {{ background:{_BG}; color:{_WHITE}; }}"
        )

        self.header = HeaderBar(self.symbols, self)
        self.tf_bar = TimeframeBar(self)
        self.chart = CandleChartWidget(self)
        self.trade_table = TradeTableWidget(self)

        self.header.combo.setCurrentText(self.current_symbol)
        self.header.combo.currentTextChanged.connect(self._on_symbol_changed)
        self.trade_table.trade_selected.connect(self._on_trade_selected)
        self.tf_bar.tf_changed.connect(self._on_tf_changed)

        self._build_ui()
        self._refresh()

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self.header)
        root.addWidget(self.tf_bar)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.chart)
        splitter.addWidget(self.trade_table)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setHandleWidth(3)
        splitter.setStyleSheet(
            f"QSplitter::handle:vertical {{ background:{_AMBER_DIM}; }}"
        )
        root.addWidget(splitter, 1)

    def _on_symbol_changed(self, symbol: str):
        self.current_symbol = symbol
        self._refresh()

    def _on_trade_selected(self, trade: dict):
        at = trade.get("executed_at")
        if at and str(at) not in ("", "None", "NaT"):
            self.chart.zoom_to_trade(at)

    def _on_tf_changed(self, tf: str):
        self.chart.set_timeframe(tf)

    def _refresh(self):
        price_df = OHLCBuilder.from_strategy(self.strategy, self.current_symbol)
        trades_df = TradeBuilder.from_strategy(self.strategy, self.current_symbol)
        markers = TradeBuilder.to_markers(trades_df)

        base_tf_min = OHLCBuilder.detect_base_tf_minutes(price_df) if not price_df.empty else 60

        default_tf = "1h"
        for tf in _TIMEFRAMES:
            if _TF_MINUTES[tf] >= base_tf_min:
                default_tf = tf
                break

        self.chart._current_tf = default_tf
        self.tf_bar.set_base_tf(base_tf_min, default_tf)
        self.chart.set_data(price_df, markers, title=self.current_symbol, base_tf_min=base_tf_min)

        # Trade table stays full
        self.trade_table.set_data(trades_df)


class BacktestViewer:
    """
    BacktestViewer(strategy).show()
    """
    def __init__(self, strategy):
        if not hasattr(strategy, "broker"):
            raise ValueError("strategy must have a .broker attribute")
        if not hasattr(strategy, "data"):
            raise ValueError("strategy must have a .data attribute")
        self._strategy = strategy
        self._window = None

    def show(self):
        app = QApplication.instance()
        owns_app = app is None
        if owns_app:
            app = QApplication(sys.argv)

        self._window = BacktestViewerWindow(self._strategy)
        self._window.show()

        if owns_app:
            sys.exit(app.exec())