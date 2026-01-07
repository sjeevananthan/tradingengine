import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class Trade:
    trade_id: int
    entry_price: float
    direction: str  # 'long' or 'short'
    stop_loss: float
    target: float
    size: float
    open_time: pd.Timestamp
    is_open: bool = True
    close_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

class VolumeProfile:
    def __init__(self, n_bins: int = 100):
        self.n_bins = n_bins

    def compute(self, df: pd.DataFrame) -> Dict[str, Any]:
        price_min = df['low'].min()
        price_max = df['high'].max()

        bins = np.linspace(price_min, price_max, self.n_bins)
        volume_profile = np.zeros(len(bins))

        for _, row in df.iterrows():
            idx = np.searchsorted(bins, row['close']) - 1
            if 0 <= idx < len(volume_profile):
                volume_profile[idx] += row['volume']

        poc_idx = np.argmax(volume_profile)
        poc = bins[poc_idx]

        total_volume = volume_profile.sum()
        sorted_idx = np.argsort(volume_profile)[::-1]

        cum_vol = 0.0
        value_area_idx = []
        for idx in sorted_idx:
            cum_vol += volume_profile[idx]
            value_area_idx.append(idx)
            if cum_vol / total_volume >= 0.70:
                break

        vah = bins[max(value_area_idx)]
        val = bins[min(value_area_idx)]

        return {
            'POC': poc,
            'VAH': vah,
            'VAL': val
        }

class EntryEngine:
    def __init__(self, tolerance: float = 0.001):
        self.tolerance = tolerance

    def check_entry(self, price: float, vp: Dict[str, float]) -> Optional[str]:
        if abs(price - vp['POC']) / price < self.tolerance:
            return 'mean_reversion'
        if price > vp['VAH']:
            return 'breakout_long'
        if price < vp['VAL']:
            return 'breakout_short'
        return None

class TradeLogger:
    def __init__(self):
        self.bar_logs: List[Dict[str, Any]] = []
        self.trade_summaries: List[Dict[str, Any]] = []

    def log_bar(self,
                timestamp: pd.Timestamp,
                bar_index: int,
                price: float,
                trade: Optional[Trade],
                vp: Dict[str, float],
                features: Optional[Dict[str, float]] = None):

        if trade is None:
            return

        self.bar_logs.append({
            'timestamp': timestamp,
            'bar_index': bar_index,
            'trade_id': trade.trade_id,
            'price': price,
            'direction': trade.direction,
            'entry_price': trade.entry_price,
            'stop_loss': trade.stop_loss,
            'target': trade.target,
            'mfe': trade.max_favorable_excursion,
            'mae': trade.max_adverse_excursion,
            'time_in_trade': (timestamp - trade.open_time).total_seconds(),
            'POC': vp['POC'],
            'VAH': vp['VAH'],
            'VAL': vp['VAL'],
            **(features or {})
        })

    def log_trade_close(self, trade: Trade):
        self.trade_summaries.append({
            'trade_id': trade.trade_id,
            'direction': trade.direction,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'open_time': trade.open_time,
            'close_time': trade.close_time,
            'mfe': trade.max_favorable_excursion,
            'mae': trade.max_adverse_excursion,
            'pnl': (trade.exit_price - trade.entry_price)
                   if trade.direction == 'long'
                   else (trade.entry_price - trade.exit_price)
        })

    def bar_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.bar_logs)

    def trade_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_summaries)

class TradeManager:
    def __init__(self):
        self.active_trade: Optional[Trade] = None
        self.trade_counter = 0

    def open_trade(self, price: float, signal: str, timestamp: pd.Timestamp) -> Trade:
        self.trade_counter += 1

        if signal == 'mean_reversion':
            direction = 'long'
            stop = price * 0.995
            target = price * 1.005
        elif signal == 'breakout_long':
            direction = 'long'
            stop = price * 0.997
            target = price * 1.010
        else:  # breakout_short
            direction = 'short'
            stop = price * 1.003
            target = price * 0.990

        self.active_trade = Trade(
            trade_id=self.trade_counter,
            entry_price=price,
            direction=direction,
            stop_loss=stop,
            target=target,
            size=1.0,
            open_time=timestamp
        )
        return self.active_trade

    def update_trade(self, price: float, timestamp: pd.Timestamp) -> Optional[Trade]:
        trade = self.active_trade
        if trade is None:
            return None

        pnl = price - trade.entry_price if trade.direction == 'long' else trade.entry_price - price
        trade.max_favorable_excursion = max(trade.max_favorable_excursion, pnl)
        trade.max_adverse_excursion = min(trade.max_adverse_excursion, pnl)

        hit_exit = (
            price <= trade.stop_loss or price >= trade.target
            if trade.direction == 'long'
            else price >= trade.stop_loss or price <= trade.target
        )

        if hit_exit:
            trade.is_open = False
            trade.close_time = timestamp
            trade.exit_price = price
            self.active_trade = None
            return trade

        return None

class TradingEngine:
    def __init__(self):
        self.vp = VolumeProfile()
        self.entry_engine = EntryEngine()
        self.trade_manager = TradeManager()
        self.logger = TradeLogger()

    def on_new_bar(self, df_slice: pd.DataFrame, bar_index: int):
        vp = self.vp.compute(df_slice)
        last_bar = df_slice.iloc[-1]
        price = last_bar['close']
        timestamp = last_bar.name

        if self.trade_manager.active_trade is None:
            signal = self.entry_engine.check_entry(price, vp)
            if signal:
                self.trade_manager.open_trade(price, signal, timestamp)

        closed_trade = self.trade_manager.update_trade(price, timestamp)

        self.logger.log_bar(
            timestamp=timestamp,
            bar_index=bar_index,
            price=price,
            trade=self.trade_manager.active_trade or closed_trade,
            vp=vp,
            features=None  # placeholder for confirmations (V2)
        )

        if closed_trade:
            self.logger.log_trade_close(closed_trade)
