
# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, informative
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa
# --------------------------------

# --- For conditions
from functools import reduce
# --- For hyperopt
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, BooleanParameter
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
# --- Other
from typing import Dict, List, Union, Optional

class SMACross_V1(IStrategy):

    """
    author@: @Arthie
    github@: https://github.com/droidoz/freqtrade/
    Other strategies for Freqtrade https://github.com/freqtrade/freqtrade
    
    This is a simple SMA crossover strategy framework to experiment with;
    Feel free to try other SMA timeperiod combo in Hyperoptable parameters.
    
    Note: always dry testing the strategy before going live ! Use at your own risk.
    """

    def version(self) -> str:
        return "v3"

    INTERFACE_VERSION = 3

    plot_config = {
        'main_plot': {
            "bb_lowerband": {
            "color": "#e66100",
            "type": "line",
            "fill_to": "bb_upperband"
            },
            "bb_upperband": {
            "color": "#F0E68C",
            "type": "line" 
            },
            'buy_MA1': {'color': '#8A2BE2', 'fill_to': 'buy_MA2'},
            'buy_MA2': {'color': '#5F9EA0', 'fill_to': 'buy_MA3'},
            },

        'subplots': {
            "MACD": {
            'macd': {'color': '#00FF00', 'fill_to': 'macdhist'},
            'macdhist': {'type': 'bar', 'color': '#FF00FF'},
            'macdsignal': {'color': '#00FFFF'},
            },

            'ADX': {
            'adx': {'color': '#FFF5EE'},    
            'minus_di': {'color': '#E9967A', 'fill_to': 'plus_di'},
            'plus_di' :{'color': '#FF4500'},
            },

            "RSI": {
                'rsi': {'type': 'bar', 'color': '#C71585'}
            },
        }
    }

    # Optional    
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 2,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]
        
    class HyperOpt:
        
        def generate_estimator(dimensions: List['Dimension'], **kwargs):
            from skopt.learning import ExtraTreesRegressor
            # Corresponds to "ET" - but allows additional parameters.
            return ExtraTreesRegressor(n_estimators=500) # Default = 100

        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.30, -0.10, decimals=2, name='stoploss')]

        # Define custom ROI space
        def roi_space() -> List[Dimension]:
            return [
                Integer(10, 120, name='roi_t1'),
                Integer(10, 60, name='roi_t2'),
                Integer(10, 40, name='roi_t3'),
                SKDecimal(0.01, 0.04, decimals=2, name='roi_p1'),
                SKDecimal(0.01, 0.07, decimals=2, name='roi_p2'),
                SKDecimal(0.01, 0.10, decimals=2, name='roi_p3'),
            ]

        def generate_roi_table(params: Dict) -> Dict[int, float]:

            roi_table = {}
            roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3']
            roi_table[params['roi_t3']] = params['roi_p1'] + params['roi_p2']
            roi_table[params['roi_t3'] + params['roi_t2']] = params['roi_p1']
            roi_table[params['roi_t3'] + params['roi_t2'] + params['roi_t1']] = 0

            return roi_table

        def trailing_space() -> List[Dimension]:
            # All parameters here are mandatory, you can only modify their type or the range.
            return [
                # Fixed to true, if optimizing trailing_stop we assume to use trailing stop at all times.
                Categorical([True], name='trailing_stop'),
                SKDecimal(0.01, 0.35, decimals=3, name='trailing_stop_positive'),
                SKDecimal(0.001, 0.1, decimals=3, name='trailing_stop_positive_offset_p1'),
                Categorical([True, False], name='trailing_only_offset_is_reached'),
            ]

    # Define protections parameters spaces
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # Optional
    # Protection hyperspace params:
    protection_params = {
        "cooldown_lookback": 5,
        "stop_duration": 12,
        "use_stop_protection": True,
    }
    
    # Hyperoptable parameters
    buy_adx = DecimalParameter(25, 50, decimals=1, default=30.1, space="buy", optimize=True)
    buy_rsi = IntParameter(20, 40, default=30, space="buy", optimize=True)
    buy_adx_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    buy_rsi_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
    buy_trigger = CategoricalParameter(["bb_lower", "macd_cross_signal"], default="bb_lower", space="buy", optimize=True)

    buy_MA1 = IntParameter(low=3, high=7, default=5, space='buy', optimize=True)
    buy_MA2 = IntParameter(low=8, high=25, default=8, space='buy', optimize=True)

    sell_trigger = CategoricalParameter(["bb_upper", "macd_cross_signal"], default="bb_upper", space="sell", optimize=True)

    # Buy hyperspace params:
    buy_params = {
        "buy_MA1": 7,
        "buy_MA2": 13,
        "buy_adx": 40.0,
        "buy_adx_enabled": False,
        "buy_rsi": 22,
        "buy_rsi_enabled": False,
        "buy_trigger": "bb_lower",
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_trigger": "bb_upper",
    }

    # ROI table:
    minimal_roi = {
        "0": 0.04,
        "20": 0.03,
        "30": 0.01,
        "120": 0
    }

    # Stoploss:
    stoploss = -0.13

    use_custom_stoploss = False

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = False

    # run "populate_indicators" only for new candle
    process_only_new_candles = True
    
    startup_candle_count = 600

    # Experimental settings (configuration will overide these if set)
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # BBANDS
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_upperband'] = bollinger['upperband']

        # SMA
        dataframe['buy_MA1'] = ta.SMA(dataframe, timeperiod=self.buy_MA1.value)
        dataframe['buy_MA2'] = ta.SMA(dataframe, timeperiod=self.buy_MA2.value)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Minus & Plus Directional Indicator / Movement
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        
        # Trend up
        conditions.append(dataframe['plus_di'] > dataframe['minus_di'])
        conditions.append(dataframe['buy_MA1'] > dataframe['buy_MA2']) 
        
        # GUARDS AND TRENDS
        if self.buy_adx_enabled.value:
            conditions.append(dataframe['adx'] > self.buy_adx.value)
        if self.buy_rsi_enabled.value:
            conditions.append(dataframe['rsi'] < self.buy_rsi.value)

        # TRIGGERS
        if self.buy_trigger.value == 'bb_lower':
            conditions.append(dataframe['close'] < dataframe['bb_lowerband'])
        if self.buy_trigger.value == 'macd_cross_signal':
            conditions.append(qtpylib.crossed_above(
                dataframe['macd'], dataframe['macdsignal']
            ))
       
        # Volume is above 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                ['enter_long', 'enter_tag']] = (1, 'buy_signal')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        
        # TRIGGERS
        if self.sell_trigger.value == 'bb_upper':
            conditions.append(dataframe['close'] > dataframe['bb_upperband'])
        if self.sell_trigger.value == 'macd_cross_signal':
            conditions.append(qtpylib.crossed_below(
                dataframe['macd'], dataframe['macdsignal']
            ))
            

        # Volume above 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                ['exit_long', 'exit_tag']] = (1, 'sell_signal')

        return dataframe
