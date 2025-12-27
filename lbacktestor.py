import polars as pl
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Trade:
    """单笔交易记录"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    side: str  # 'long' or 'short'
    quantity: float
    pnl: float
    pnl_pct: float
    duration_minutes: int

@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 100000  # 初始资金
    position_size_pct: float = 0.1   # 每次交易使用资金比例
    commission_rate: float = 0.001   # 手续费率
    slippage: float = 0.0001        # 滑点
    max_positions: int = 1           # 最大同时持仓数

class SignalBacktester:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: List[Trade] = []
        self.positions = []  # 当前持仓
        self.capital_history = []
        self.equity_curve = []
        
    def calculate_position_size(self, price: float, capital: float) -> float:
        """计算持仓数量"""
        position_value = capital * self.config.position_size_pct
        return position_value / price
    
    def apply_costs(self, price: float, side: str) -> float:
        """应用滑点和手续费"""
        slippage_cost = self.config.slippage if side == 'buy' else -self.config.slippage
        return price * (1 + slippage_cost)
    
    def backtest_signals(self, df_with_signals: pl.DataFrame) -> Dict:
        """
        执行回测
        df_with_signals: 包含交易信号的DataFrame
        """
        
        # 转换为pandas便于处理
        df = df_with_signals.to_pandas()
        
        current_capital = self.config.initial_capital
        current_positions = {}  # {side: {'entry_price': float, 'quantity': float, 'entry_time': datetime}}
        
        equity_history = []
        
        for idx, row in df.iterrows():
            current_time = row['end_tm']
            current_price = row['close']
            
            # 计算当前权益
            current_equity = current_capital
            for side, pos in current_positions.items():
                if side == 'long':
                    unrealized_pnl = (current_price - pos['entry_price']) * pos['quantity']
                else:  # short
                    unrealized_pnl = (pos['entry_price'] - current_price) * pos['quantity']
                
                # 减去手续费
                unrealized_pnl -= pos['entry_price'] * pos['quantity'] * self.config.commission_rate
                current_equity += unrealized_pnl
            
            equity_history.append({
                'end_tm': current_time,
                'equity': current_equity,
                'capital': current_capital
            })
            
            # 处理平仓信号
            if row.get('long_exit', False) and 'long' in current_positions:
                self._close_position('long', current_positions, current_price, current_time)
                current_capital = self._update_capital_after_trade()
                
            if row.get('short_exit', False) and 'short' in current_positions:
                self._close_position('short', current_positions, current_price, current_time)
                current_capital = self._update_capital_after_trade()
            
            # 处理开仓信号
            if row.get('long_entry', False) and 'long' not in current_positions:
                if len(current_positions) < self.config.max_positions:
                    self._open_position('long', current_positions, current_price, current_time, current_capital)
                    
            if row.get('short_entry', False) and 'short' not in current_positions:
                if len(current_positions) < self.config.max_positions:
                    self._open_position('short', current_positions, current_price, current_time, current_capital)
        
        # 平掉所有剩余持仓
        final_price = df.iloc[-1]['close']
        final_time = df.iloc[-1]['end_tm']
        
        for side in list(current_positions.keys()):
            self._close_position(side, current_positions, final_price, final_time)
            current_capital = self._update_capital_after_trade()
        
        self.equity_curve = equity_history
        return self._generate_performance_report()
    
    def _open_position(self, side: str, positions: dict, price: float, time: datetime, capital: float):
        """开仓"""
        entry_price = self.apply_costs(price, 'buy' if side == 'long' else 'sell')
        quantity = self.calculate_position_size(entry_price, capital)
        
        positions[side] = {
            'entry_price': entry_price,
            'quantity': quantity,
            'entry_time': time
        }
    
    def _close_position(self, side: str, positions: dict, price: float, time: datetime):
        """平仓"""
        if side not in positions:
            return
            
        pos = positions[side]
        exit_price = self.apply_costs(price, 'sell' if side == 'long' else 'buy')
        
        # 计算PnL
        if side == 'long':
            pnl = (exit_price - pos['entry_price']) * pos['quantity']
        else:  # short
            pnl = (pos['entry_price'] - exit_price) * pos['quantity']
        
        # 减去双边手续费
        total_commission = (pos['entry_price'] + exit_price) * pos['quantity'] * self.config.commission_rate
        pnl -= total_commission
        
        pnl_pct = pnl / (pos['entry_price'] * pos['quantity']) * 100
        
        duration_minutes = int((time - pos['entry_time']).total_seconds() / 60)
        
        trade = Trade(
            entry_time=pos['entry_time'],
            exit_time=time,
            entry_price=pos['entry_price'],
            exit_price=exit_price,
            side=side,
            quantity=pos['quantity'],
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_minutes=duration_minutes
        )
        
        self.trades.append(trade)
        del positions[side]
    
    def _update_capital_after_trade(self) -> float:
        """更新可用资金"""
        if self.trades:
            last_trade = self.trades[-1]
            return self.config.initial_capital + sum(t.pnl for t in self.trades)
        return self.config.initial_capital
    
    def _generate_performance_report(self) -> Dict:
        """生成绩效报告"""
        if not self.trades:
            return {"error": "No trades executed"}
        
        # 基础统计
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_return_pct = total_pnl / self.config.initial_capital * 100
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
        
        # 最大回撤
        equity_curve = pd.DataFrame(self.equity_curve)
        equity_curve['running_max'] = equity_curve['equity'].expanding().max()
        equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['running_max']) / equity_curve['running_max']
        max_drawdown = equity_curve['drawdown'].min() * 100
        
        # 夏普比率 (假设无风险利率为0)
        if len(self.equity_curve) > 1:
            returns = equity_curve['equity'].pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 6) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 交易持续时间统计
        avg_duration = np.mean([t.duration_minutes for t in self.trades])
        
        report = {
            'total_trades': total_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round(total_return_pct, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'avg_duration_minutes': round(avg_duration, 2),
            'final_capital': round(self.config.initial_capital + total_pnl, 2)
        }
        
        return report
    
    def get_trade_history(self) -> pl.DataFrame:
        """获取交易历史"""
        if not self.trades:
            return pl.DataFrame()
        
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'side': trade.side,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'duration_minutes': trade.duration_minutes
            })
        
        return pl.DataFrame(trade_data)
    
    def get_equity_curve(self) -> pl.DataFrame:
        """获取权益曲线"""
        return pl.DataFrame(self.equity_curve)


def run_backtest_example(df_with_signals: pl.DataFrame):
    """运行回测示例"""
    
    # 配置回测参数
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.1,  # 每次使用10%资金
        commission_rate=0.001,  # 0.1%手续费
        slippage=0.0001,       # 0.01%滑点
        max_positions=1        # 最多持有1个位置
    )
    
    # 创建回测器
    backtester = SignalBacktester(config)
    
    # 运行回测
    performance = backtester.backtest_signals(df_with_signals)
    
    # 打印结果
    print("=== 回测结果 ===")
    for key, value in performance.items():
        print(f"{key}: {value}")
    
    # 获取交易历史
    trade_history = backtester.get_trade_history()
    print(f"\n=== 交易明细 ===")
    print(trade_history)
    
    # 获取权益曲线
    equity_curve = backtester.get_equity_curve()
    
    return {
        'performance': performance,
        'trades': trade_history,
        'equity_curve': equity_curve,
        'backtester': backtester
    }

# 使用示例
def main():
    """主函数示例"""
    # 假设您已经有了带信号的DataFrame
    # df_with_signals = generate_trading_signals(your_1min_df)
    
    # 运行回测
    # results = run_backtest_example(df_with_signals)
    
    # 分析结果
    # print("胜率:", results['performance']['win_rate'], "%")
    # print("总收益率:", results['performance']['total_return_pct'], "%") 
    # print("最大回撤:", results['performance']['max_drawdown_pct'], "%")
    # print("盈亏比:", results['performance']['profit_factor'])
    
    pass

if __name__ == "__main__":
    main()