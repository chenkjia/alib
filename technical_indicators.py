#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
加密货币技术指标计算和信号生成模块

此文件包含加密货币技术指标计算、交易信号生成和简单回测功能
"""

# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

def calculate_technical_indicators(df):
    """
    计算常用技术指标
    
    参数:
        df: 包含价格数据的DataFrame
        
    返回:
        DataFrame: 添加了技术指标的DataFrame
    """
    # 创建副本以避免修改原始数据
    result = df.copy()
    
    # 按币种分组计算技术指标
    all_symbols = result['symbol'].unique()
    result_dfs = []
    
    for symbol in all_symbols:
        symbol_data = result[result['symbol'] == symbol].copy()
        
        # 确保按时间排序
        symbol_data = symbol_data.sort_values('datetime')
        
        # 1. 移动平均线
        symbol_data['MA5'] = symbol_data['$close'].rolling(window=5).mean()
        symbol_data['MA10'] = symbol_data['$close'].rolling(window=10).mean()
        symbol_data['MA20'] = symbol_data['$close'].rolling(window=20).mean()
        symbol_data['MA60'] = symbol_data['$close'].rolling(window=60).mean()
        
        # 2. 布林带 (20日均线, 2倍标准差)
        symbol_data['BB_middle'] = symbol_data['MA20']
        rolling_std = symbol_data['$close'].rolling(window=20).std()
        symbol_data['BB_upper'] = symbol_data['BB_middle'] + 2 * rolling_std
        symbol_data['BB_lower'] = symbol_data['BB_middle'] - 2 * rolling_std
        
        # 3. RSI (14天)
        delta = symbol_data['$close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        # 使用 .where() 处理除零问题
        rs = avg_gain / avg_loss.where(avg_loss != 0, 0.0001)
        symbol_data['RSI'] = 100 - (100 / (1 + rs))
        
        # 4. MACD
        ema12 = symbol_data['$close'].ewm(span=12, adjust=False).mean()
        ema26 = symbol_data['$close'].ewm(span=26, adjust=False).mean()
        symbol_data['MACD'] = ema12 - ema26
        symbol_data['MACD_signal'] = symbol_data['MACD'].ewm(span=9, adjust=False).mean()
        symbol_data['MACD_hist'] = symbol_data['MACD'] - symbol_data['MACD_signal']
        
        # 5. 收益率
        symbol_data['returns_1d'] = symbol_data['$close'].pct_change(1)
        symbol_data['returns_5d'] = symbol_data['$close'].pct_change(5)
        symbol_data['returns_10d'] = symbol_data['$close'].pct_change(10)
        
        # 6. 波动率（20日标准差）
        symbol_data['volatility_20d'] = symbol_data['returns_1d'].rolling(window=20).std()
        
        # 添加到结果列表
        result_dfs.append(symbol_data)
    
    # 合并结果
    result = pd.concat(result_dfs)
    
    # 删除NaN行（由于计算移动窗口产生的）
    # result = result.dropna()
    
    return result

def generate_simple_signals(df):
    """
    生成简单的交易信号
    
    参数:
        df: 包含技术指标的DataFrame
        
    返回:
        DataFrame: 添加了交易信号的DataFrame
    """
    # 创建副本以避免修改原始数据
    result = df.copy()
    
    # 添加信号列
    result['MA_signal'] = 0
    result['RSI_signal'] = 0
    result['BB_signal'] = 0
    
    # 移动平均线交叉信号 (MA5 穿过 MA20)
    result['MA_signal'] = np.where(result['MA5'] > result['MA20'], 1, 0)
    
    # RSI信号 (超买超卖)
    result['RSI_signal'] = np.where(result['RSI'] < 30, 1, 0)  # RSI < 30 买入
    result['RSI_signal'] = np.where(result['RSI'] > 70, -1, result['RSI_signal'])  # RSI > 70 卖出
    
    # 布林带信号
    result['BB_signal'] = np.where(result['$close'] < result['BB_lower'], 1, 0)  # 价格低于下轨买入
    result['BB_signal'] = np.where(result['$close'] > result['BB_upper'], -1, result['BB_signal'])  # 价格高于上轨卖出
    
    # 组合信号 (简单加权)
    result['combined_signal'] = 0.5 * result['MA_signal'] + 0.3 * result['RSI_signal'] + 0.2 * result['BB_signal']
    
    return result

def plot_signals(df, symbol, start_date=None, end_date=None):
    """
    绘制价格和信号
    
    参数:
        df: 包含价格和信号的DataFrame
        symbol: 要绘制的币种
        start_date: 开始日期
        end_date: 结束日期
    """
    # 筛选特定币种的数据
    symbol_data = df[df['symbol'] == symbol].copy()
    symbol_data = symbol_data.sort_values('datetime')
    
    # 如果指定了日期范围，则进行过滤
    if start_date:
        symbol_data = symbol_data[symbol_data['datetime'] >= pd.Timestamp(start_date)]
    if end_date:
        symbol_data = symbol_data[symbol_data['datetime'] <= pd.Timestamp(end_date)]
    
    # 绘制价格和MA
    plt.figure(figsize=(12, 10))
    
    # 价格和移动平均线
    plt.subplot(3, 1, 1)
    plt.plot(symbol_data['datetime'], symbol_data['$close'], label='Close Price')
    plt.plot(symbol_data['datetime'], symbol_data['MA5'], label='MA5')
    plt.plot(symbol_data['datetime'], symbol_data['MA20'], label='MA20')
    plt.fill_between(symbol_data['datetime'], symbol_data['BB_lower'], symbol_data['BB_upper'], alpha=0.2, color='gray')
    
    # 添加买入卖出点
    buy_points = symbol_data[symbol_data['combined_signal'] > 0.5]
    sell_points = symbol_data[symbol_data['combined_signal'] < -0.5]
    plt.scatter(buy_points['datetime'], buy_points['$close'], color='green', marker='^', s=100, label='Buy Signal')
    plt.scatter(sell_points['datetime'], sell_points['$close'], color='red', marker='v', s=100, label='Sell Signal')
    
    plt.title(f'{symbol} Price and Moving Averages')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # RSI
    plt.subplot(3, 1, 2)
    plt.plot(symbol_data['datetime'], symbol_data['RSI'], label='RSI')
    plt.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
    plt.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
    plt.title(f'{symbol} RSI')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    
    # MACD
    plt.subplot(3, 1, 3)
    plt.bar(symbol_data['datetime'], symbol_data['MACD_hist'], label='MACD Histogram', width=1)
    plt.plot(symbol_data['datetime'], symbol_data['MACD'], label='MACD')
    plt.plot(symbol_data['datetime'], symbol_data['MACD_signal'], label='Signal Line')
    plt.title(f'{symbol} MACD')
    plt.ylabel('MACD')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{symbol}_analysis.png')
    plt.close()
    
    print(f"图表已保存为 {symbol}_analysis.png")

def run_simple_backtest(df, symbol, start_date, end_date, initial_capital=10000):
    """
    运行简单的回测
    
    参数:
        df: 包含价格和信号的DataFrame
        symbol: 要回测的币种
        start_date: 回测开始日期
        end_date: 回测结束日期
        initial_capital: 初始资金
        
    返回:
        dict: 回测结果
    """
    # 筛选数据
    symbol_data = df[(df['symbol'] == symbol) & 
                     (df['datetime'] >= pd.Timestamp(start_date)) & 
                     (df['datetime'] <= pd.Timestamp(end_date))].copy()
    
    symbol_data = symbol_data.sort_values('datetime')
    
    # 投资组合初始化
    portfolio = pd.DataFrame(index=symbol_data.index)
    portfolio['datetime'] = symbol_data['datetime']
    portfolio['close'] = symbol_data['$close']
    portfolio['position'] = 0
    portfolio['position_value'] = 0
    portfolio['cash'] = initial_capital
    portfolio['portfolio_value'] = initial_capital
    
    # 设置交易费率
    fee_rate = 0.001  # 0.1% 交易费
    
    # 记录买入和卖出点
    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []
    
    # 回测循环
    position = 0
    for i in range(1, len(symbol_data)):
        # 获取当前行和前一行
        current = symbol_data.iloc[i]
        prev = symbol_data.iloc[i-1]
        
        # 默认继承前一天的仓位和资金状态
        portfolio.iloc[i, portfolio.columns.get_loc('position')] = portfolio.iloc[i-1, portfolio.columns.get_loc('position')]
        portfolio.iloc[i, portfolio.columns.get_loc('cash')] = portfolio.iloc[i-1, portfolio.columns.get_loc('cash')]
        
        # 检查信号
        if current['combined_signal'] > 0.5 and position == 0:  # 买入信号且当前无仓位
            # 计算可买入的数量
            available_cash = portfolio.iloc[i-1, portfolio.columns.get_loc('cash')]
            price = current['$close']
            # 考虑交易费用
            max_quantity = available_cash / (price * (1 + fee_rate))
            # 实际买入90%的资金
            quantity = max_quantity * 0.9
            
            # 更新仓位
            position = quantity
            portfolio.iloc[i, portfolio.columns.get_loc('position')] = position
            
            # 更新现金
            cost = position * price * (1 + fee_rate)
            portfolio.iloc[i, portfolio.columns.get_loc('cash')] = available_cash - cost
            
            # 记录买入点
            buy_dates.append(current['datetime'])
            buy_prices.append(price)
            
        elif current['combined_signal'] < -0.5 and position > 0:  # 卖出信号且有仓位
            # 计算卖出金额
            price = current['$close']
            sale_value = position * price * (1 - fee_rate)
            
            # 更新现金
            portfolio.iloc[i, portfolio.columns.get_loc('cash')] += sale_value
            
            # 更新仓位
            position = 0
            portfolio.iloc[i, portfolio.columns.get_loc('position')] = position
            
            # 记录卖出点
            sell_dates.append(current['datetime'])
            sell_prices.append(price)
        
        # 计算仓位价值
        portfolio.iloc[i, portfolio.columns.get_loc('position_value')] = portfolio.iloc[i, portfolio.columns.get_loc('position')] * current['$close']
        
        # 计算投资组合总价值
        portfolio.iloc[i, portfolio.columns.get_loc('portfolio_value')] = portfolio.iloc[i, portfolio.columns.get_loc('position_value')] + portfolio.iloc[i, portfolio.columns.get_loc('cash')]
    
    # 计算回测指标
    portfolio['daily_returns'] = portfolio['portfolio_value'].pct_change()
    portfolio['cumulative_returns'] = (1 + portfolio['daily_returns']).cumprod()
    
    # 计算策略指标
    total_return = portfolio['portfolio_value'].iloc[-1] / initial_capital - 1
    
    # 年化收益率
    days = (portfolio['datetime'].iloc[-1] - portfolio['datetime'].iloc[0]).days
    annual_return = (1 + total_return) ** (365 / days) - 1
    
    # 计算最大回撤
    portfolio['cumulative_max'] = portfolio['portfolio_value'].cummax()
    portfolio['drawdown'] = (portfolio['portfolio_value'] - portfolio['cumulative_max']) / portfolio['cumulative_max']
    max_drawdown = portfolio['drawdown'].min()
    
    # 计算夏普比率 (假设无风险利率为0)
    sharpe_ratio = np.sqrt(252) * portfolio['daily_returns'].mean() / portfolio['daily_returns'].std()
    
    # 绘制回测结果
    plt.figure(figsize=(12, 6))
    
    # 绘制资产价值变化
    plt.plot(portfolio['datetime'], portfolio['portfolio_value'], label='Portfolio Value')
    
    # 买入点和卖出点
    for date, price in zip(buy_dates, buy_prices):
        plt.scatter(date, price, color='green', marker='^', s=100)
    
    for date, price in zip(sell_dates, sell_prices):
        plt.scatter(date, price, color='red', marker='v', s=100)
    
    # 绘制持有现金的基准线
    plt.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Capital')
    
    plt.title(f'{symbol} Backtest Results')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{symbol}_backtest.png')
    plt.close()
    
    print(f"回测图表已保存为 {symbol}_backtest.png")
    
    # 输出回测结果
    backtest_result = {
        'symbol': symbol,
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': initial_capital,
        'final_value': portfolio['portfolio_value'].iloc[-1],
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': len(buy_dates)
    }
    
    return backtest_result

def demo_simple_workflow():
    """
    使用技术指标和回测的简单工作流演示
    """
    print("技术分析和回测工作流示例")
    print("-------------------------")
    
    # 这里应该是从mongo_to_qlib.py导入数据后的处理流程
    print("1. 加载qlib格式数据")
    print("2. 计算技术指标")
    print("3. 生成交易信号")
    print("4. 进行回测分析")
    
    print("注意: 此演示需要先运行mongo_to_qlib.py获取数据")

# 示例使用代码
if __name__ == "__main__":
    print("该模块提供技术指标计算和回测功能")
    print("请通过导入方式使用，例如:")
    print("from technical_indicators import calculate_technical_indicators, generate_simple_signals")
    
    demo_simple_workflow() 