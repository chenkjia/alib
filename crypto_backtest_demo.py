#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
加密货币回测演示脚本

此脚本演示如何结合mongo_to_qlib.py和technical_indicators.py的功能进行加密货币回测
"""

# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

# 导入自定义模块
from mongo_to_qlib import connect_mongodb, get_crypto_list, process_all_crypto_data, prepare_qlib_format_data
from technical_indicators import calculate_technical_indicators, generate_simple_signals, plot_signals, run_simple_backtest

def main():
    """
    主函数 - 演示从MongoDB获取数据并进行回测的完整流程
    """
    print("加密货币技术分析和回测演示")
    print("==========================")
    
    # 步骤1: 从MongoDB获取数据并转换为qlib格式
    print("\n步骤1: 数据获取与预处理")
    print("------------------------")
    
    # 连接MongoDB
    client, db = connect_mongodb()
    print("已连接到MongoDB")
    
    # 获取加密货币列表
    crypto_list = get_crypto_list(db)
    print(f"找到{len(crypto_list)}个加密货币")
    
    # 为了演示，只处理前3个加密货币
    demo_cryptos = crypto_list[:3]
    print(f"演示用例将使用以下加密货币: {demo_cryptos}")
    
    # 获取并处理数据
    crypto_data = process_all_crypto_data(db, demo_cryptos)
    
    if crypto_data is not None:
        print(f"成功提取数据，共{len(crypto_data)}行")
        
        # 转换为qlib格式
        qlib_data = prepare_qlib_format_data(crypto_data)
        print("数据已转换为qlib格式")
        
        # 步骤2: 计算技术指标
        print("\n步骤2: 计算技术指标")
        print("------------------")
        tech_data = calculate_technical_indicators(qlib_data)
        print("技术指标计算完成")
        
        # 步骤3: 生成交易信号
        print("\n步骤3: 生成交易信号")
        print("------------------")
        signal_data = generate_simple_signals(tech_data)
        print("交易信号生成完成")
        
        # 步骤4: 绘制技术分析图表
        print("\n步骤4: 技术分析图表")
        print("------------------")
        
        # 只处理第一个币种
        demo_symbol = demo_cryptos[0]
        print(f"为{demo_symbol}绘制技术分析图表")
        
        # 只使用最近的180天数据进行分析
        latest_date = signal_data['datetime'].max()
        start_date = latest_date - pd.Timedelta(days=180)
        plot_signals(signal_data, demo_symbol, start_date=start_date)
        
        # 步骤5: 回测
        print("\n步骤5: 策略回测")
        print("---------------")
        print(f"对{demo_symbol}进行回测分析")
        
        # 运行回测
        backtest_result = run_simple_backtest(
            signal_data, 
            demo_symbol, 
            start_date=start_date, 
            end_date=latest_date
        )
        
        # 打印回测结果
        print("\n回测结果摘要:")
        print(f"币种: {backtest_result['symbol']}")
        print(f"回测期间: {backtest_result['start_date']} 至 {backtest_result['end_date']}")
        print(f"初始资金: ${backtest_result['initial_capital']}")
        print(f"最终价值: ${backtest_result['final_value']:.2f}")
        print(f"总收益率: {backtest_result['total_return']:.2%}")
        print(f"年化收益率: {backtest_result['annual_return']:.2%}")
        print(f"最大回撤: {backtest_result['max_drawdown']:.2%}")
        print(f"夏普比率: {backtest_result['sharpe_ratio']:.4f}")
        print(f"交易次数: {backtest_result['num_trades']}")
    else:
        print("未能获取加密货币数据")
    
    # 关闭MongoDB连接
    client.close()
    print("\nMongoDB连接已关闭")
    
    print("\n演示完成。")

if __name__ == "__main__":
    main() 