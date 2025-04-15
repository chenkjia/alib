#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
加密货币数据转换工具 - MongoDB到qlib格式

此文件包含将MongoDB中的加密货币数据转换为qlib格式的工具函数
"""

# 导入必要的库
import pandas as pd
import numpy as np
from pymongo import MongoClient
import datetime
import os
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

# ---------------------------------------------
# 函数定义部分
# ---------------------------------------------

def connect_mongodb():
    """
    连接到MongoDB数据库
    
    返回:
        client: MongoDB客户端连接
        db: 数据库对象
    """
    # 连接到MongoDB, 默认端口27017
    client = MongoClient('mongodb://localhost:27017/')
    db = client['crypto']  # 数据库名为crypto
    return client, db

def get_crypto_list(db):
    """
    获取所有加密货币的列表，按日线数据量排序
    
    参数:
        db: MongoDB数据库连接
        
    返回:
        list: 按日线数据量排序的加密货币symbol列表
    """
    # 假设集合名为'crypto_data'
    collection = db['crypto_data']
    
    # 使用聚合管道获取所有币种及其日线数据量
    pipeline = [
        {
            "$project": {
                "_id": 0,
                "symbol": 1,
                "dayline_count": {"$size": {"$ifNull": ["$dayline", []]}}
            }
        },
        {"$sort": {"dayline_count": -1}}  # 按dayline数量降序排序
    ]
    
    result = list(collection.aggregate(pipeline))
    
    # 提取排序后的symbol列表
    symbols = [doc["symbol"] for doc in result]
    return symbols

def fetch_crypto_data(db, symbol):
    """
    从MongoDB获取指定加密货币的日线数据
    
    参数:
        db: MongoDB数据库连接
        symbol: 加密货币的symbol
        
    返回:
        DataFrame: 包含该货币日线数据的DataFrame
    """
    collection = db['crypto_data']
    crypto_doc = collection.find_one({'symbol': symbol})
    
    if not crypto_doc or 'dayline' not in crypto_doc:
        return None
    
    # 将dayline数组转换为DataFrame
    df = pd.DataFrame(crypto_doc['dayline'])
    # 设置日期为索引
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    # 添加symbol列，用于后续合并多个币种数据
    df['symbol'] = symbol
    
    return df

def process_all_crypto_data(db, symbols=None):
    """
    处理所有加密货币数据，并合并为一个大的DataFrame
    
    参数:
        db: MongoDB数据库连接
        symbols: 要处理的加密货币列表，如果为None则处理全部
        
    返回:
        DataFrame: 合并后的所有加密货币数据
    """
    if symbols is None:
        symbols = get_crypto_list(db)
    
    all_data = []
    
    for symbol in symbols:
        df = fetch_crypto_data(db, symbol)
        if df is not None and not df.empty:
            all_data.append(df)
    
    # 合并所有数据
    if all_data:
        combined_df = pd.concat(all_data)
        return combined_df
    
    return None

def prepare_qlib_format_data(crypto_df):
    """
    将加密货币数据转换为qlib所需的格式
    
    参数:
        crypto_df: 原始加密货币数据DataFrame
        
    返回:
        DataFrame: qlib格式的数据
    """
    # qlib需要的基本格式: symbol, datetime, feature1, feature2, ...
    # 重置索引，将日期从索引转为列
    df = crypto_df.reset_index()
    
    # 重命名列以符合qlib命名规范
    df = df.rename(columns={
        'date': 'datetime',
        'open': '$open',
        'high': '$high',
        'low': '$low',
        'close': '$close',
        'volume': '$volume'
    })
    
    # 排序 - 按symbol和日期
    df = df.sort_values(['symbol', 'datetime'])
    
    return df

def demo_init_qlib():
    """
    初始化qlib环境的示例函数
    
    注意: 这个函数在实际运行时需要取消注释
    """
    # 这里是qlib初始化的示例代码
    # qlib.init(provider_uri='~/.qlib/qlib_data/crypto_data', 
    #          region=REG_CN)
    
    print("qlib环境初始化示例函数 - 实际运行时需要取消注释")

def save_to_csv(qlib_data, output_path='qlib_crypto_data'):
    """
    将qlib格式的数据保存为CSV文件
    
    参数:
        qlib_data: qlib格式的DataFrame
        output_path: 输出目录路径
    """
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 按币种分组保存
    for symbol in qlib_data['symbol'].unique():
        symbol_data = qlib_data[qlib_data['symbol'] == symbol]
        symbol_file = os.path.join(output_path, f"{symbol}.csv")
        symbol_data.to_csv(symbol_file, index=False)
        print(f"已保存 {symbol} 数据到 {symbol_file}")
    
    # 保存合并数据
    all_data_file = os.path.join(output_path, "all_crypto_data.csv")
    qlib_data.to_csv(all_data_file, index=False)
    print(f"已保存所有数据到 {all_data_file}")

# ---------------------------------------------
# 主函数
# ---------------------------------------------

def main():
    """
    主函数 - 演示基本的数据提取和处理流程
    """
    print("MongoDB加密货币数据转qlib格式工具")
    print("---------------------------")
    
    # 连接MongoDB
    client, db = connect_mongodb()
    print("已连接到MongoDB")
    
    # 获取加密货币列表
    crypto_list = get_crypto_list(db)
    print(f"找到{len(crypto_list)}个加密货币")
    
    # 处理所有加密货币数据
    print("处理所有加密货币数据...")
    crypto_data = process_all_crypto_data(db, crypto_list)
    
    if crypto_data is not None:
        print(f"成功提取数据，共{len(crypto_data)}行")
        
        # 转换为qlib格式
        qlib_data = prepare_qlib_format_data(crypto_data)
        print("数据已转换为qlib格式")
        
        # 显示数据示例
        print("\n数据示例:")
        print(qlib_data.head())
        
        # 可选：保存数据到CSV
        # save_to_csv(qlib_data)
    else:
        print("未能获取加密货币数据")
    
    # 关闭MongoDB连接
    client.close()
    print("\nMongoDB连接已关闭")
    
    print("\n数据转换完成。可以使用此数据进行后续的qlib回测分析。")

if __name__ == "__main__":
    main() 