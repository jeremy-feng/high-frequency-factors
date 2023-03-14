"""
Author: Chao Feng
Date: 2023-03-14
Description: Functions for calculating high frequency factors

Requirements:
pandas
numpy
"""

import pandas as pd
import numpy as np


def format_factor(data: pd.DataFrame) -> pd.DataFrame:
    """
    Format columns

    Parameters
    ----------
    data : pd.DataFrame
        Data

    Returns
    -------
    pd.DataFrame
        Data with formatted columns
    """
    # Reset index
    data = data.reset_index()
    # Rename columns
    data.rename(
        columns={
            "Code_Mkt": "ticker_str",
            "Qdate": "info_date_ymd",
            "Qtime": "info_time_hms",
        },
        inplace=True,
    )
    # Change data type
    data["ticker_str"] = data["ticker_str"].apply(lambda x: x.split(".")[0])
    data["info_date_ymd"] = data["info_date_ymd"].apply(
        lambda x: int(x.replace("-", ""))
    )
    data["info_time_hms"] = data["info_time_hms"].apply(
        lambda x: int(x.replace(":", ""))
    )
    return data


def calculate_A1(data: pd.DataFrame) -> pd.DataFrame:
    """
    Number of orders arriving in the last 60 s

    Parameters
    ----------
    data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Number of orders arriving in the last 60 s
    """
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderRecNo"].count()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 order 数量之和
    factor = (
        factor.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "OrderRecNo": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A2(data: pd.DataFrame) -> pd.DataFrame:
    """
    Total number of arrived orders up to that time

    Parameters
    ----------
    data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Total number of arrived orders up to that time
    """
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderRecNo"].count()
    # 按照股票和日期分组。在组内，对于每一秒，计算当前累积的 order 数量之和
    factor = factor.groupby(by=["Code_Mkt", "Qdate"]).cumsum()
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "OrderRecNo": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A3(data: pd.DataFrame) -> pd.DataFrame:
    """
    Quantity of arrived orders in the last 60 s

    Parameters
    ----------
    data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Quantity of arrived orders in the last 60 s
    """
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 OrderVol 之和
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"].sum()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 OrderVol 之和
    factor = (
        factor.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "OrderVol": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A4(data: pd.DataFrame) -> pd.DataFrame:
    """
    Total quantity of arrived orders up to that time

    Parameters
    ----------
    data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Total quantity of arrived orders up to that time
    """
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 OrderVol 之和
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"].sum()
    # 按照股票和日期分组。在组内，对于每一秒，计算当前累积的 OrderVol 之和之和
    factor = factor.groupby(by=["Code_Mkt", "Qdate"]).cumsum()
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "OrderVol": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A5(data: pd.DataFrame) -> pd.DataFrame:
    """
    Number of buy orders arriving in the last 60 s

    Parameters
    ----------
    data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Number of buy orders arriving in the last 60 s
    """
    # 筛选出 buy orders
    data = data[data["FunctionCode"] == "1"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderRecNo"].count()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 order 数量之和
    factor = (
        factor.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "OrderRecNo": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A6(data: pd.DataFrame) -> pd.DataFrame:
    """
    Number of sell orders arriving in the last 60 s

    Parameters
    ----------
    data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Number of sell orders arriving in the last 60 s
    """
    # 筛选出 sell orders
    data = data[data["FunctionCode"] == "2"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderRecNo"].count()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 order 数量之和
    factor = (
        factor.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "OrderRecNo": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A7(data: pd.DataFrame) -> pd.DataFrame:
    """
    Quantity of buy orders arriving in the last 60 s

    Parameters
    ----------
    data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Quantity of buy orders arriving in the last 60 s
    """
    # 筛选出 buy orders
    data = data[data["FunctionCode"] == "1"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 OrderVol 之和
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"].sum()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 OrderVol 之和
    factor = (
        factor.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "OrderVol": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A8(data: pd.DataFrame) -> pd.DataFrame:
    """
    Quantity of sell orders arriving in the last 60 s

    Parameters
    ----------
    data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Quantity of sell orders arriving in the last 60 s
    """
    # 筛选出 sell orders
    data = data[data["FunctionCode"] == "2"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 OrderVol 之和
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"].sum()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 OrderVol 之和
    factor = (
        factor.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "OrderVol": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A9(data: pd.DataFrame) -> pd.DataFrame:
    """
    Number of fill and kill orders arriving in the last 60 s

    Parameters
    ----------
    data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Number of fill and kill orders arriving in the last 60 s
    """
    # 没有 fill and kill orders 数据，无法计算
    return pd.DataFrame()


def calculate_A10(data: pd.DataFrame) -> pd.DataFrame:
    """
    Number of cancelled orders in the last 60 s

    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Number of cancelled orders in the last 60 s
    """
    # 筛选出 cancelled orders
    data = data[data["FunctionCode"] == "C"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 order 数量之和
    factor = (
        factor.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "RecNo": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A11(data: pd.DataFrame) -> pd.DataFrame:
    """
    Quantity of cancelled orders in the last 60 s

    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Quantity of cancelled orders in the last 60 s
    """
    # 筛选出 cancelled orders
    data = data[data["FunctionCode"] == "C"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 Tvolume 之和
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 Tvolume 之和
    factor = (
        factor.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "Tvolume": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A12(data: pd.DataFrame) -> pd.DataFrame:
    """
    Number of cancelled buy orders in the last 60 s

    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Number of cancelled buy orders in the last 60 s
    """
    # 筛选出 cancelled buy orders
    data = data[(data["FunctionCode"] == "C") & (data["BuyOrderRecNo"] != 0.0)]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 order 数量之和
    factor = (
        factor.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "RecNo": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A13(data: pd.DataFrame) -> pd.DataFrame:
    """
    Number of cancelled sell orders in the last 60 s

    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Number of cancelled sell orders in the last 60 s
    """
    # 筛选出 cancelled sell orders
    data = data[(data["FunctionCode"] == "C") & (data["SellOrderRecNo"] != 0.0)]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 order 数量之和
    factor = (
        factor.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "RecNo": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A14(data: pd.DataFrame) -> pd.DataFrame:
    """
    Quantity of cancelled buy orders in the last 60 s

    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Quantity of cancelled buy orders in the last 60 s
    """
    # 筛选出 cancelled buy orders
    data = data[(data["FunctionCode"] == "C") & (data["BuyOrderRecNo"] != 0.0)]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 Tvolume 之和
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 Tvolume 之和
    factor = (
        factor.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "Tvolume": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A15(data: pd.DataFrame) -> pd.DataFrame:
    """
    Quantity of cancelled sell orders in the last 60 s

    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Quantity of cancelled sell orders in the last 60 s
    """
    # 筛选出 cancelled sell orders
    data = data[(data["FunctionCode"] == "C") & (data["SellOrderRecNo"] != 0.0)]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 Tvolume 之和
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 Tvolume 之和
    factor = (
        factor.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "Tvolume": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A16(data: pd.DataFrame) -> pd.DataFrame:
    """
    Total number of cancelled orders up to that time

    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Total number of cancelled orders up to that time
    """
    # 筛选出 cancelled orders
    data = data[data["FunctionCode"] == "C"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
    # 按照股票和日期分组。在组内，对于每一秒，计算当前累积的 order 数量之和
    factor = factor.groupby(by=["Code_Mkt", "Qdate"]).cumsum()
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "RecNo": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A17(trade_data: pd.DataFrame, order_data: pd.DataFrame) -> pd.DataFrame:
    """
    Volume weighted average price of cancelled orders up to that time

    Parameters
    ----------
    trade_data : pd.DataFrame
        Trade data
    order_data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Volume weighted average price of cancelled orders up to that time
    """
    # 筛选出 cancelled orders
    trade_data = trade_data[trade_data["FunctionCode"] == "C"]
    # 提取出 trade_data 中必要的数据
    trade_data = trade_data[
        [
            "Code_Mkt",
            "Qdate",
            "Qtime",
            "BuyOrderRecNo",
            "SellOrderRecNo",
            "Tvolume",
        ]
    ]
    # 将 BuyOrderRecNo 和 SellOrderRecNo 合并为 OrderRecNo
    trade_data = trade_data.assign(
        OrderRecNo=trade_data["BuyOrderRecNo"] + trade_data["SellOrderRecNo"]
    )
    # 提取出 order_data 中必要的数据
    order_data = order_data[
        [
            "Code_Mkt",
            "Qdate",
            "OrderRecNo",
            "OrderPr",
        ]
    ]
    # 合并 trade_data 和 order_data
    trade_order = pd.merge(
        trade_data,
        order_data,
        on=["Code_Mkt", "Qdate", "OrderRecNo"],
    )
    # 按照股票和日期分组。在组内，计算 Volume weighted average price of cancelled orders up to that time
    # 计算成交额
    trade_order["Tamount"] = trade_order["Tvolume"] * trade_order["OrderPr"]
    # 计算累计成交额
    trade_order["Tamount_cumsum"] = trade_order.groupby(["Code_Mkt", "Qdate"])[
        "Tamount"
    ].cumsum()
    # 计算累计成交量
    trade_order["Tvolume_cumsum"] = trade_order.groupby(["Code_Mkt", "Qdate"])[
        "Tvolume"
    ].cumsum()
    # 计算 VWAP
    trade_order["VWAP"] = trade_order["Tamount_cumsum"] / trade_order["Tvolume_cumsum"]
    # 提取出必要的数据
    factor = trade_order[
        [
            "Code_Mkt",
            "Qdate",
            "Qtime",
            "VWAP",
        ]
    ]
    # 如果有两行的 Code_Mkt，Qdate 和 Qtime	数据完全相同，则保留最后一行数据
    factor = factor.drop_duplicates(
        subset=["Code_Mkt", "Qdate", "Qtime"],
        keep="last",
    )
    # 将秒钟数据转换为 datetime 对象，并设置为索引
    factor.index = pd.to_datetime(factor["Qdate"] + " " + factor["Qtime"])
    # 重采样并填充空缺值
    factor = factor.resample("S").ffill()
    # 只保留交易时段的数据
    idx1 = factor.index.indexer_between_time("9:15", "11:30")
    idx2 = factor.index.indexer_between_time("13:00", "15:00")
    factor = factor.iloc[np.union1d(idx1, idx2)]
    # 重置索引
    factor.reset_index(inplace=True, drop=True)
    # 设置为多重索引，便于应用整理格式的函数
    factor.set_index(["Code_Mkt", "Qdate", "Qtime"], inplace=True)
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "VWAP": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A18(trade_data: pd.DataFrame, order_data: pd.DataFrame) -> pd.DataFrame:
    """
    Volume weighted average price of cancelled buy orders up to that time

    Parameters
    ----------
    trade_data : pd.DataFrame
        Trade data
    order_data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Volume weighted average price of cancelled buy orders up to that time
    """
    # 筛选出 cancelled orders
    trade_data = trade_data[
        (trade_data["FunctionCode"] == "C") & (trade_data["BuyOrderRecNo"] != 0.0)
    ]
    # 提取出 trade_data 中必要的数据
    trade_data = trade_data[
        [
            "Code_Mkt",
            "Qdate",
            "Qtime",
            "BuyOrderRecNo",
            "Tvolume",
        ]
    ]
    # 提取出 order_data 中必要的数据
    order_data = order_data[
        [
            "Code_Mkt",
            "Qdate",
            "OrderRecNo",
            "OrderPr",
        ]
    ]
    # 合并 trade_data 和 order_data
    trade_order = pd.merge(
        trade_data,
        order_data,
        left_on=["Code_Mkt", "Qdate", "BuyOrderRecNo"],
        right_on=["Code_Mkt", "Qdate", "OrderRecNo"],
    )
    # 按照股票和日期分组。在组内，计算 Volume weighted average price of cancelled buy orders up to that time
    # 计算成交额
    trade_order["Tamount"] = trade_order["Tvolume"] * trade_order["OrderPr"]
    # 计算累计成交额
    trade_order["Tamount_cumsum"] = trade_order.groupby(["Code_Mkt", "Qdate"])[
        "Tamount"
    ].cumsum()
    # 计算累计成交量
    trade_order["Tvolume_cumsum"] = trade_order.groupby(["Code_Mkt", "Qdate"])[
        "Tvolume"
    ].cumsum()
    # 计算 VWAP
    trade_order["VWAP"] = trade_order["Tamount_cumsum"] / trade_order["Tvolume_cumsum"]
    # 提取出必要的数据
    factor = trade_order[
        [
            "Code_Mkt",
            "Qdate",
            "Qtime",
            "VWAP",
        ]
    ]
    # 如果有两行的 Code_Mkt，Qdate 和 Qtime	数据完全相同，则保留最后一行数据
    factor = factor.drop_duplicates(
        subset=["Code_Mkt", "Qdate", "Qtime"],
        keep="last",
    )
    # 将秒钟数据转换为 datetime 对象，并设置为索引
    factor.index = pd.to_datetime(factor["Qdate"] + " " + factor["Qtime"])
    # 重采样并填充空缺值
    factor = factor.resample("S").ffill()
    # 只保留交易时段的数据
    idx1 = factor.index.indexer_between_time("9:15", "11:30")
    idx2 = factor.index.indexer_between_time("13:00", "15:00")
    factor = factor.iloc[np.union1d(idx1, idx2)]
    # 重置索引
    factor.reset_index(inplace=True, drop=True)
    # 设置为多重索引，便于应用整理格式的函数
    factor.set_index(["Code_Mkt", "Qdate", "Qtime"], inplace=True)
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "VWAP": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A19(trade_data: pd.DataFrame, order_data: pd.DataFrame) -> pd.DataFrame:
    """
    Volume weighted average price of cancelled sell orders up to that time

    Parameters
    ----------
    trade_data : pd.DataFrame
        Trade data
    order_data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Volume weighted average price of cancelled sell orders up to that time
    """
    # 筛选出 cancelled orders
    trade_data = trade_data[
        (trade_data["FunctionCode"] == "C") & (trade_data["SellOrderRecNo"] != 0.0)
    ]
    # 提取出 trade_data 中必要的数据
    trade_data = trade_data[
        [
            "Code_Mkt",
            "Qdate",
            "Qtime",
            "SellOrderRecNo",
            "Tvolume",
        ]
    ]
    # 提取出 order_data 中必要的数据
    order_data = order_data[
        [
            "Code_Mkt",
            "Qdate",
            "OrderRecNo",
            "OrderPr",
        ]
    ]
    # 合并 trade_data 和 order_data
    trade_order = pd.merge(
        trade_data,
        order_data,
        left_on=["Code_Mkt", "Qdate", "SellOrderRecNo"],
        right_on=["Code_Mkt", "Qdate", "OrderRecNo"],
    )
    # 按照股票和日期分组。在组内，计算 Volume weighted average price of cancelled sell orders up to that time
    # 计算成交额
    trade_order["Tamount"] = trade_order["Tvolume"] * trade_order["OrderPr"]
    # 计算累计成交额
    trade_order["Tamount_cumsum"] = trade_order.groupby(["Code_Mkt", "Qdate"])[
        "Tamount"
    ].cumsum()
    # 计算累计成交量
    trade_order["Tvolume_cumsum"] = trade_order.groupby(["Code_Mkt", "Qdate"])[
        "Tvolume"
    ].cumsum()
    # 计算 VWAP
    trade_order["VWAP"] = trade_order["Tamount_cumsum"] / trade_order["Tvolume_cumsum"]
    # 提取出必要的数据
    factor = trade_order[
        [
            "Code_Mkt",
            "Qdate",
            "Qtime",
            "VWAP",
        ]
    ]
    # 如果有两行的 Code_Mkt，Qdate 和 Qtime	数据完全相同，则保留最后一行数据
    factor = factor.drop_duplicates(
        subset=["Code_Mkt", "Qdate", "Qtime"],
        keep="last",
    )
    # 将秒钟数据转换为 datetime 对象，并设置为索引
    factor.index = pd.to_datetime(factor["Qdate"] + " " + factor["Qtime"])
    # 重采样并填充空缺值
    factor = factor.resample("S").ffill()
    # 只保留交易时段的数据
    idx1 = factor.index.indexer_between_time("9:15", "11:30")
    idx2 = factor.index.indexer_between_time("13:00", "15:00")
    factor = factor.iloc[np.union1d(idx1, idx2)]
    # 重置索引
    factor.reset_index(inplace=True, drop=True)
    # 设置为多重索引，便于应用整理格式的函数
    factor.set_index(["Code_Mkt", "Qdate", "Qtime"], inplace=True)
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "VWAP": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A20(trade_data: pd.DataFrame, order_data: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio of the number of cancelled orders to the number of arrived orders in the last 60 s

    Parameters
    ----------
    trade_data : pd.DataFrame
        Trade data
    order_data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Ratio of the number of cancelled orders to the number of arrived orders in the last 60 s
    """
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 cancelled orders 数量
    cabcelled_orders = trade_data[trade_data["FunctionCode"] == "C"]
    cabcelled_orders = cabcelled_orders.groupby(["Code_Mkt", "Qdate", "Qtime"])[
        "RecNo"
    ].count()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 cancelled orders 数量之和
    cabcelled_orders = (
        cabcelled_orders.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 arrived orders 数量
    arrived_orders = order_data.groupby(["Code_Mkt", "Qdate", "Qtime"])[
        "OrderRecNo"
    ].count()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 arrived orders 数量之和
    arrived_orders = (
        arrived_orders.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 计算 Ratio of the number of cancelled orders to the number of arrived orders in the last 60 s
    factor = cabcelled_orders / arrived_orders
    # 整理格式
    factor.name = "ratio_of_cancelled_orders_to_arrived_orders"
    factor = format_factor(factor)
    factor.rename(
        columns={
            "ratio_of_cancelled_orders_to_arrived_orders": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A21(trade_data: pd.DataFrame, order_data: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio of the quantity of cancelled orders to the quantity of arrived orders in the last 60 s

    Parameters
    ----------
    trade_data : pd.DataFrame
        Trade data
    order_data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Ratio of the quantity of cancelled orders to the quantity of arrived orders in the last 60 s
    """
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 quantity of cancelled orders 数量
    cabcelled_orders = trade_data[trade_data["FunctionCode"] == "C"]
    cabcelled_orders = cabcelled_orders.groupby(["Code_Mkt", "Qdate", "Qtime"])[
        "Tvolume"
    ].sum()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 quantity of cancelled orders 数量之和
    cabcelled_orders = (
        cabcelled_orders.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 quantity of arrived orders 数量之和
    arrived_orders = order_data.groupby(["Code_Mkt", "Qdate", "Qtime"])[
        "OrderVol"
    ].sum()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 quantity of arrived orders 数量之和
    arrived_orders = (
        arrived_orders.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 计算 Ratio of the quantity of cancelled orders to the quantity of arrived orders in the last 60 s
    factor = cabcelled_orders / arrived_orders
    # 整理格式
    factor.name = "ratio_of_quantity_cancelled_orders_to_arrived_orders"
    factor = format_factor(factor)
    factor.rename(
        columns={
            "ratio_of_quantity_cancelled_orders_to_arrived_orders": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A22(trade_data: pd.DataFrame, order_data: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio of the total number of cancelled orders to the total number of arrived orders up to that time

    Parameters
    ----------
    trade_data : pd.DataFrame
        Trade data
    order_data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Ratio of the total number of cancelled orders to the total number of arrived orders up to that time
    """
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 cancelled orders 数量
    cabcelled_orders = trade_data[trade_data["FunctionCode"] == "C"]
    cabcelled_orders = cabcelled_orders.groupby(["Code_Mkt", "Qdate", "Qtime"])[
        "RecNo"
    ].count()
    # 按照股票和日期分组。在组内，对于每一秒，计算至今的 cancelled orders 数量之和
    cabcelled_orders = cabcelled_orders.groupby(by=["Code_Mkt", "Qdate"]).cumsum()
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 arrived orders 数量
    arrived_orders = order_data.groupby(["Code_Mkt", "Qdate", "Qtime"])[
        "OrderRecNo"
    ].count()
    # 按照股票和日期分组。在组内，对于每一秒，计算至今的 arrived orders 数量之和
    arrived_orders = arrived_orders.groupby(by=["Code_Mkt", "Qdate"]).cumsum()
    # 计算 Ratio of the number of cancelled orders to the number of arrived orders in the last 60 s
    factor = cabcelled_orders / arrived_orders
    # 整理格式
    factor.name = "ratio_of_cancelled_orders_to_arrived_orders"
    factor = format_factor(factor)
    factor.rename(
        columns={
            "ratio_of_cancelled_orders_to_arrived_orders": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A23(trade_data: pd.DataFrame, order_data: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio of the total quantity of cancelled orders to the total quantity of arrived orders up to that time

    Parameters
    ----------
    trade_data : pd.DataFrame
        Trade data
    order_data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Ratio of the total quantity of cancelled orders to the total quantity of arrived orders up to that time
    """
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 quantity of cancelled orders 数量
    cabcelled_orders = trade_data[trade_data["FunctionCode"] == "C"]
    cabcelled_orders = cabcelled_orders.groupby(["Code_Mkt", "Qdate", "Qtime"])[
        "Tvolume"
    ].sum()
    # 按照股票和日期分组。在组内，对于每一秒，计算至今的 quantity of cancelled orders 数量之和
    cabcelled_orders = cabcelled_orders.groupby(by=["Code_Mkt", "Qdate"]).cumsum()
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 quantity of arrived orders 数量之和
    arrived_orders = order_data.groupby(["Code_Mkt", "Qdate", "Qtime"])[
        "OrderVol"
    ].sum()
    # 按照股票和日期分组。在组内，对于每一秒，计算至今的 quantity of arrived orders 数量之和
    arrived_orders = arrived_orders.groupby(by=["Code_Mkt", "Qdate"]).cumsum()
    # 计算 Ratio of the quantity of cancelled orders to the quantity of arrived orders in the last 60 s
    factor = cabcelled_orders / arrived_orders
    # 整理格式
    factor.name = "ratio_of_quantity_cancelled_orders_to_arrived_orders"
    factor = format_factor(factor)
    factor.rename(
        columns={
            "ratio_of_quantity_cancelled_orders_to_arrived_orders": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A24(data: pd.DataFrame) -> pd.DataFrame:
    """
    Average quantity of buy orders in the last 5 min

    Parameters
    ----------
    data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Average quantity of buy orders in the last 5 min
    """
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 quantity of buy orders
    factor = (
        data[data["FunctionCode"] == "1"]
        .groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"]
        .sum()
    )
    # 将秒钟索引转换为 datetime 格式
    factor.index = factor.index.set_levels(
        [
            factor.index.levels[0],
            factor.index.levels[1],
            pd.to_datetime(factor.index.levels[2]),
        ]
    )
    # 按照股票和日期分组。在组内，对于每 5 分钟，计算 quantity of buy orders 的平均值
    factor = factor.groupby(
        [
            pd.Grouper(level=0),
            pd.Grouper(level=1),
            pd.Grouper(level=2, freq="5min", label="right"),
        ]
    ).mean()
    # 将秒钟索引转换为字符串格式
    factor.index = factor.index.set_levels(
        [
            factor.index.levels[0],
            factor.index.levels[1],
            factor.index.levels[2].strftime("%H:%M:%S"),
        ]
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "OrderVol": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A25(data: pd.DataFrame) -> pd.DataFrame:
    """
    Average quantity of sell orders in the last 5 min

    Parameters
    ----------
    data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Average quantity of sell orders in the last 5 min
    """
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 quantity of sell orders 数量
    factor = (
        data[data["FunctionCode"] == "2"]
        .groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"]
        .sum()
    )
    # 将秒钟索引转换为 datetime 格式
    factor.index = factor.index.set_levels(
        [
            factor.index.levels[0],
            factor.index.levels[1],
            pd.to_datetime(factor.index.levels[2]),
        ]
    )
    # 按照股票和日期分组。在组内，对于每 5 分钟，计算 quantity of sell orders 数量的平均值
    factor = factor.groupby(
        [
            pd.Grouper(level=0),
            pd.Grouper(level=1),
            pd.Grouper(level=2, freq="5min", label="right"),
        ]
    ).mean()
    # 将秒钟索引转换为字符串格式
    factor.index = factor.index.set_levels(
        [
            factor.index.levels[0],
            factor.index.levels[1],
            factor.index.levels[2].strftime("%H:%M:%S"),
        ]
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "OrderVol": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A26(data: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility of buy order quantity in the last 5 min

    Parameters
    ----------
    data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Volatility of buy order quantity in the last 5 min
    """
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 quantity of buy orders
    factor = (
        data[data["FunctionCode"] == "1"]
        .groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"]
        .sum()
    )
    # 将秒钟索引转换为 datetime 格式
    factor.index = factor.index.set_levels(
        [
            factor.index.levels[0],
            factor.index.levels[1],
            pd.to_datetime(factor.index.levels[2]),
        ]
    )
    # 按照股票和日期分组。在组内，对于每 5 分钟，计算 quantity of buy orders 的标准差
    factor = factor.groupby(
        [
            pd.Grouper(level=0),
            pd.Grouper(level=1),
            pd.Grouper(level=2, freq="5min", label="right"),
        ]
    ).std()
    # 将秒钟索引转换为字符串格式
    factor.index = factor.index.set_levels(
        [
            factor.index.levels[0],
            factor.index.levels[1],
            factor.index.levels[2].strftime("%H:%M:%S"),
        ]
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "OrderVol": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A27(data: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility of sell order quantity in the last 5 min

    Parameters
    ----------
    data : pd.DataFrame
        Order data

    Returns
    -------
    pd.DataFrame
        Volatility of sell order quantity in the last 5 min
    """
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 quantity of sell orders
    factor = (
        data[data["FunctionCode"] == "2"]
        .groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"]
        .sum()
    )
    # 将秒钟索引转换为 datetime 格式
    factor.index = factor.index.set_levels(
        [
            factor.index.levels[0],
            factor.index.levels[1],
            pd.to_datetime(factor.index.levels[2]),
        ]
    )
    # 按照股票和日期分组。在组内，对于每 5 分钟，计算 quantity of sell orders 的标准差
    factor = factor.groupby(
        [
            pd.Grouper(level=0),
            pd.Grouper(level=1),
            pd.Grouper(level=2, freq="5min", label="right"),
        ]
    ).std()
    # 将秒钟索引转换为字符串格式
    factor.index = factor.index.set_levels(
        [
            factor.index.levels[0],
            factor.index.levels[1],
            factor.index.levels[2].strftime("%H:%M:%S"),
        ]
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "OrderVol": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A28(data: pd.DataFrame) -> pd.DataFrame:
    """
    Volume weighted average price (VWAP) of trades in the last 5 min

    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Volume weighted average price (VWAP) of trades in the last 5 min
    """
    # 筛选出 trades
    trade_data = data[data["FunctionCode"] == "F"]
    # 提取出 trade_data 中必要的数据
    trade_data = trade_data[
        [
            "Code_Mkt",
            "Qdate",
            "Qtime",
            "Tprice",
            "Tvolume",
        ]
    ]
    # 计算每笔 trade 的成交额
    trade_data["Tamount"] = trade_data["Tprice"] * trade_data["Tvolume"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 Volume 和 Amount 之和
    factor = trade_data.groupby(["Code_Mkt", "Qdate", "Qtime"])[
        ["Tvolume", "Tamount"]
    ].sum()
    # 将秒钟索引转换为 datetime 格式
    factor.index = factor.index.set_levels(
        [
            factor.index.levels[0],
            factor.index.levels[1],
            pd.to_datetime(factor.index.levels[2]),
        ]
    )
    # 按照股票和日期分组。在组内，计算 Volume weighted average price (VWAP) of trades in the last 5 min
    factor = factor.groupby(
        [
            pd.Grouper(level=0),
            pd.Grouper(level=1),
            pd.Grouper(level=2, freq="5min", label="right"),
        ]
    ).apply(lambda x: np.sum(x["Tamount"]) / np.sum(x["Tvolume"]))
    # 将秒钟索引转换为字符串格式
    factor.index = factor.index.set_levels(
        [
            factor.index.levels[0],
            factor.index.levels[1],
            factor.index.levels[2].strftime("%H:%M:%S"),
        ]
    )
    # 整理格式
    factor.name = "VWAP_of_trades_in_the_last_5_min"
    factor = format_factor(factor)
    factor.rename(
        columns={
            "VWAP_of_trades_in_the_last_5_min": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A29(data: pd.DataFrame) -> pd.DataFrame:
    """
    Volume weighted average price (VWAP) of trades up to that time

    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Volume weighted average price (VWAP) of trades up to that time
    """
    # 筛选出 trades
    trade_data = data[data["FunctionCode"] == "F"]
    # 提取出 trade_data 中必要的数据
    trade_data = trade_data[
        [
            "Code_Mkt",
            "Qdate",
            "Qtime",
            "Tprice",
            "Tvolume",
        ]
    ]
    # 计算每笔 trade 的成交额
    trade_data["Tamount"] = trade_data["Tprice"] * trade_data["Tvolume"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 Volume 和 Amount 之和
    factor = trade_data.groupby(["Code_Mkt", "Qdate", "Qtime"])[
        ["Tvolume", "Tamount"]
    ].sum()
    # 计算累计成交额
    factor["Tamount_cumsum"] = factor.groupby(["Code_Mkt", "Qdate"])["Tamount"].cumsum()
    # 计算累计成交量
    factor["Tvolume_cumsum"] = factor.groupby(["Code_Mkt", "Qdate"])["Tvolume"].cumsum()
    # 计算 VWAP
    factor["VWAP"] = factor["Tamount_cumsum"] / factor["Tvolume_cumsum"]
    # 提取出必要的数据
    factor = factor[["VWAP"]]
    # 重置索引
    factor.reset_index(inplace=True)
    # 将秒钟数据转换为 datetime 对象，并设置为索引
    factor.index = pd.to_datetime(factor["Qdate"] + " " + factor["Qtime"])
    # 重采样并填充空缺值
    factor = factor.resample("S").ffill()
    # 只保留交易时段的数据
    idx1 = factor.index.indexer_between_time("9:15", "11:30")
    idx2 = factor.index.indexer_between_time("13:00", "15:00")
    factor = factor.iloc[np.union1d(idx1, idx2)]
    # 重置索引
    factor.reset_index(inplace=True, drop=True)
    # 设置为多重索引，便于应用整理格式的函数
    factor.set_index(["Code_Mkt", "Qdate", "Qtime"], inplace=True)
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "VWAP": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A30(data: pd.DataFrame) -> pd.DataFrame:
    """
    Volume weighted average price (VWAP) of buyer-initiated trades in the last 5 min

    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Volume weighted average price (VWAP) of buyer-initiated trades in the last 5 min
    """
    # 筛选出 buyer-initiated trades
    trade_data = data[(data["FunctionCode"] == "F") & (data["Trdirec"] == "5")]
    # 提取出 trade_data 中必要的数据
    trade_data = trade_data[
        [
            "Code_Mkt",
            "Qdate",
            "Qtime",
            "Tprice",
            "Tvolume",
        ]
    ]
    # 计算每笔 trade 的成交额
    trade_data["Tamount"] = trade_data["Tprice"] * trade_data["Tvolume"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 Volume 和 Amount 之和
    factor = trade_data.groupby(["Code_Mkt", "Qdate", "Qtime"])[
        ["Tvolume", "Tamount"]
    ].sum()
    # 将秒钟索引转换为 datetime 格式
    factor.index = factor.index.set_levels(
        [
            factor.index.levels[0],
            factor.index.levels[1],
            pd.to_datetime(factor.index.levels[2]),
        ]
    )
    # 按照股票和日期分组。在组内，计算 Volume weighted average price (VWAP) of trades in the last 5 min
    factor = factor.groupby(
        [
            pd.Grouper(level=0),
            pd.Grouper(level=1),
            pd.Grouper(level=2, freq="5min", label="right"),
        ]
    ).apply(lambda x: np.sum(x["Tamount"]) / np.sum(x["Tvolume"]))
    # 将秒钟索引转换为字符串格式
    factor.index = factor.index.set_levels(
        [
            factor.index.levels[0],
            factor.index.levels[1],
            factor.index.levels[2].strftime("%H:%M:%S"),
        ]
    )
    # 整理格式
    factor.name = "VWAP_of_buyer-initiated_trades_in_the_last_5_min"
    factor = format_factor(factor)
    factor.rename(
        columns={
            "VWAP_of_buyer-initiated_trades_in_the_last_5_min": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A31(data: pd.DataFrame) -> pd.DataFrame:
    """
    Volume weighted average price (VWAP) of seller-initiated trades in the last 5 min

    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Volume weighted average price (VWAP) of seller-initiated trades in the last 5 min
    """
    # 筛选出 seller-initiated trades
    trade_data = data[(data["FunctionCode"] == "F") & (data["Trdirec"] == "1")]
    # 提取出 trade_data 中必要的数据
    trade_data = trade_data[
        [
            "Code_Mkt",
            "Qdate",
            "Qtime",
            "Tprice",
            "Tvolume",
        ]
    ]
    # 计算每笔 trade 的成交额
    trade_data["Tamount"] = trade_data["Tprice"] * trade_data["Tvolume"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 Volume 和 Amount 之和
    factor = trade_data.groupby(["Code_Mkt", "Qdate", "Qtime"])[
        ["Tvolume", "Tamount"]
    ].sum()
    # 将秒钟索引转换为 datetime 格式
    factor.index = factor.index.set_levels(
        [
            factor.index.levels[0],
            factor.index.levels[1],
            pd.to_datetime(factor.index.levels[2]),
        ]
    )
    # 按照股票和日期分组。在组内，计算 Volume weighted average price (VWAP) of trades in the last 5 min
    factor = factor.groupby(
        [
            pd.Grouper(level=0),
            pd.Grouper(level=1),
            pd.Grouper(level=2, freq="5min", label="right"),
        ]
    ).apply(lambda x: np.sum(x["Tamount"]) / np.sum(x["Tvolume"]))
    # 将秒钟索引转换为字符串格式
    factor.index = factor.index.set_levels(
        [
            factor.index.levels[0],
            factor.index.levels[1],
            factor.index.levels[2].strftime("%H:%M:%S"),
        ]
    )
    # 整理格式
    factor.name = "VWAP_of_seller-initiated_trades_in_the_last_5_min"
    factor = format_factor(factor)
    factor.rename(
        columns={
            "VWAP_of_seller-initiated_trades_in_the_last_5_min": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A32(data: pd.DataFrame) -> pd.DataFrame:
    """
    Number of buyer-initiated trades in the last 60 s

    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Number of buyer-initiated trades in the last 60 s
    """
    # 筛选出 buyer-initiated trades
    data = data[data["Trdirec"] == "5"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 order 数量之和
    factor = (
        factor.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "RecNo": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A33(data: pd.DataFrame) -> pd.DataFrame:
    """
    Number of seller-initiated trades in the last 60 s

    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Number of seller-initiated trades in the last 60 s
    """
    # 筛选出 seller-initiated trades
    data = data[data["Trdirec"] == "1"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 order 数量之和
    factor = (
        factor.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "RecNo": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A34(data: pd.DataFrame) -> pd.DataFrame:
    """
    Quantity of buyer-initiated trades in the last 60 s

    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Quantity of buyer-initiated trades in the last 60 s
    """
    # 筛选出 buyer-initiated trades
    data = data[data["Trdirec"] == "5"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 order Tvolume 之和
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 Tvolume 之和
    factor = (
        factor.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "Tvolume": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A35(data: pd.DataFrame) -> pd.DataFrame:
    """
    Quantity of seller-initiated trades in the last 60 s


    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Quantity of seller-initiated trades in the last 60 s

    """
    # 筛选出 seller-initiated trades
    data = data[data["Trdirec"] == "1"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 order Tvolume 之和
    factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 Tvolume 之和
    factor = (
        factor.groupby(by=["Code_Mkt", "Qdate"])
        .rolling(60)
        .sum()
        .droplevel(level=[0, 1])
    )
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "Tvolume": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A36(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio of the number of buyer-initiated trades to the number of seller-initiated trades in the last 60 s


    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Ratio of the number of buyer-initiated trades to the number of seller-initiated trades in the last 60 s

    """
    # 计算 buyer-initiated trades
    buyer_initiated_trades = calculate_A32(data)
    # 设置多重索引
    buyer_initiated_trades.set_index(
        ["ticker_str", "info_date_ymd", "info_time_hms"], inplace=True
    )
    # 计算 seller-initiated trades
    seller_initiated_trades = calculate_A33(data)
    # 设置多重索引
    seller_initiated_trades.set_index(
        ["ticker_str", "info_date_ymd", "info_time_hms"], inplace=True
    )
    # 计算 buyer-initiated trades / seller-initiated trades
    factor = buyer_initiated_trades / seller_initiated_trades
    factor.reset_index(inplace=True)
    return factor


def calculate_A37(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio of the quantity of buyer-initiated trades to the quantity of seller-initiated trades in the last 60 s


    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Ratio of the quantity of buyer-initiated trades to the quantity of seller-initiated trades in the last 60 s

    """
    # 计算 buyer-initiated trades
    buyer_initiated_trades = calculate_A34(data)
    # 设置多重索引
    buyer_initiated_trades.set_index(
        ["ticker_str", "info_date_ymd", "info_time_hms"], inplace=True
    )
    # 计算 seller-initiated trades
    seller_initiated_trades = calculate_A35(data)
    # 设置多重索引
    seller_initiated_trades.set_index(
        ["ticker_str", "info_date_ymd", "info_time_hms"], inplace=True
    )
    # 计算 buyer-initiated trades / seller-initiated trades
    factor = buyer_initiated_trades / seller_initiated_trades
    factor.reset_index(inplace=True)
    return factor


def calculate_A38(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative ratio of the total number of buyer-initiated trades to the total number of seller-initiated trades


    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Cumulative ratio of the total number of buyer-initiated trades to the total number of seller-initiated trades

    """
    # 计算 Cumulative buyer-initiated trades
    # 筛选出 buyer-initiated trades
    buyer_initiated_data = data[data["Trdirec"] == "5"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 trade 数量之和
    buyer_initiated_trades = buyer_initiated_data.groupby(
        ["Code_Mkt", "Qdate", "Qtime"]
    )["RecNo"].count()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去所有秒的 trade 数量之和
    buyer_initiated_trades = buyer_initiated_trades.groupby(
        by=["Code_Mkt", "Qdate"]
    ).cumsum()
    # 计算 Cumulative seller-initiated trades
    # 筛选出 seller-initiated trades
    seller_initiated_data = data[data["Trdirec"] == "1"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 trade 数量之和
    seller_initiated_trades = seller_initiated_data.groupby(
        ["Code_Mkt", "Qdate", "Qtime"]
    )["RecNo"].count()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去所有秒的 trade 数量之和
    seller_initiated_trades = seller_initiated_trades.groupby(
        by=["Code_Mkt", "Qdate"]
    ).cumsum()
    # 计算 Cumulative buyer-initiated trades / Cumulative seller-initiated trades
    factor = buyer_initiated_trades / seller_initiated_trades
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "RecNo": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor


def calculate_A39(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative ratio of the total quantity of buyer-initiated trades to the total quantity of seller-initiated trades


    Parameters
    ----------
    data : pd.DataFrame
        Trade data

    Returns
    -------
    pd.DataFrame
        Cumulative ratio of the total quantity of buyer-initiated trades to the total quantity of seller-initiated trades

    """
    # 计算 Cumulative buyer-initiated trades
    # 筛选出 buyer-initiated trades
    buyer_initiated_data = data[data["Trdirec"] == "5"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 volume 之和
    buyer_initiated_trades = buyer_initiated_data.groupby(
        ["Code_Mkt", "Qdate", "Qtime"]
    )["Tvolume"].sum()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去所有秒的 volume 之和
    buyer_initiated_trades = buyer_initiated_trades.groupby(
        by=["Code_Mkt", "Qdate"]
    ).cumsum()
    # 计算 Cumulative seller-initiated trades
    # 筛选出 seller-initiated trades
    seller_initiated_data = data[data["Trdirec"] == "1"]
    # 按照股票、日期和秒钟分组。在组内，计算每秒的 volume 之和
    seller_initiated_trades = seller_initiated_data.groupby(
        ["Code_Mkt", "Qdate", "Qtime"]
    )["Tvolume"].sum()
    # 按照股票和日期分组。在组内，对于每一秒，计算过去所有秒的 volume 之和
    seller_initiated_trades = seller_initiated_trades.groupby(
        by=["Code_Mkt", "Qdate"]
    ).cumsum()
    # 计算 Cumulative quantity of buyer-initiated trades / Cumulative seller-initiated trades
    factor = buyer_initiated_trades / seller_initiated_trades
    # 整理格式
    factor = format_factor(factor)
    factor.rename(
        columns={
            "Tvolume": "value",
        },
        inplace=True,
    )
    factor["value"] = factor["value"].astype(float)
    return factor
