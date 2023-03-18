"""
Author: Chao Feng
Date: 2023-03-14
Description: Functions for calculating high frequency factors

Requirements:
pandas
numpy
os
tqdm
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm


class FactorCalculator:
    def __init__(
        self,
        order_path: str,
        trade_path: str,
        factors_index_second_path: str,
    ) -> None:
        """

        Parameters
        ----------
        order_path : str
            Path of order data
        trade_path : str
            Path of trade data
        factors_index_second_path : str
            Path of factors index with second frequency

        """
        self.order_path = order_path
        self.trade_path = trade_path
        self.factors_index_second_path = factors_index_second_path
        # 读取 order 数据
        self.order = pd.read_csv(
            self.order_path,
            dtype={
                "Exchflg": "int",
                "Code": "string",
                "Code_Mkt": "string",
                "Qdate": "string",
                "Qtime": "string",
                "SetNo": "int",
                "OrderRecNo": "int",
                "OrderPr": "float",
                "OrderVol": "float",
                "OrderKind": "string",
                "FunctionCode": "string",
            },
        )
        # 读取 trade 数据
        self.trade = pd.read_csv(
            "./data/trade_stkhf202101_000001sz.csv",
            dtype={
                "Exchflg": "int",
                "Code": "string",
                "Code_Mkt": "string",
                "Qdate": "string",
                "Qtime": "string",
                "SetNo": "int",
                "RecNo": "int",
                "BuyOrderRecNo": "int",
                "SellOrderRecNo": "int",
                "Tprice": "float",
                "Tvolume": "float",
                "Tsum": "float",
                "Tvolume_accu": "float",
                "OrderKind": "string",
                "FunctionCode": "string",
                "Trdirec": "string",
            },
        )
        # 如果存在所有因子的数据文件，则读取因子数据
        if os.path.exists(self.factors_index_second_path):
            self.factors_index_second = pd.read_csv(self.factors_index_second_path)
            self.factors_index_second = self.factors_index_second.set_index(
                ["Code_Mkt", "Qdate", "Qtime"]
            )
        else:
            raise FileNotFoundError("Please provide the path of factors index.")

    def format_factor(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def calculate_A1(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.order，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.order
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderRecNo"].count()
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 order 数量之和
        factor = (
            factor.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "OrderRecNo": "A1",
            },
            inplace=True,
        )
        factor["A1"] = factor["A1"].astype(float)
        return factor

    def calculate_A2(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.order，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.order
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderRecNo"].count()
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        # 按照股票和日期分组。在组内，对于每一秒，计算当前累积的 order 数量之和
        factor = factor.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "OrderRecNo": "A2",
            },
            inplace=True,
        )
        factor["A2"] = factor["A2"].astype(float)
        return factor

    def calculate_A3(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.order，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.order
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 OrderVol 之和
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"].sum()
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 OrderVol 之和
        factor = (
            factor.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "OrderVol": "A3",
            },
            inplace=True,
        )
        factor["A3"] = factor["A3"].astype(float)
        return factor

    def calculate_A4(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.order，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.order
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 OrderVol 之和
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"].sum()
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        # 按照股票和日期分组。在组内，对于每一秒，计算当前累积的 OrderVol 之和之和
        factor = factor.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "OrderVol": "A4",
            },
            inplace=True,
        )
        factor["A4"] = factor["A4"].astype(float)
        return factor

    def calculate_A5(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.order，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.order
        # 筛选出 buy orders
        data = data[data["FunctionCode"] == "1"]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderRecNo"].count()
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 order 数量之和
        factor = (
            factor.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "OrderRecNo": "A5",
            },
            inplace=True,
        )
        factor["A5"] = factor["A5"].astype(float)
        return factor

    def calculate_A6(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.order，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.order
        # 筛选出 sell orders
        data = data[data["FunctionCode"] == "2"]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderRecNo"].count()
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 order 数量之和
        factor = (
            factor.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "OrderRecNo": "A6",
            },
            inplace=True,
        )
        factor["A6"] = factor["A6"].astype(float)
        return factor

    def calculate_A7(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.order，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.order
        # 筛选出 buy orders
        data = data[data["FunctionCode"] == "1"]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 OrderVol 之和
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"].sum()
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 OrderVol 之和
        factor = (
            factor.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "OrderVol": "A7",
            },
            inplace=True,
        )
        factor["A7"] = factor["A7"].astype(float)
        return factor

    def calculate_A8(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.order，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.order
        # 筛选出 sell orders
        data = data[data["FunctionCode"] == "2"]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 OrderVol 之和
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"].sum()
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 OrderVol 之和
        factor = (
            factor.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "OrderVol": "A8",
            },
            inplace=True,
        )
        factor["A8"] = factor["A8"].astype(float)
        return factor

    def calculate_A9(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 由于中国股市没有 fill and kill 数据，因此将 A9 全部设为空值
        factor = pd.DataFrame(index=self.factors_index_second.index, columns=["A9"])
        factor["A9"] = np.NaN
        # 整理格式
        factor = self.format_factor(factor)
        factor["A9"] = factor["A9"].astype(float)
        return factor

    def calculate_A10(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
        # 筛选出 cancelled orders
        data = data[data["FunctionCode"] == "C"]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 order 数量之和
        factor = (
            factor.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "RecNo": "A10",
            },
            inplace=True,
        )
        factor["A10"] = factor["A10"].astype(float)
        return factor

    def calculate_A11(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
        # 筛选出 cancelled orders
        data = data[data["FunctionCode"] == "C"]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 Tvolume 之和
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 Tvolume 之和
        factor = (
            factor.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "Tvolume": "A11",
            },
            inplace=True,
        )
        factor["A11"] = factor["A11"].astype(float)
        return factor

    def calculate_A12(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
        # 筛选出 cancelled buy orders
        data = data[(data["FunctionCode"] == "C") & (data["BuyOrderRecNo"] != 0.0)]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 order 数量之和
        factor = (
            factor.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "RecNo": "A12",
            },
            inplace=True,
        )
        factor["A12"] = factor["A12"].astype(float)
        return factor

    def calculate_A13(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
        # 筛选出 cancelled sell orders
        data = data[(data["FunctionCode"] == "C") & (data["SellOrderRecNo"] != 0.0)]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 order 数量之和
        factor = (
            factor.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "RecNo": "A13",
            },
            inplace=True,
        )
        factor["A13"] = factor["A13"].astype(float)
        return factor

    def calculate_A14(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
        # 筛选出 cancelled buy orders
        data = data[(data["FunctionCode"] == "C") & (data["BuyOrderRecNo"] != 0.0)]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 Tvolume 之和
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 Tvolume 之和
        factor = (
            factor.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "Tvolume": "A14",
            },
            inplace=True,
        )
        factor["A14"] = factor["A14"].astype(float)
        return factor

    def calculate_A15(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
        # 筛选出 cancelled sell orders
        data = data[(data["FunctionCode"] == "C") & (data["SellOrderRecNo"] != 0.0)]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 Tvolume 之和
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 Tvolume 之和
        factor = (
            factor.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "Tvolume": "A15",
            },
            inplace=True,
        )
        factor["A15"] = factor["A15"].astype(float)
        return factor

    def calculate_A16(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
        # 筛选出 cancelled orders
        data = data[data["FunctionCode"] == "C"]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        # 按照股票和日期分组。在组内，对于每一秒，计算当前累积的 order 数量之和
        factor = factor.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "RecNo": "A16",
            },
            inplace=True,
        )
        factor["A16"] = factor["A16"].astype(float)
        return factor

    def calculate_A17(
        self, trade_data: pd.DataFrame = None, order_data: pd.DataFrame = None
    ) -> pd.DataFrame:
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
        # 默认使用 self.trade 和 self.order，如果指定了 data，则使用 data 中的数据
        if trade_data is None:
            trade_data = self.trade
        if order_data is None:
            order_data = self.order
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
        trade_order["Tamount_cumsum"] = (
            trade_order.groupby(["Code_Mkt", "Qdate"])["Tamount"].cumsum().shift(1)
        )
        # 计算累计成交量
        trade_order["Tvolume_cumsum"] = (
            trade_order.groupby(["Code_Mkt", "Qdate"])["Tvolume"].cumsum().shift(1)
        )
        # 计算 VWAP
        trade_order["VWAP"] = (
            trade_order["Tamount_cumsum"] / trade_order["Tvolume_cumsum"]
        )
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
        # 对比 factor_index，用前一秒钟的数据填补缺失值
        factor.set_index(["Code_Mkt", "Qdate", "Qtime"], inplace=True)
        factor = factor.reindex(self.factors_index_second.index, method="ffill")
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "VWAP": "A17",
            },
            inplace=True,
        )
        factor["A17"] = factor["A17"].astype(float)
        return factor

    def calculate_A18(
        self, trade_data: pd.DataFrame = None, order_data: pd.DataFrame = None
    ) -> pd.DataFrame:
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
        # 默认使用 self.trade 和 self.order，如果指定了 data，则使用 data 中的数据
        if trade_data is None:
            trade_data = self.trade
        if order_data is None:
            order_data = self.order
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
        trade_order["Tamount_cumsum"] = (
            trade_order.groupby(["Code_Mkt", "Qdate"])["Tamount"].cumsum().shift(1)
        )
        # 计算累计成交量
        trade_order["Tvolume_cumsum"] = (
            trade_order.groupby(["Code_Mkt", "Qdate"])["Tvolume"].cumsum().shift(1)
        )
        # 计算 VWAP
        trade_order["VWAP"] = (
            trade_order["Tamount_cumsum"] / trade_order["Tvolume_cumsum"]
        )
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
        # 对比 factor_index，用前一秒钟的数据填补缺失值
        factor.set_index(["Code_Mkt", "Qdate", "Qtime"], inplace=True)
        factor = factor.reindex(self.factors_index_second.index, method="ffill")
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "VWAP": "A18",
            },
            inplace=True,
        )
        factor["A18"] = factor["A18"].astype(float)
        return factor

    def calculate_A19(
        self, trade_data: pd.DataFrame = None, order_data: pd.DataFrame = None
    ) -> pd.DataFrame:
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
        # 默认使用 self.trade 和 self.order，如果指定了 data，则使用 data 中的数据
        if trade_data is None:
            trade_data = self.trade
        if order_data is None:
            order_data = self.order
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
        trade_order["Tamount_cumsum"] = (
            trade_order.groupby(["Code_Mkt", "Qdate"])["Tamount"].cumsum().shift(1)
        )
        # 计算累计成交量
        trade_order["Tvolume_cumsum"] = (
            trade_order.groupby(["Code_Mkt", "Qdate"])["Tvolume"].cumsum().shift(1)
        )
        # 计算 VWAP
        trade_order["VWAP"] = (
            trade_order["Tamount_cumsum"] / trade_order["Tvolume_cumsum"]
        )
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
        # 对比 factor_index，用前一秒钟的数据填补缺失值
        factor.set_index(["Code_Mkt", "Qdate", "Qtime"], inplace=True)
        factor = factor.reindex(self.factors_index_second.index, method="ffill")
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "VWAP": "A19",
            },
            inplace=True,
        )
        factor["A19"] = factor["A19"].astype(float)
        return factor

    def calculate_A20(
        self, trade_data: pd.DataFrame = None, order_data: pd.DataFrame = None
    ) -> pd.DataFrame:
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
        # 默认使用 self.trade 和 self.order，如果指定了 data，则使用 data 中的数据
        if trade_data is None:
            trade_data = self.trade
        if order_data is None:
            order_data = self.order
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 cancelled orders 数量
        cabcelled_orders = trade_data[trade_data["FunctionCode"] == "C"]
        cabcelled_orders = cabcelled_orders.groupby(["Code_Mkt", "Qdate", "Qtime"])[
            "RecNo"
        ].count()
        # 对比 factor_index，填补缺失值为 0
        cabcelled_orders = cabcelled_orders.reindex(self.factors_index_second.index)
        cabcelled_orders = cabcelled_orders.fillna(0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 cancelled orders 数量之和
        cabcelled_orders = (
            cabcelled_orders.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 arrived orders 数量
        arrived_orders = order_data.groupby(["Code_Mkt", "Qdate", "Qtime"])[
            "OrderRecNo"
        ].count()
        # 对比 factor_index，填补缺失值为 0
        arrived_orders = arrived_orders.reindex(self.factors_index_second.index)
        arrived_orders = arrived_orders.fillna(0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 arrived orders 数量之和
        arrived_orders = (
            arrived_orders.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 计算 Ratio of the number of cancelled orders to the number of arrived orders in the last 60 s
        factor = cabcelled_orders / arrived_orders
        # 整理格式
        factor.name = "ratio_of_cancelled_orders_to_arrived_orders"
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "ratio_of_cancelled_orders_to_arrived_orders": "A20",
            },
            inplace=True,
        )
        factor["A20"] = factor["A20"].astype(float)
        return factor

    def calculate_A21(
        self, trade_data: pd.DataFrame = None, order_data: pd.DataFrame = None
    ) -> pd.DataFrame:
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
        # 默认使用 self.trade 和 self.order，如果指定了 data，则使用 data 中的数据
        if trade_data is None:
            trade_data = self.trade
        if order_data is None:
            order_data = self.order
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 quantity of cancelled orders 数量
        cabcelled_orders = trade_data[trade_data["FunctionCode"] == "C"]
        cabcelled_orders = cabcelled_orders.groupby(["Code_Mkt", "Qdate", "Qtime"])[
            "Tvolume"
        ].sum()
        # 对比 factor_index，填补缺失值为 0
        cabcelled_orders = cabcelled_orders.reindex(self.factors_index_second.index)
        cabcelled_orders = cabcelled_orders.fillna(0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 quantity of cancelled orders 数量之和
        cabcelled_orders = (
            cabcelled_orders.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 quantity of arrived orders 数量之和
        arrived_orders = order_data.groupby(["Code_Mkt", "Qdate", "Qtime"])[
            "OrderVol"
        ].sum()
        # 对比 factor_index，填补缺失值为 0
        arrived_orders = arrived_orders.reindex(self.factors_index_second.index)
        arrived_orders = arrived_orders.fillna(0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 quantity of arrived orders 数量之和
        arrived_orders = (
            arrived_orders.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 计算 Ratio of the quantity of cancelled orders to the quantity of arrived orders in the last 60 s
        factor = cabcelled_orders / arrived_orders
        # 整理格式
        factor.name = "ratio_of_quantity_cancelled_orders_to_arrived_orders"
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "ratio_of_quantity_cancelled_orders_to_arrived_orders": "A21",
            },
            inplace=True,
        )
        factor["A21"] = factor["A21"].astype(float)
        return factor

    def calculate_A22(
        self, trade_data: pd.DataFrame = None, order_data: pd.DataFrame = None
    ) -> pd.DataFrame:
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
        # 默认使用 self.trade 和 self.order，如果指定了 data，则使用 data 中的数据
        if trade_data is None:
            trade_data = self.trade
        if order_data is None:
            order_data = self.order
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 cancelled orders 数量
        cabcelled_orders = trade_data[trade_data["FunctionCode"] == "C"]
        cabcelled_orders = cabcelled_orders.groupby(["Code_Mkt", "Qdate", "Qtime"])[
            "RecNo"
        ].count()
        # 对比 factor_index，填补缺失值为 0
        cabcelled_orders = cabcelled_orders.reindex(self.factors_index_second.index)
        cabcelled_orders = cabcelled_orders.fillna(0)
        # 按照股票和日期分组。在组内，对于每一秒，计算至今的 cancelled orders 数量之和
        cabcelled_orders = (
            cabcelled_orders.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)
        )
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 arrived orders 数量
        arrived_orders = order_data.groupby(["Code_Mkt", "Qdate", "Qtime"])[
            "OrderRecNo"
        ].count()
        # 对比 factor_index，填补缺失值为 0
        arrived_orders = arrived_orders.reindex(self.factors_index_second.index)
        arrived_orders = arrived_orders.fillna(0)
        # 按照股票和日期分组。在组内，对于每一秒，计算至今的 arrived orders 数量之和
        arrived_orders = (
            arrived_orders.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)
        )
        # 计算 Ratio of the total number of cancelled orders to the total number of arrived orders up to that time
        factor = cabcelled_orders / arrived_orders
        # 整理格式
        factor.name = "ratio_of_cancelled_orders_to_arrived_orders"
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "ratio_of_cancelled_orders_to_arrived_orders": "A22",
            },
            inplace=True,
        )
        factor["A22"] = factor["A22"].astype(float)
        return factor

    def calculate_A23(
        self, trade_data: pd.DataFrame = None, order_data: pd.DataFrame = None
    ) -> pd.DataFrame:
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
        # 默认使用 self.trade 和 self.order，如果指定了 data，则使用 data 中的数据
        if trade_data is None:
            trade_data = self.trade
        if order_data is None:
            order_data = self.order
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 quantity of cancelled orders 数量
        cabcelled_orders = trade_data[trade_data["FunctionCode"] == "C"]
        cabcelled_orders = cabcelled_orders.groupby(["Code_Mkt", "Qdate", "Qtime"])[
            "Tvolume"
        ].sum()
        # 对比 factor_index，填补缺失值为 0
        cabcelled_orders = cabcelled_orders.reindex(self.factors_index_second.index)
        cabcelled_orders = cabcelled_orders.fillna(0)
        # 按照股票和日期分组。在组内，对于每一秒，计算至今的 quantity of cancelled orders 数量之和
        cabcelled_orders = (
            cabcelled_orders.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)
        )
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 quantity of arrived orders 数量之和
        arrived_orders = order_data.groupby(["Code_Mkt", "Qdate", "Qtime"])[
            "OrderVol"
        ].sum()
        # 对比 factor_index，填补缺失值为 0
        arrived_orders = arrived_orders.reindex(self.factors_index_second.index)
        arrived_orders = arrived_orders.fillna(0)
        # 按照股票和日期分组。在组内，对于每一秒，计算至今的 quantity of arrived orders 数量之和
        arrived_orders = (
            arrived_orders.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)
        )
        # 计算 Ratio of the total quantity of cancelled orders to the total quantity of arrived orders up to that time
        factor = cabcelled_orders / arrived_orders
        # 整理格式
        factor.name = "ratio_of_quantity_cancelled_orders_to_arrived_orders"
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "ratio_of_quantity_cancelled_orders_to_arrived_orders": "A23",
            },
            inplace=True,
        )
        factor["A23"] = factor["A23"].astype(float)
        return factor

    def calculate_A24(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.order，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.order
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 quantity of buy orders
        factor = (
            data[data["FunctionCode"] == "1"]
            .groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"]
            .sum()
        )
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
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
        # 对比 factor_index，缺失值为 NaN
        factor = factor.reindex(self.factors_index_second.index)
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "OrderVol": "A24",
            },
            inplace=True,
        )
        factor["A24"] = factor["A24"].astype(float)
        return factor

    def calculate_A25(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.order，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.order
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 quantity of sell orders 数量
        factor = (
            data[data["FunctionCode"] == "2"]
            .groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"]
            .sum()
        )
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
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
        # 对比 factor_index，缺失值为 NaN
        factor = factor.reindex(self.factors_index_second.index)
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "OrderVol": "A25",
            },
            inplace=True,
        )
        factor["A25"] = factor["A25"].astype(float)
        return factor

    def calculate_A26(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.order，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.order
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 quantity of buy orders
        factor = (
            data[data["FunctionCode"] == "1"]
            .groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"]
            .sum()
        )
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
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
        # 对比 factor_index，缺失值为 NaN
        factor = factor.reindex(self.factors_index_second.index)
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "OrderVol": "A26",
            },
            inplace=True,
        )
        factor["A26"] = factor["A26"].astype(float)
        return factor

    def calculate_A27(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.order，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.order
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 quantity of sell orders
        factor = (
            data[data["FunctionCode"] == "2"]
            .groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"]
            .sum()
        )
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
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
        # 对比 factor_index，缺失值为 NaN
        factor = factor.reindex(self.factors_index_second.index)
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "OrderVol": "A27",
            },
            inplace=True,
        )
        factor["A27"] = factor["A27"].astype(float)
        return factor

    def calculate_A28(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
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
        # 对比 factor_index，缺失值为 NaN
        factor = factor.reindex(self.factors_index_second.index)
        # 整理格式
        factor.name = "VWAP_of_trades_in_the_last_5_min"
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "VWAP_of_trades_in_the_last_5_min": "A28",
            },
            inplace=True,
        )
        factor["A28"] = factor["A28"].astype(float)
        return factor

    def calculate_A29(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
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
        # 对比 factor_index，填补缺失值为 0
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        # 计算累计成交额
        factor["Tamount_cumsum"] = (
            factor.groupby(["Code_Mkt", "Qdate"])["Tamount"].cumsum().shift(1)
        )
        # 计算累计成交量
        factor["Tvolume_cumsum"] = (
            factor.groupby(["Code_Mkt", "Qdate"])["Tvolume"].cumsum().shift(1)
        )
        # 计算 VWAP
        factor["VWAP"] = factor["Tamount_cumsum"] / factor["Tvolume_cumsum"]
        # 提取出必要的数据
        factor = factor[["VWAP"]]
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "VWAP": "A29",
            },
            inplace=True,
        )
        factor["A29"] = factor["A29"].astype(float)
        return factor

    def calculate_A30(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
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
        # 按照股票和日期分组。在组内，计算 Volume weighted average price (VWAP) of buyer-initiated trades in the last 5 min
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
        # 对比 factor_index，缺失值为 NaN
        factor = factor.reindex(self.factors_index_second.index)
        # 整理格式
        factor.name = "VWAP_of_buyer-initiated_trades_in_the_last_5_min"
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "VWAP_of_buyer-initiated_trades_in_the_last_5_min": "A30",
            },
            inplace=True,
        )
        factor["A30"] = factor["A30"].astype(float)
        return factor

    def calculate_A31(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
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
        # 按照股票和日期分组。在组内，计算 Volume weighted average price (VWAP) of seller-initiated trades in the last 5 min
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
        # 对比 factor_index，缺失值为 NaN
        factor = factor.reindex(self.factors_index_second.index)
        # 整理格式
        factor.name = "VWAP_of_seller-initiated_trades_in_the_last_5_min"
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "VWAP_of_seller-initiated_trades_in_the_last_5_min": "A31",
            },
            inplace=True,
        )
        factor["A31"] = factor["A31"].astype(float)
        return factor

    def calculate_A32(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
        # 筛选出 buyer-initiated trades
        data = data[data["Trdirec"] == "5"]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
        # 对比 factor_index，填充缺失值为 0
        factor = factor.reindex(self.factors_index_second.index).fillna(0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 order 数量之和
        factor = (
            factor.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "RecNo": "A32",
            },
            inplace=True,
        )
        factor["A32"] = factor["A32"].astype(float)
        return factor

    def calculate_A33(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
        # 筛选出 seller-initiated trades
        data = data[data["Trdirec"] == "1"]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 order 数量
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
        # 对比 factor_index，填充缺失值为 0
        factor = factor.reindex(self.factors_index_second.index).fillna(0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 order 数量之和
        factor = (
            factor.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "RecNo": "A33",
            },
            inplace=True,
        )
        factor["A33"] = factor["A33"].astype(float)
        return factor

    def calculate_A34(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
        # 筛选出 buyer-initiated trades
        data = data[data["Trdirec"] == "5"]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 order Tvolume 之和
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
        # 对比 factor_index，填充缺失值为 0
        factor = factor.reindex(self.factors_index_second.index).fillna(0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 Tvolume 之和
        factor = (
            factor.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "Tvolume": "A34",
            },
            inplace=True,
        )
        factor["A34"] = factor["A34"].astype(float)
        return factor

    def calculate_A35(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
        # 筛选出 seller-initiated trades
        data = data[data["Trdirec"] == "1"]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 order Tvolume 之和
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
        # 对比 factor_index，填充缺失值为 0
        factor = factor.reindex(self.factors_index_second.index).fillna(0)
        # 按照股票和日期分组。在组内，对于每一秒，计算过去 60 秒的 Tvolume 之和
        factor = (
            factor.groupby(by=["Code_Mkt", "Qdate"])
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "Tvolume": "A35",
            },
            inplace=True,
        )
        factor["A35"] = factor["A35"].astype(float)
        return factor

    def calculate_A36(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
        # 计算 buyer-initiated trades
        buyer_initiated_trades = self.calculate_A32(data)
        # 设置多重索引
        buyer_initiated_trades.set_index(
            ["ticker_str", "info_date_ymd", "info_time_hms"], inplace=True
        )
        # 计算 seller-initiated trades
        seller_initiated_trades = self.calculate_A33()
        # 设置多重索引
        seller_initiated_trades.set_index(
            ["ticker_str", "info_date_ymd", "info_time_hms"], inplace=True
        )
        # 计算 buyer-initiated trades / seller-initiated trades
        factor = buyer_initiated_trades["A32"] / seller_initiated_trades["A33"]
        # 整理格式
        factor.name = "A36"
        factor = factor.reset_index()
        return factor

    def calculate_A37(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
        # 计算 buyer-initiated trades
        buyer_initiated_trades = self.calculate_A34(data)
        # 设置多重索引
        buyer_initiated_trades.set_index(
            ["ticker_str", "info_date_ymd", "info_time_hms"], inplace=True
        )
        # 计算 seller-initiated trades
        seller_initiated_trades = self.calculate_A35(data)
        # 设置多重索引
        seller_initiated_trades.set_index(
            ["ticker_str", "info_date_ymd", "info_time_hms"], inplace=True
        )
        # 计算 buyer-initiated trades / seller-initiated trades
        factor = buyer_initiated_trades["A34"] / seller_initiated_trades["A35"]
        # 整理格式
        factor.name = "A37"
        factor = factor.reset_index()
        return factor

    def calculate_A38(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
        # 计算 Cumulative buyer-initiated trades
        # 筛选出 buyer-initiated trades
        buyer_initiated_data = data[data["Trdirec"] == "5"]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 trade 数量之和
        buyer_initiated_trades = buyer_initiated_data.groupby(
            ["Code_Mkt", "Qdate", "Qtime"]
        )["RecNo"].count()
        # 对比 factor_index，填充缺失值为 0
        buyer_initiated_trades = buyer_initiated_trades.reindex(
            self.factors_index_second.index, fill_value=0
        )
        # 按照股票和日期分组。在组内，对于每一秒，计算过去所有秒的 trade 数量之和
        buyer_initiated_trades = (
            buyer_initiated_trades.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)
        )
        # 计算 Cumulative seller-initiated trades
        # 筛选出 seller-initiated trades
        seller_initiated_data = data[data["Trdirec"] == "1"]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 trade 数量之和
        seller_initiated_trades = seller_initiated_data.groupby(
            ["Code_Mkt", "Qdate", "Qtime"]
        )["RecNo"].count()
        # 对比 factor_index，填充缺失值为 0
        seller_initiated_trades = seller_initiated_trades.reindex(
            self.factors_index_second.index, fill_value=0
        )
        # 按照股票和日期分组。在组内，对于每一秒，计算过去所有秒的 trade 数量之和
        seller_initiated_trades = (
            seller_initiated_trades.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)
        )
        # 计算 Cumulative buyer-initiated trades / Cumulative seller-initiated trades
        factor = buyer_initiated_trades / seller_initiated_trades
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "RecNo": "A38",
            },
            inplace=True,
        )
        factor["A38"] = factor["A38"].astype(float)
        return factor

    def calculate_A39(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
        # 默认使用 self.trade，如果指定了 data，则使用 data 中的数据
        if data is None:
            data = self.trade
        # 计算 Cumulative buyer-initiated trades
        # 筛选出 buyer-initiated trades
        buyer_initiated_data = data[data["Trdirec"] == "5"]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 volume 之和
        buyer_initiated_trades = buyer_initiated_data.groupby(
            ["Code_Mkt", "Qdate", "Qtime"]
        )["Tvolume"].sum()
        # 对比 factor_index，填充缺失值为 0
        buyer_initiated_trades = buyer_initiated_trades.reindex(
            self.factors_index_second.index, fill_value=0
        )
        # 按照股票和日期分组。在组内，对于每一秒，计算过去所有秒的 volume 之和
        buyer_initiated_trades = (
            buyer_initiated_trades.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)
        )
        # 计算 Cumulative seller-initiated trades
        # 筛选出 seller-initiated trades
        seller_initiated_data = data[data["Trdirec"] == "1"]
        # 按照股票、日期和秒钟分组。在组内，计算每秒的 volume 之和
        seller_initiated_trades = seller_initiated_data.groupby(
            ["Code_Mkt", "Qdate", "Qtime"]
        )["Tvolume"].sum()
        # 对比 factor_index，填充缺失值为 0
        seller_initiated_trades = seller_initiated_trades.reindex(
            self.factors_index_second.index, fill_value=0
        )
        # 按照股票和日期分组。在组内，对于每一秒，计算过去所有秒的 volume 之和
        seller_initiated_trades = (
            seller_initiated_trades.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)
        )
        # 计算 Cumulative quantity of buyer-initiated trades / Cumulative seller-initiated trades
        factor = buyer_initiated_trades / seller_initiated_trades
        # 整理格式
        factor = self.format_factor(factor)
        factor.rename(
            columns={
                "Tvolume": "A39",
            },
            inplace=True,
        )
        factor["A39"] = factor["A39"].astype(float)
        return factor
