# -*- coding: utf-8 -*-
import prefect
from prefect import flow, task, get_run_logger
from db.database import Connection
import pandas as pd
from iFinDPy import *
from common.config import *

fields = ["INDIC_ID","PERIOD_DATE","DATA_VALUE"]

@task(name = "login_ths",description = "同花顺iFind插件登录")
def login_ths():
    logger = get_run_logger()
    HttpCode = THS_iFinDLogin(THS_ACCOUNT, THS_PASSWORD)
    logger.info("同花顺api登录状态码: {}".format(HttpCode))

@task(name = "logout_ths",description = "同花顺iFind插件登出")
def logout_ths():
    logger = get_run_logger()
    HttpCode = THS_iFinDLogout()
    logger.info("登出状态码:%s" % HttpCode)

@task(name = "get_ifind_tradeday",description = "获取同花顺api当前交易日")
def get_ifind_tradeday():
    """
    获取当前交易日
    :return:
    """
    currentDate = datetime.now() # 要补之前的修改这里 72
    currentDate = currentDate.strftime("%Y-%m-%d")
    # 获取交易日期
    tdate = THS_Date_Query('212001', 'mode:1,dateType:0,period:D,dateFormat:0', currentDate, currentDate)
    data_str = tdate.data
    # 如果交易日期取不出来 返回空字符串
    return data_str

# 上涨数量
@task(name = "get_StockUp",description = "获取同花顺api上涨数量",retries = 1,retry_delay_seconds = 10)
def get_StockUp(conn:Connection,date:str):
    IndicID = 1000000002
    ret = conn.get("SELECT DATA_TABLE FROM `indic_info_pro` WHERE INDIC_ID = %s",(IndicID))
    tablename = ret.get("DATA_TABLE")

    stockup_str = '上涨 '+ str(int(date.replace('-','')))
    stockup = THS_WCQuery(stockup_str,'stock')
    df_stockup = pd.DataFrame(stockup.data)
    stockup_count = len(df_stockup)
    stockup_dict = {'Date':[date],'上涨数量': [stockup_count]}
    df_stockup_count = pd.DataFrame(data = stockup_dict)
    # df_stockup_count = df_stockup_count.set_index('Date')
    # return df_stockup_count
    item = dict(zip(fields, [IndicID] + df_stockup_count.values.tolist()[0]))
    conn.table_insert(tablename,item)
    conn.table_update("indic_info_pro",{"END_DATE":date},"INDIC_ID",IndicID)


# 下跌数量
@task(name = "get_StockDown",description = "获取同花顺api下跌数量",retries = 1,retry_delay_seconds = 10)
def get_StockDown(conn:Connection,date:str):
    IndicID = 1000000003
    ret = conn.get("SELECT DATA_TABLE FROM `indic_info_pro` WHERE INDIC_ID = %s",(IndicID))
    tablename = ret.get("DATA_TABLE")

    stockdown_str = '下跌 '+ str(int(date.replace('-','')))
    stockdown = THS_WCQuery(stockdown_str,'stock')
    df_stockdown = pd.DataFrame(stockdown.data)
    stockdown_count = len(df_stockdown)
    stockdown_dict = {'Date':[date],'下跌数量': [stockdown_count]}
    df_stockdown_count = pd.DataFrame(data = stockdown_dict)
    # df_stockdown_count = df_stockdown_count.set_index('Date')
    # return df_stockdown_count
    item = dict(zip(fields, [IndicID] + df_stockdown_count.values.tolist()[0]))
    conn.table_insert(tablename,item)
    conn.table_update("indic_info_pro",{"END_DATE":date},"INDIC_ID",IndicID)


# 平盘数量
@task(name = "get_StockNoChange",description = "获取同花顺api平盘数量",retries = 1,retry_delay_seconds = 10)
def get_StockNoChange(conn:Connection,date:str):

    IndicID = 1000000004
    ret = conn.get("SELECT DATA_TABLE FROM `indic_info_pro` WHERE INDIC_ID = %s",(IndicID))
    tablename = ret.get("DATA_TABLE")

    stocknochange_str = '平盘 '+ str(int(date.replace('-','')))
    stocknochange = THS_WCQuery(stocknochange_str,'stock')
    df_stocknochange = pd.DataFrame(stocknochange.data)
    stocknochange_count = len(df_stocknochange)
    stocknochange_dict = {'Date':[date],'平盘数量': [stocknochange_count]}
    df_stocknochange_count = pd.DataFrame(data = stocknochange_dict)
    # df_stocknochange_count = df_stocknochange_count.set_index('Date')
    # return df_stocknochange_count
    item = dict(zip(fields, [IndicID] + df_stocknochange_count.values.tolist()[0]))
    conn.table_insert(tablename,item)
    conn.table_update("indic_info_pro",{"END_DATE":date},"INDIC_ID",IndicID)

# 涨停数量
@task(name = "get_StockLimitUp",description = "获取同花顺api涨停数量",retries = 1,retry_delay_seconds = 10)
def get_StockLimitUp(conn:Connection,date:str):
    IndicID = 1000000005
    ret = conn.get("SELECT DATA_TABLE FROM `indic_info_pro` WHERE INDIC_ID = %s",(IndicID))
    tablename = ret.get("DATA_TABLE")

    stocklimitup_str = '涨停 '+ str(int(date.replace('-','')))
    stocklimitup = THS_WCQuery(stocklimitup_str,'stock')
    df_stocklimitup = pd.DataFrame(stocklimitup.data)
    stocklimitup_count = len(df_stocklimitup)
    stocklimitup_dict = {'Date':[date],'涨停数量': [stocklimitup_count]}
    df_stocklimitup_count = pd.DataFrame(data = stocklimitup_dict)
    # df_stocklimitup_count = df_stocklimitup_count.set_index('Date')
    # return df_stocklimitup_count
    item = dict(zip(fields, [IndicID] + df_stocklimitup_count.values.tolist()[0]))
    conn.table_insert(tablename,item)
    conn.table_update("indic_info_pro",{"END_DATE":date},"INDIC_ID",IndicID)

# 跌停数量
@task(name = "get_StockLimitDown",description = "获取同花顺api跌停数量",retries = 1,retry_delay_seconds = 10)
def get_StockLimitDown(conn:Connection,date:str):
    IndicID = 1000000006
    ret = conn.get("SELECT DATA_TABLE FROM `indic_info_pro` WHERE INDIC_ID = %s",(IndicID))
    tablename = ret.get("DATA_TABLE")

    stocklimitdown_str = '跌停 '+ str(int(date.replace('-','')))
    stocklimitdown = THS_WCQuery(stocklimitdown_str,'stock')
    df_stocklimitdown = pd.DataFrame(stocklimitdown.data)
    stocklimitdown_count = len(df_stocklimitdown)
    stocklimitdown_dict = {'Date':[date],'跌停数量': [stocklimitdown_count]}
    df_stocklimitdown_count = pd.DataFrame(data = stocklimitdown_dict)
    # df_stocklimitdown_count = df_stocklimitdown_count.set_index('Date')
    # return df_stocklimitdown_count
    item = dict(zip(fields, [IndicID] + df_stocklimitdown_count.values.tolist()[0]))
    conn.table_insert(tablename,item)
    conn.table_update("indic_info_pro",{"END_DATE":date},"INDIC_ID",IndicID)

# 停牌数量 suspension
@task(name = "get_StockSuspension",description = "获取同花顺api停牌数量",retries = 1,retry_delay_seconds = 10)
def get_StockSuspension(conn:Connection,date:str):
    IndicID = 1000000007
    ret = conn.get("SELECT DATA_TABLE FROM `indic_info_pro` WHERE INDIC_ID = %s",(IndicID))
    tablename = ret.get("DATA_TABLE")

    stocksuspension_str = '停牌 '+ str(int(date.replace('-','')))
    stocksuspension = THS_WCQuery(stocksuspension_str,'stock')
    df_stocksuspension = pd.DataFrame(stocksuspension.data)
    stocksuspension_count = len(df_stocksuspension)
    stocksuspension_dict = {'Date':[date],'停牌数量': [stocksuspension_count]}
    df_stocksuspension_count = pd.DataFrame(data = stocksuspension_dict)
    # df_stocksuspension_count = df_stocksuspension_count.set_index('Date')
    # return df_stocksuspension_count
    item = dict(zip(fields, [IndicID] +  df_stocksuspension_count.values.tolist()[0]))
    conn.table_insert(tablename, item)
    conn.table_update("indic_info_pro", {"END_DATE": date}, "INDIC_ID", IndicID)

# 创新高数量（近一年新高）
@task(name = "get_StockPriceHitNewPeakProportion",description = "获取同花顺api创新高数量（近一年）占比",retries = 1,retry_delay_seconds = 10)
def get_StockPriceHitNewPeakProportion(conn:Connection,date:str):
    IndicID = 1000000008
    ret = conn.get("SELECT DATA_TABLE FROM `indic_info_pro` WHERE INDIC_ID = %s",(IndicID))
    tablename = ret.get("DATA_TABLE")

    stockpricehitnewpeak_str = '近一年新高 '+ str(int(date.replace('-','')))
    stockpricehitnewpeak = THS_WCQuery(stockpricehitnewpeak_str,'stock')
    df_stockpricehitnewpeak = pd.DataFrame(stockpricehitnewpeak.data)
    stockpricehitnewpeak_count = len(df_stockpricehitnewpeak)
    number_of_stock = len(pd.DataFrame((THS_WCQuery( '交易中 '+ str(int(date.replace('-',''))),'stock')).data))
    stockpricehitnewpeak_proportion = stockpricehitnewpeak_count/number_of_stock
    stockpricehitnewpeak_proportion_dict = {'Date':[date],'创新高个股数量占比': [stockpricehitnewpeak_proportion * 100]} # * 100 百分号单位
    df_stockpricehitnewpeak_proportion = pd.DataFrame(data = stockpricehitnewpeak_proportion_dict)
    # df_stockpricehitnewpeak_proportion = df_stockpricehitnewpeak_proportion.set_index('Date')
    item = dict(zip(fields, [IndicID] + df_stockpricehitnewpeak_proportion.values.tolist()[0]))
    # return df_stockpricehitnewpeak_proportion
    conn.table_insert(tablename, item)
    conn.table_update("indic_info_pro", {"END_DATE": date}, "INDIC_ID", IndicID)


# 市值/GDP
@task(name = "get_MarketValueToGDP",description = "获取同花顺api 市值/GDP 比值",retries = 1,retry_delay_seconds = 10)
def get_MarketValueToGDP(conn:Connection,date:str):
    IndicID = 1000000009
    ret = conn.get("SELECT DATA_TABLE FROM `indic_info_pro` WHERE INDIC_ID = %s",(IndicID))
    tablename = ret.get("DATA_TABLE")

    year = str(datetime.now().year - 1) + "-12-31"
    marketvalue_str = '市值 '+ str(int(date.replace('-','')))
    marketvalue = THS_WCQuery(marketvalue_str,'stock')
    df_marketvalue = pd.DataFrame(marketvalue.data)
    marketvalue_amount = df_marketvalue.iloc[:,-1].sum()
    GDP = float(pd.DataFrame((THS_EDB('M001620247','',year,year).data))['value'][0])
    marketvaluetogdp = marketvalue_amount/GDP/100000000
    marketvaluetogdp_dict = {'Date':[date],'总市值比GDP': [marketvaluetogdp * 100]} # * 100 百分号单位
    df_marketvaluetogdp = pd.DataFrame(data = marketvaluetogdp_dict)
    item = dict(zip(fields, [IndicID] + df_marketvaluetogdp.values.tolist()[0]))
    # df_marketvaluetogdp = df_marketvaluetogdp.set_index('Date')
    # return df_marketvaluetogdp
    conn.table_insert(tablename, item)
    conn.table_update("indic_info_pro", {"END_DATE": date}, "INDIC_ID", IndicID)

@flow(name = "market_state",description = "A股市场状态工作流")
def get_market_state():
    logger = get_run_logger()
    # 获取mysql数据库连接
    conn = Connection(host=CAT_MYSQL_HOST,port=CAT_MYSQL_PORT,user=CAT_MYSQL_USER,password=CAT_MYSQL_PASSWORD,database=CAT_MYSQL_DB)
    login_ths()
    currentDate = get_ifind_tradeday()
    if len(currentDate) == 0:
        logger.info("今天不是交易日,workflow结束......")
    else:
        # 取数逻辑
        state = get_StockUp(conn=conn,date = currentDate,return_state=True)
        if state.is_failed():
            logger.error("获取同花顺api-上涨数量失败...")
        state = get_StockDown(conn=conn,date = currentDate ,return_state=True)
        if state.is_failed():
            logger.error("获取同花顺api-下跌数量失败...")
        state = get_StockNoChange(conn=conn,date = currentDate,return_state=True)
        if state.is_failed():
            logger.error("获取同花顺api-平盘数量失败...")
        state = get_StockLimitUp(conn=conn,date = currentDate,return_state=True)
        if state.is_failed():
            logger.error("获取同花顺api-涨停数量失败...")
        state = get_StockLimitDown(conn=conn,date = currentDate,return_state=True)
        if state.is_failed():
            logger.error("获取同花顺api-跌停数量失败...")
        state = get_StockSuspension(conn=conn,date = currentDate,return_state=True)
        if state.is_failed():
            logger.error("获取同花顺api-停牌数量失败...")
        state = get_StockPriceHitNewPeakProportion(conn=conn,date = currentDate,return_state=True)
        if state.is_failed():
            logger.error("获取同花顺api-创新高比值失败...")
        state =get_MarketValueToGDP(conn=conn,date = currentDate,return_state=True)
        if state.is_failed():
            logger.error("获取同花顺api-市值/GDP比值失败...")
    logout_ths()
