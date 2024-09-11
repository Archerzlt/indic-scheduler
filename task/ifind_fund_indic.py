# -*- coding: utf-8 -*-
import pandas as pd
from iFinDPy import *
from common.config import *

THS_iFinDLogin(THS_ACCOUNT, THS_PASSWORD)
# 公募基金数据
# 月更 其他为日更
def get_FundShares(start_time, end_time):
    shares = THS_EDB('S003138022;S003138023;S003138024;S003138025;S003138026;S003138028','',start_time, end_time)
    df_shares = pd.DataFrame(shares.data)
    df_shares.sort_values('time', ascending=True, inplace=True)
    df_shares['value'] = df_shares.apply(lambda x: float(x['value']), axis=1)
    df_shares.rename(columns={'value': '份额(亿份)'}, inplace=True)
    df_shares.rename(columns={'time': 'Date'}, inplace=True)
    df_shares.rename(columns={'index_name': '指标名称'}, inplace=True)
    df_shares = df_shares.loc[:, ['Date', '指标名称','份额(亿份)']]
    df_shares = df_shares.set_index('Date')
    return df_shares

print(get_FundShares("2022-12-01","2023-04-19"))