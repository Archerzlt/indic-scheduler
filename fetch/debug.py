# import json
# import requests
#
# def get_s_price_low_d(code,trade_date,adjust_type,count_type):
#     url = 'http://192.168.195.214:4444/get_s_price_low_d/'
#     headers = {
#         'accept': 'application/json',
#         'Content-Type': 'application/json'
#     }
#
#     data =  {
#         "code": {
#             "symbols": code
#         },
#         "trade_date": {
#             "dates": trade_date
#         },
#         "adjust_type": {
#             "type": adjust_type
#         },
#         "count_type": {
#             "type": count_type
#         }
#     }
#
#     response = requests.post(url, headers=headers, data=json.dumps(data))
#
#     if response.status_code == 200:
#         print("Request successful.")
#         return response.json()
#     else:
#         print("Request failed with status code:", response.status_code)
#         return False
#
# print(get_s_price_low_d(code = ["002568"],trade_date = ["20240830"],count_type = '1', adjust_type = 'B'))



"""
2 Example
"""
import requests
import json
import math
import numpy as np
import pandas as pd

"""
Series 类

这个是下面以DataFrame为输入的基础函数
return pd.Series format
"""


def EMA(Series, N):
    return pd.Series.ewm(Series, span=N, min_periods=N - 1, adjust=True).mean()


def MA(Series, N):
    return pd.Series.rolling(Series, N).mean()


# 威廉SMA  参考https://www.joinquant.com/post/867


def SMA(Series, N, M=1):
    """
    威廉SMA算法

    本次修正主要是对于返回值的优化,现在的返回值会带上原先输入的索引index
    2018/5/3
    @yutiansut
    """
    ret = []
    i = 1
    length = len(Series)
    # 跳过X中前面几个 nan 值
    while i < length:
        if np.isnan(Series.iloc[i]):
            i += 1
        else:
            break
    try:
        preY = Series.iloc[i]  # Y'
        ret.append(preY)
    except Exception as e:
        print("威廉SMA算法exception...")
    while i < length:
        Y = (M * Series.iloc[i] + (N - M) * preY) / float(N)
        ret.append(Y)
        preY = Y
        i += 1
    return pd.Series(ret, index=Series.tail(len(ret)).index)


def DIFF(Series, N=1):
    return pd.Series(Series).diff(N)


def HHV(Series, N):
    return pd.Series(Series).rolling(N).max()


def LLV(Series, N):
    return pd.Series(Series).rolling(N).min()


def SUM(Series, N):
    # todo 进行了修正 N =0 会输出全为0 的 list 改为累积函数
    if N == 0:
        return Series.cumsum()
    else:
        return pd.Series.rolling(Series, N).sum()


def ABS(Series):
    return abs(Series)


def MAX(A, B):
    var = IF(A > B, A, B)
    return var


def MIN(A, B):
    var = IF(A < B, A, B)
    return var


def SINGLE_CROSS(A, B):
    if A.iloc[-2] < B.iloc[-2] and A.iloc[-1] > B.iloc[-1]:
        return True
    else:
        return False


def EXIST(Series, N=5):
    '''
    n日内是否存在某一值，
    输入值为true or false的Series
    '''
    res = pd.DataFrame(Series) + 0
    res = res.rolling(N).sum() > 0
    res = res[res.columns[0]]
    return res


def EVERY(Series, N=5):
    '''
    n日内是否一直存在某一值，
    输入值为true or false的Series
    '''
    res = pd.DataFrame(Series) + 0
    res = res.rolling(N).sum() > N - 1
    res = res[res.columns[0]]
    return res


def CROSS(A, B):
    """A<B then A>B  A上穿B B下穿A

    Arguments:
        A {[type]} -- [description]
        B {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    var = np.where(A < B, 1, 0)
    try:
        index = A.index
    except:
        index = B.index
    return (pd.Series(var, index=index).diff() < 0).apply(int)


def CROSS_STATUS(A, B):
    """
    A 穿过 B 产生持续的 1 序列信号
    """
    return np.where(A > B, 1, 0)


def FILTER(COND, N):
    k1 = pd.Series(np.where(COND, 1, 0), index=COND.index)
    idx = k1[k1 == 1].index.codes[0]
    needfilter = pd.Series(idx, index=idx)
    afterfilter = needfilter.diff().apply(lambda x: False if x < N else True)
    k1.iloc[afterfilter[afterfilter].index] = 2
    return k1.apply(lambda x: 1 if x == 2 else 0)


def COUNT(COND, N):
    """
    2018/05/23 修改

    参考https://github.com/QUANTAXIS/QUANTAXIS/issues/429

    现在返回的是series
    """
    return pd.Series(np.where(COND, 1, 0), index=COND.index).rolling(N).sum()


def IF(COND, V1, V2):
    var = np.where(COND, V1, V2)
    try:
        try:
            index = V1.index
        except:
            index = COND.index
    except:
        index = V2.index
    return pd.Series(var, index=index)


def IFAND(COND1, COND2, V1, V2):
    var = np.where(np.logical_and(COND1, COND2), V1, V2)
    return pd.Series(var, index=V1.index)


def IFOR(COND1, COND2, V1, V2):
    var = np.where(np.logical_or(COND1, COND2), V1, V2)
    return pd.Series(var, index=V1.index)


def REF(Series, N):
    return Series.shift(N)


def LAST(COND, N1, N2):
    """表达持续性
    从前N1日到前N2日一直满足COND条件

    Arguments:
        COND {[type]} -- [description]
        N1 {[type]} -- [description]
        N2 {[type]} -- [description]
    """
    N2 = 1 if N2 == 0 else N2
    assert N2 > 0
    assert N1 > N2
    return COND.iloc[-N1:-N2].all()


def STD(Series, N):
    return pd.Series.rolling(Series, N).std()


def AVEDEV(Series, N):
    """
    平均绝对偏差 mean absolute deviation
    修正: 2018-05-25

    之前用mad的计算模式依然返回的是单值
    """
    return Series.rolling(N).apply(lambda x: (np.abs(x - x.mean())).mean(), raw=True)


def MACD(Series, FAST, SLOW, MID):
    """macd指标 仅适用于Series
    对于DATAFRAME的应用请使用QA_indicator_macd
    """
    EMAFAST = EMA(Series, FAST)
    EMASLOW = EMA(Series, SLOW)
    DIFF = EMAFAST - EMASLOW
    DEA = EMA(DIFF, MID)
    MACD = (DIFF - DEA) * 2
    DICT = {'DIFF': DIFF, 'DEA': DEA, 'MACD': MACD}
    VAR = pd.DataFrame(DICT)
    return VAR


def BBIBOLL(Series, N1, N2, N3, N4, N, M):  # 多空布林线

    bbiboll = BBI(Series, N1, N2, N3, N4)
    UPER = bbiboll + M * STD(bbiboll, N)
    DOWN = bbiboll - M * STD(bbiboll, N)
    DICT = {'BBIBOLL': bbiboll, 'UPER': UPER, 'DOWN': DOWN}
    VAR = pd.DataFrame(DICT)
    return VAR


def BBI(Series, N1, N2, N3, N4):
    '多空指标'

    bbi = (MA(Series, N1) + MA(Series, N2) +
           MA(Series, N3) + MA(Series, N4)) / 4
    DICT = {'BBI': bbi}
    VAR = pd.DataFrame(DICT)
    return VAR


def BARLAST(cond, yes=True):
    """支持MultiIndex的cond和DateTimeIndex的cond
    条件成立  yes= True 或者 yes=1 根据不同的指标自己定

    最后一次条件成立  到 当前到周期数

    Arguments:
        cond {[type]} -- [description]
    """
    if isinstance(cond.index, pd.MultiIndex):
        return len(cond) - cond.index.levels[0].tolist().index(cond[cond == yes].index[-1][0]) - 1
    elif isinstance(cond.index, pd.DatetimeIndex):
        return len(cond) - cond.index.tolist().index(cond[cond == yes].index[-1]) - 1


def BARLAST_EXIST(cond, yes=True):
    """
    上一次条件成立   持续到当前到数量


    支持MultiIndex的cond和DateTimeIndex的cond
    条件成立  yes= True 或者 yes=1 根据不同的指标自己定

    Arguments:
        cond {[type]} -- [description]
    """
    if isinstance(cond.index, pd.MultiIndex):
        return len(cond) - cond.index.levels[0].tolist().index(cond[cond != yes].index[-1][0]) - 1
    elif isinstance(cond.index, pd.DatetimeIndex):
        return len(cond) - cond.index.tolist().index(cond[cond != yes].index[-1]) - 1


def XARROUND(x, y): return np.round(
    y * (round(x / y - math.floor(x / y) + 0.00000000001) + math.floor(x / y)), 2)


def RENKO(Series, N, condensed=True):
    last_price = Series[0]
    chart = [last_price]
    for price in Series:
        bricks = math.floor(abs(price - last_price) / N)
        if bricks == 0:
            if condensed:
                chart.append(chart[-1])
            continue
        sign = int(np.sign(price - last_price))
        chart += [sign * (last_price + (sign * N * x)) for x in range(1, bricks + 1)]
        last_price = abs(chart[-1])

    return pd.Series(chart)


def RENKOP(Series, N, condensed=True):
    last_price = Series[0]
    chart = [last_price]
    for price in Series:
        inc = (price - last_price) / last_price
        # print(inc)
        if abs(inc) < N:
            # if condensed:
            #     chart.append(chart[-1])
            continue

        sign = int(np.sign(price - last_price))
        bricks = math.floor(inc / N)
        # print(bricks)
        # print((N * (price-last_price)) / inc)
        step = math.floor((N * (price - last_price)) / inc)
        print(step)
        # print(sign)
        chart += [sign * (last_price + (sign * step * x))
                  for x in range(1, abs(bricks) + 1)]
        last_price = abs(chart[-1])
    return pd.Series(chart)



"""
DataFrame 类

以下的函数都可以被直接add_func


"""

"""
1.	趋向指标 
又叫趋势跟踪类指标,主要用于跟踪并预测股价的发展趋势

包含的主要指标
1. 移动平均线 MA
2. 指数平滑移动平均线 MACD
3. 趋向指标 DMI
4. 瀑布线 PBX
5. 平均线差 DMA
6. 动力指标(动量线)  MTM
7. 指数平均线 EXPMA
8. 佳庆指标 CHO
"""

def QA_indicator_MA(DataFrame, *args, **kwargs):
    """MA

    Arguments:
        DataFrame {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    CLOSE = DataFrame['close']
    return pd.DataFrame({'MA{}'.format(N): MA(CLOSE, N) for N in list(args)})

def QA_indicator_MA_VOL(DataFrame, *args, **kwargs):
    """MA_VOLU

    Arguments:
        DataFrame {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    VOL = DataFrame['volume']
    return pd.DataFrame({'MA_VOL{}'.format(N): MA(VOL, N) for N in list(args)})

def QA_indicator_EMA(DataFrame, *args):
    CLOSE = DataFrame['close']
    return pd.DataFrame({'EMA{}'.format(N): EMA(CLOSE, N) for N in list(args)})


def QA_indicator_SMA(DataFrame, *args):
    CLOSE = DataFrame['close']
    return pd.DataFrame({'SMA{}'.format(N): SMA(CLOSE, N) for N in list(args)})


def QA_indicator_MACD(DataFrame, short=12, long=26, mid=9):
    """
    MACD CALC
    """
    CLOSE = DataFrame['close']

    DIF = EMA(CLOSE, short) - EMA(CLOSE, long)
    DEA = EMA(DIF, mid)
    MACD = (DIF - DEA) * 2

    return pd.DataFrame({'DIF': DIF, 'DEA': DEA, 'MACD': MACD})


def QA_indicator_DMI(DataFrame, M1=14, M2=6):
    """
    趋向指标 DMI
    """
    HIGH = DataFrame.high
    LOW = DataFrame.low
    CLOSE = DataFrame.close
    OPEN = DataFrame.open

    TR = SUM(MAX(MAX(HIGH - LOW, ABS(HIGH - REF(CLOSE, 1))),
                 ABS(LOW - REF(CLOSE, 1))), M1)
    HD = HIGH - REF(HIGH, 1)
    LD = REF(LOW, 1) - LOW
    DMP = SUM(IFAND(HD > 0, HD > LD, HD, 0), M1)
    DMM = SUM(IFAND(LD > 0, LD > HD, LD, 0), M1)
    DI1 = DMP * 100 / TR
    DI2 = DMM * 100 / TR
    ADX = MA(ABS(DI2 - DI1) / (DI1 + DI2) * 100, M2)
    ADXR = (ADX + REF(ADX, M2)) / 2

    return pd.DataFrame({
        'DI1': DI1, 'DI2': DI2,
        'ADX': ADX, 'ADXR': ADXR
    })


def QA_indicator_PBX(DataFrame, N1=3, N2=5, N3=8, N4=13, N5=18, N6=24):
    '瀑布线'
    C = DataFrame['close']
    PBX1 = (EMA(C, N1) + EMA(C, 2 * N1) + EMA(C, 4 * N1)) / 3
    PBX2 = (EMA(C, N2) + EMA(C, 2 * N2) + EMA(C, 4 * N2)) / 3
    PBX3 = (EMA(C, N3) + EMA(C, 2 * N3) + EMA(C, 4 * N3)) / 3
    PBX4 = (EMA(C, N4) + EMA(C, 2 * N4) + EMA(C, 4 * N4)) / 3
    PBX5 = (EMA(C, N5) + EMA(C, 2 * N5) + EMA(C, 4 * N5)) / 3
    PBX6 = (EMA(C, N6) + EMA(C, 2 * N6) + EMA(C, 4 * N6)) / 3
    DICT = {'PBX1': PBX1, 'PBX2': PBX2, 'PBX3': PBX3,
            'PBX4': PBX4, 'PBX5': PBX5, 'PBX6': PBX6}

    return pd.DataFrame(DICT)


def QA_indicator_DMA(DataFrame, M1=10, M2=50, M3=10):
    """
    平均线差 DMA
    """
    CLOSE = DataFrame.close
    DDD = MA(CLOSE, M1) - MA(CLOSE, M2)
    AMA = MA(DDD, M3)
    return pd.DataFrame({
        'DDD': DDD, 'AMA': AMA
    })


def QA_indicator_MTM(DataFrame, N=12, M=6):
    '动量线'
    C = DataFrame.close
    mtm = C - REF(C, N)
    MTMMA = MA(mtm, M)
    DICT = {'MTM': mtm, 'MTMMA': MTMMA}

    return pd.DataFrame(DICT)


# def QA_indicator_EXPMA(DataFrame, P1=5, P2=10, P3=20, P4=60):
#     """ 指数平均线 EXPMA"""
#     CLOSE = DataFrame.close
#     MA1 = EMA(CLOSE, P1)
#     MA2 = EMA(CLOSE, P2)
#     MA3 = EMA(CLOSE, P3)
#     MA4 = EMA(CLOSE, P4)
#     return pd.DataFrame({
#         'MA1': MA1, 'MA2': MA2, 'MA3': MA3, 'MA4': MA4
#     })


def QA_indicator_CHO(DataFrame, N1=10, N2=20, M=6):
    """
    佳庆指标 CHO
    """
    HIGH = DataFrame.high
    LOW = DataFrame.low
    CLOSE = DataFrame.close
    VOL = DataFrame.volume
    MID = SUM(VOL * (2 * CLOSE - HIGH - LOW) / (HIGH + LOW), 0)
    CHO = MA(MID, N1) - MA(MID, N2)
    MACHO = MA(CHO, M)
    return pd.DataFrame({
        'CHO': CHO, 'MACHO': MACHO
    })


"""

2.	反趋向指标
主要捕捉趋势的转折点

随机指标KDJ
乖离率 BIAS
变动速率 ROC
顺势指标 CCI
威廉指标 W&R
震荡量(变动速率) OSC
相对强弱指标 RSI
动态买卖指标 ADTM

"""


def QA_indicator_KDJ(DataFrame, N=9, M1=3, M2=3):
    C = DataFrame['close']
    H = DataFrame['high']
    L = DataFrame['low']

    # RSV = ((C - LLV(L, N)) / (HHV(H, N) - LLV(L, N)) * 100).groupby('symbol').fillna(method='ffill') # code
    RSV = ((C - LLV(L, N)) / (HHV(H, N) - LLV(L, N)) * 100).ffill() # code
    K = SMA(RSV, M1)
    D = SMA(K, M2)
    J = 3 * K - 2 * D
    DICT = {'KDJ_K': K, 'KDJ_D': D, 'KDJ_J': J}
    return pd.DataFrame(DICT)


def QA_indicator_BIAS(DataFrame, N1, N2, N3):
    '乖离率'
    CLOSE = DataFrame['close']
    BIAS1 = (CLOSE - MA(CLOSE, N1)) / MA(CLOSE, N1) * 100
    BIAS2 = (CLOSE - MA(CLOSE, N2)) / MA(CLOSE, N2) * 100
    BIAS3 = (CLOSE - MA(CLOSE, N3)) / MA(CLOSE, N3) * 100
    DICT = {'BIAS1': BIAS1, 'BIAS2': BIAS2, 'BIAS3': BIAS3}

    return pd.DataFrame(DICT)


def QA_indicator_ROC(DataFrame, N=12, M=6):
    '变动率指标'
    C = DataFrame['close']
    roc = 100 * (C - REF(C, N)) / REF(C, N)
    ROCMA = MA(roc, M)
    DICT = {'ROC': roc, 'ROCMA': ROCMA}

    return pd.DataFrame(DICT)


def QA_indicator_CCI(DataFrame, N=14):
    """
    TYP:=(HIGH+LOW+CLOSE)/3;
    CCI:(TYP-MA(TYP,N))/(0.015*AVEDEV(TYP,N));
    """
    typ = (DataFrame['high'] + DataFrame['low'] + DataFrame['close']) / 3
    ## 此处AVEDEV可能为0值  因此导致出错 +0.0000000000001
    cci = ((typ - MA(typ, N)) / (0.015 * AVEDEV(typ, N) + 0.00000001))
    a = 100
    b = -100

    return pd.DataFrame({
        'CCI': cci, 'CCI-a': a, 'CCI-b': b
    })



def QA_indicator_WR(DataFrame, N, N1):
    '威廉指标'
    HIGH = DataFrame['high']
    LOW = DataFrame['low']
    CLOSE = DataFrame['close']
    WR1 = 100 * (HHV(HIGH, N) - CLOSE) / (HHV(HIGH, N) - LLV(LOW, N))
    WR2 = 100 * (HHV(HIGH, N1) - CLOSE) / (HHV(HIGH, N1) - LLV(LOW, N1))
    DICT = {'WR1': WR1, 'WR2': WR2}

    return pd.DataFrame(DICT)


def QA_indicator_OSC(DataFrame, N=20, M=6):
    """变动速率线

    震荡量指标OSC，也叫变动速率线。属于超买超卖类指标,是从移动平均线原理派生出来的一种分析指标。

    它反应当日收盘价与一段时间内平均收盘价的差离值,从而测出股价的震荡幅度。

    按照移动平均线原理，根据OSC的值可推断价格的趋势，如果远离平均线，就很可能向平均线回归。
    """
    C = DataFrame['close']
    OS = (C - MA(C, N)) * 100
    MAOSC = EMA(OS, M)
    DICT = {'OSC': OS, 'MAOSC': MAOSC}

    return pd.DataFrame(DICT)


def QA_indicator_RSI(DataFrame, N1=12, N2=26, N3=9):
    '相对强弱指标RSI1:SMA(MAX(CLOSE-LC,0),N1,1)/SMA(ABS(CLOSE-LC),N1,1)*100;'
    CLOSE = DataFrame['close']
    LC = REF(CLOSE, 1)
    RSI1 = SMA(MAX(CLOSE - LC, 0), N1) / SMA(ABS(CLOSE - LC), N1) * 100
    RSI2 = SMA(MAX(CLOSE - LC, 0), N2) / SMA(ABS(CLOSE - LC), N2) * 100
    RSI3 = SMA(MAX(CLOSE - LC, 0), N3) / SMA(ABS(CLOSE - LC), N3) * 100
    DICT = {'RSI1': RSI1, 'RSI2': RSI2, 'RSI3': RSI3}

    return pd.DataFrame(DICT)


def QA_indicator_ADTM(DataFrame, N=23, M=8):
    '动态买卖气指标'
    HIGH = DataFrame.high
    LOW = DataFrame.low
    OPEN = DataFrame.open
    DTM = IF(OPEN > REF(OPEN, 1), MAX((HIGH - OPEN), (OPEN - REF(OPEN, 1))), 0)
    DBM = IF(OPEN < REF(OPEN, 1), MAX((OPEN - LOW), (OPEN - REF(OPEN, 1))), 0)
    STM = SUM(DTM, N)
    SBM = SUM(DBM, N)
    ADTM1 = IF(STM > SBM, (STM - SBM) / STM,
               IF(STM != SBM, (STM - SBM) / SBM, 0))
    MAADTM = MA(ADTM1, M)
    DICT = {'ADTM': ADTM1, 'MAADTM': MAADTM}

    return pd.DataFrame(DICT)


"""
3.	量能指标
通过成交量的大小和变化研判趋势变化
容量指标 VR
量相对强弱 VRSI
能量指标 CR
人气意愿指标 ARBR
成交量标准差 VSTD"""


def QA_indicator_VR(DataFrame, M1=26, M2=100, M3=200):
    VOL = DataFrame.volume
    CLOSE = DataFrame.close
    LC = REF(CLOSE, 1)
    VR = SUM(IF(CLOSE > LC, VOL, 0), M1) / SUM(IF(CLOSE <= LC, VOL, 0), M1) * 100
    a = M2
    b = M3
    return pd.DataFrame({
        'VR': VR, 'VR-a': a, 'VR-b': b
    })


def QA_indicator_VRSI(DataFrame, N=6):
    VOL = DataFrame.volume
    vrsi = SMA(MAX(VOL - REF(VOL, 1), 0), N, 1) / \
           SMA(ABS(VOL - REF(VOL, 1)), N, 1) * 100

    return pd.DataFrame({'VRSI': vrsi})


def QA_indicator_CR(DataFrame, N=26, M1=5, M2=10, M3=20):
    HIGH = DataFrame.high
    LOW = DataFrame.low
    CLOSE = DataFrame.close
    VOL = DataFrame.volume
    MID = (HIGH + LOW + CLOSE) / 3

    CR = SUM(MAX(0, HIGH - REF(MID, 1)), N) / SUM(MAX(0, REF(MID, 1) - LOW), N) * 100
    MA1 = REF(MA(CR, M1), int(M1 / 2.5 + 1))
    MA2 = REF(MA(CR, M2), int(M2 / 2.5 + 1))
    MA3 = REF(MA(CR, M3), int(M3 / 2.5 + 1))
    return pd.DataFrame({
        'CR': CR, 'CR-MA1': MA1, 'CR-MA2': MA2, 'CR-MA3': MA3
    })


def QA_indicator_ARBR(DataFrame, M1=26, M2=70, M3=150):
    HIGH = DataFrame.high
    LOW = DataFrame.low
    CLOSE = DataFrame.close
    OPEN = DataFrame.open
    AR = SUM(HIGH - OPEN, M1) / SUM(OPEN - LOW, M1) * 100
    BR = SUM(MAX(0, HIGH - REF(CLOSE, 1)), M1) / \
         SUM(MAX(0, REF(CLOSE, 1) - LOW), M1) * 100
    a = M2
    b = M3
    return pd.DataFrame({
        'AR': AR, 'BR': BR, 'ARBR-a': a, 'ARBR-b': b
    })


def QA_indicator_VSTD(DataFrame, N=10):
    VOL = DataFrame.volume
    vstd = STD(VOL, N)
    return pd.DataFrame({'VSTD': vstd})


"""
4.	量价指标
通过成交量和股价变动关系分析未来趋势
震荡升降指标ASI
价量趋势PVT
能量潮OBV
量价趋势VPT
"""


def QA_indicator_ASI(DataFrame, M1=26, M2=10):
    """
    LC=REF(CLOSE,1);
    AA=ABS(HIGH-LC);
    BB=ABS(LOW-LC);
    CC=ABS(HIGH-REF(LOW,1));
    DD=ABS(LC-REF(OPEN,1));
    R=IF(AA>BB AND AA>CC,AA+BB/2+DD/4,IF(BB>CC AND BB>AA,BB+AA/2+DD/4,CC+DD/4));
    X=(CLOSE-LC+(CLOSE-OPEN)/2+LC-REF(OPEN,1));
    SI=16*X/R*MAX(AA,BB);
    ASI:SUM(SI,M1);
    ASIT:MA(ASI,M2);
    """
    CLOSE = DataFrame['close']
    HIGH = DataFrame['high']
    LOW = DataFrame['low']
    OPEN = DataFrame['open']
    LC = REF(CLOSE, 1)
    AA = ABS(HIGH - LC)
    BB = ABS(LOW - LC)
    CC = ABS(HIGH - REF(LOW, 1))
    DD = ABS(LC - REF(OPEN, 1))

    R = IFAND(AA > BB, AA > CC, AA + BB / 2 + DD / 4,
              IFAND(BB > CC, BB > AA, BB + AA / 2 + DD / 4, CC + DD / 4))
    X = (CLOSE - LC + (CLOSE - OPEN) / 2 + LC - REF(OPEN, 1))
    SI = 16 * X / R * MAX(AA, BB)
    ASI = SUM(SI, M1)
    ASIT = MA(ASI, M2)
    return pd.DataFrame({
        'ASI': ASI, 'ASIT': ASIT
    })


def QA_indicator_PVT(DataFrame):
    CLOSE = DataFrame.close
    VOL = DataFrame.volume
    PVT = SUM((CLOSE - REF(CLOSE, 1)) / REF(CLOSE, 1) * VOL, 0)
    return pd.DataFrame({'PVT': PVT})


def QA_indicator_OBV(DataFrame):
    """能量潮"""
    VOL = DataFrame.volume
    CLOSE = DataFrame.close
    return pd.DataFrame({
        'OBV': np.cumsum(IF(CLOSE > REF(CLOSE, 1), VOL, IF(CLOSE < REF(CLOSE, 1), -VOL, 0))) / 10000
    })


def QA_indicator_VPT(DataFrame, N=51, M=6):
    VOL = DataFrame.volume
    CLOSE = DataFrame.close
    VPT = SUM(VOL * (CLOSE - REF(CLOSE, 1)) / REF(CLOSE, 1), 0)
    MAVPT = MA(VPT, M)
    return pd.DataFrame({
        'VPT': VPT, 'MAVPT': MAVPT
    })


"""
5.	压力支撑指标
主要用于分析股价目前收到的压力和支撑
布林带 BOLL
麦克指标 MIKE
"""


def QA_indicator_BOLL(DataFrame, N=20, P=2):
    '布林线'
    C = DataFrame['close']
    boll = MA(C, N)
    UB = boll + P * STD(C, N)
    LB = boll - P * STD(C, N)
    DICT = {'BOLL': boll, 'UB': UB, 'LB': LB}

    return pd.DataFrame(DICT)


def QA_indicator_MIKE(DataFrame, N=12):
    """
    MIKE指标
    指标说明
    MIKE是另外一种形式的路径指标。
    买卖原则
    1  WEAK-S，MEDIUM-S，STRONG-S三条线代表初级、中级、强力支撑。
    2  WEAK-R，MEDIUM-R，STRONG-R三条线代表初级、中级、强力压力。
    """
    HIGH = DataFrame.high
    LOW = DataFrame.low
    CLOSE = DataFrame.close

    TYP = (HIGH + LOW + CLOSE) / 3
    LL = LLV(LOW, N)
    HH = HHV(HIGH, N)

    WR = TYP + (TYP - LL)
    MR = TYP + (HH - LL)
    SR = 2 * HH - LL
    WS = TYP - (HH - TYP)
    MS = TYP - (HH - LL)
    SS = 2 * LL - HH
    return pd.DataFrame({
        'WR': WR, 'MR': MR, 'SR': SR,
        'WS': WS, 'MS': MS, 'SS': SS
    })


def QA_indicator_BBI(DataFrame, N1=3, N2=6, N3=12, N4=24):
    '多空指标'
    C = DataFrame['close']
    bbi = (MA(C, N1) + MA(C, N2) + MA(C, N3) + MA(C, N4)) / 4
    DICT = {'BBI': bbi}

    return pd.DataFrame(DICT)


def QA_indicator_MFI(DataFrame, N=14):
    """
    资金指标
    TYP := (HIGH + LOW + CLOSE)/3;
    V1:=SUM(IF(TYP>REF(TYP,1),TYP*VOL,0),N)/SUM(IF(TYP<REF(TYP,1),TYP*VOL,0),N);
    MFI:100-(100/(1+V1));
    赋值: (最高价 + 最低价 + 收盘价)/3
    V1赋值:如果TYP>1日前的TYP,返回TYP*成交量(手),否则返回0的N日累和/如果TYP<1日前的TYP,返回TYP*成交量(手),否则返回0的N日累和
    输出资金流量指标:100-(100/(1+V1))
    """
    C = DataFrame['close']
    H = DataFrame['high']
    L = DataFrame['low']
    VOL = DataFrame['volume']
    TYP = (C + H + L) / 3
    V1 = SUM(IF(TYP > REF(TYP, 1), TYP * VOL, 0), N) / \
         SUM(IF(TYP < REF(TYP, 1), TYP * VOL, 0), N)
    mfi = 100 - (100 / (1 + V1))
    DICT = {'MFI': mfi}

    return pd.DataFrame(DICT)


def QA_indicator_ATR(DataFrame, N=14):
    """
    输出TR:(最高价-最低价)和昨收-最高价的绝对值的较大值和昨收-最低价的绝对值的较大值
    输出真实波幅:TR的N日简单移动平均
    算法：今日振幅、今日最高与昨收差价、今日最低与昨收差价中的最大值，为真实波幅，求真实波幅的N日移动平均

    参数：N　天数，一般取14

    """
    C = DataFrame['close']
    H = DataFrame['high']
    L = DataFrame['low']
    TR = MAX(MAX((H - L), ABS(REF(C, 1) - H)), ABS(REF(C, 1) - L))
    atr = MA(TR, N)
    return pd.DataFrame({'TR': TR, 'ATR': atr})


def QA_indicator_SKDJ(DataFrame, N=9, M=3):
    """
    1.指标>80 时，回档机率大；指标<20 时，反弹机率大；
    2.K在20左右向上交叉D时，视为买进信号参考；
    3.K在80左右向下交叉D时，视为卖出信号参考；
    4.SKDJ波动于50左右的任何讯号，其作用不大。

    """
    CLOSE = DataFrame['close']
    LOWV = LLV(DataFrame['low'], N)
    HIGHV = HHV(DataFrame['high'], N)
    RSV = EMA((CLOSE - LOWV) / (HIGHV - LOWV) * 100, M)
    K = EMA(RSV, M)
    D = MA(K, M)
    DICT = {'RSV': RSV, 'SKDJ_K': K, 'SKDJ_D': D}

    return pd.DataFrame(DICT)


def QA_indicator_DDI(DataFrame, N=13, N1=26, M=1, M1=5):
    """
    '方向标准离差指数'
    分析DDI柱状线，由红变绿(正变负)，卖出信号参考；由绿变红，买入信号参考。
    """

    H = DataFrame['high']
    L = DataFrame['low']
    DMZ = IF((H + L) > (REF(H, 1) + REF(L, 1)),
             MAX(ABS(H - REF(H, 1)), ABS(L - REF(L, 1))), 0)
    DMF = IF((H + L) < (REF(H, 1) + REF(L, 1)),
             MAX(ABS(H - REF(H, 1)), ABS(L - REF(L, 1))), 0)
    DIZ = SUM(DMZ, N) / (SUM(DMZ, N) + SUM(DMF, N))
    DIF = SUM(DMF, N) / (SUM(DMF, N) + SUM(DMZ, N))
    ddi = DIZ - DIF
    ADDI = SMA(ddi, N1, M)
    AD = MA(ADDI, M1)
    DICT = {'DDI': ddi, 'ADDI': ADDI, 'AD': AD}

    return pd.DataFrame(DICT)


def QA_indicator_shadow(DataFrame):
    """
    上下影线指标
    """
    return pd.DataFrame({
        'LOW': lower_shadow(DataFrame), 'UP': upper_shadow(DataFrame),
        'BODY': body(DataFrame), 'BODY_ABS': body_abs(DataFrame), 'PRICE_PCG': price_pcg(DataFrame)
    })


def lower_shadow(DataFrame):  # 下影线
    return abs(DataFrame.low - MIN(DataFrame.open, DataFrame.close))


def upper_shadow(DataFrame):  # 上影线
    return abs(DataFrame.high - MAX(DataFrame.open, DataFrame.close))


def body_abs(DataFrame):
    return abs(DataFrame.open - DataFrame.close)


def body(DataFrame):
    return DataFrame.close - DataFrame.open


def price_pcg(DataFrame):
    return body(DataFrame) / DataFrame.open


def amplitude(DataFrame):
    return (DataFrame.high - DataFrame.low) / DataFrame.low


"""

6.	大盘指标
通过涨跌家数研究大盘指数的走势
涨跌比率 ADR
绝对幅度指标 ABI
新三价率 TBR
腾落指数 ADL
广量冲力指标
指数平滑广量 STIX
"""

def get_s_tdays(start_date,end_date,period='d',exchange='001002',is_calendar = False):
    url = 'http://192.168.195.214:4444/get_s_tdays/'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    data =  {
        "start_date": {
            "date": start_date
        },
        "end_date": {
            "date": end_date
        },
        "period": {
            "type": period
        },
        "exchange": {
            "type": exchange
        },
        "is_calendar": {
            "bool_type": is_calendar
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        print("Request successful.")
        return response.json()
    else:
        print("Request failed with status code:", response.status_code)
        return False

def get_index_indexcomponent(symbols):
    url = 'http://192.168.195.214:4444/get_index_indexcomponent/'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    data =  {
        "symbols":symbols
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        print("Request successful.")
        return response.json()
    else:
        print("Request failed with status code:", response.status_code)
        return False


def get_s_price_pchg_d(code,trade_date,count_type):
    url = 'http://192.168.195.214:4444/get_s_price_pchg_d/'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    data =  {
        "code": {
            "symbols": code
        },
        "trade_date": {
            "dates": trade_date
        },
        "count_type": {
            "type": count_type
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        print("Request successful.")
        return response.json()
    else:
        print("Request failed with status code:", response.status_code)
        return False

def get_s_price_amount_d(code,trade_date,count_type):
    url = 'http://192.168.195.214:4444/get_s_price_amount_d/'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    data =  {
        "code": {
            "symbols": code
        },
        "trade_date": {
            "dates": trade_date
        },
        "count_type": {
            "type": count_type
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        print("Request successful.")
        return response.json()
    else:
        print("Request failed with status code:", response.status_code)
        return False


def get_s_price_close_d(code,trade_date,adjust_type,count_type):
    url = 'http://192.168.195.214:4444/get_s_price_close_d/'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    data =  {
        "code": {
            "symbols": code
        },
        "trade_date": {
            "dates": trade_date
        },
        "adjust_type": {
            "type": adjust_type
        },
        "count_type": {
            "type": count_type
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        print("Request successful.")
        return response.json()
    else:
        print("Request failed with status code:", response.status_code)
        return False

def get_s_price_open_d(code,trade_date,adjust_type,count_type):
    url = 'http://192.168.195.214:4444/get_s_price_open_d/'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    data =  {
        "code": {
            "symbols": code
        },
        "trade_date": {
            "dates": trade_date
        },
        "adjust_type": {
            "type": adjust_type
        },
        "count_type": {
            "type": count_type
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        print("Request successful.")
        return response.json()
    else:
        print("Request failed with status code:", response.status_code)
        return False


def get_s_price_high_d(code,trade_date,adjust_type,count_type):
    url = 'http://192.168.195.214:4444/get_s_price_high_d/'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    data =  {
        "code": {
            "symbols": code
        },
        "trade_date": {
            "dates": trade_date
        },
        "adjust_type": {
            "type": adjust_type
        },
        "count_type": {
            "type": count_type
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        print("Request successful.")
        return response.json()
    else:
        print("Request failed with status code:", response.status_code)
        return False

def get_s_price_low_d(code,trade_date,adjust_type,count_type):
    url = 'http://192.168.195.214:4444/get_s_price_low_d/'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    data =  {
        "code": {
            "symbols": code
        },
        "trade_date": {
            "dates": trade_date
        },
        "adjust_type": {
            "type": adjust_type
        },
        "count_type": {
            "type": count_type
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        print("Request successful.")
        return response.json()
    else:
        print("Request failed with status code:", response.status_code)
        return False

def get_s_price_vol_d(code,trade_date,adjust_type,count_type):
    url = 'http://192.168.195.214:4444/get_s_price_vol_d/'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    data =  {
        "code": {
            "symbols": code
        },
        "trade_date": {
            "dates": trade_date
        },
        "adjust_type": {
            "type": adjust_type
        },
        "count_type": {
            "type": count_type
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        print("Request successful.")
        return response.json()
    else:
        print("Request failed with status code:", response.status_code)
        return False

def get_Astocks():
    """
    取出A股成分股
    """
    index_component = get_index_indexcomponent(symbols=["000001", "399107", "000688"])  # symbols 可以为数组 A股交易代码
    symbols = index_component['symbol']
    return symbols

def get_quote_datas(code,trade_date):
    """
    Api获取A股基本行情指标
    """
    results = []
    # field = ["pchg", "amount", "close", "open", "high", "low"]
    lows = get_s_price_low_d(code = code,trade_date = trade_date,adjust_type = 'B',count_type = '1')
    highs = get_s_price_high_d(code = code,trade_date = trade_date,adjust_type = 'B',count_type = '1')
    opens = get_s_price_open_d(code = code,trade_date = trade_date,adjust_type = 'B',count_type = '1')
    closes = get_s_price_close_d(code = code,trade_date = trade_date,adjust_type = 'B',count_type = '1')
    vols = get_s_price_vol_d(code = code,trade_date = trade_date,adjust_type = 'B',count_type = '1')

    # amounts = get_s_price_amount_d(code = code,trade_date = trade_date,count_type = '1')
    # pchgs = get_s_price_pchg_d(code = code,trade_date = trade_date,count_type = '1')

    lows = dict(lows)
    lows['field_name'] = 'low'
    highs = dict(highs)
    highs['field_name'] = 'high'
    opens = dict(opens)
    opens['field_name'] = 'open'
    closes = dict(closes)
    closes['field_name'] = 'close'
    vols = dict(vols)
    vols['field_name'] = 'volume'
    results.extend([lows, highs, opens, closes, vols])
    # amounts = dict(amounts)
    # amounts['field_name'] = 'amount'
    # pchgs = dict(pchgs)
    # pchgs['field_name'] = 'pchg'
    # results.extend([lows,highs,opens,closes,amounts,pchgs])

    merged_crosssection_data = pd.DataFrame()
    merged_temporal_data = pd.DataFrame()
    for result in results:
        symbol_colname = result['symbol_colname']
        date_colname = result['date_colname']
        value_colname = result['value_colname']
        field_name = result['field_name']
        df = pd.DataFrame(result['data_frame'])

        if ',' in field_name:
            print(',' * 10)
        else:
            # Rename symbol_colname and date_colname to fixed values SYMBOL and DATE
            df = df.rename(columns={symbol_colname: 'symbol', date_colname: 'date', value_colname: field_name})

        # Convert value_colname to numeric type
        if result['value_coltype'] == 'number':
            df[field_name] = pd.to_numeric(df[field_name])
        # Merge data
        if 'date' in df.columns.tolist():
            if merged_temporal_data.empty:
                merged_temporal_data = df
            else:
                merged_temporal_data = pd.merge(merged_temporal_data, df, on=['symbol', 'date'], how='outer')
        else:
            if merged_crosssection_data.empty:
                merged_crosssection_data = df
            else:
                merged_crosssection_data = pd.merge(merged_crosssection_data, df, on=['symbol'], how='outer')

    if merged_temporal_data.empty or merged_crosssection_data.empty:
        if merged_temporal_data.empty:
            return merged_crosssection_data.fillna(np.nan)
        else:
            return merged_temporal_data.fillna(np.nan)
    else:
        merged_temporal_data = pd.merge(merged_temporal_data, merged_crosssection_data, on=['symbol'], how='outer')
        return merged_temporal_data.fillna(np.nan)

def build_indicator_MA(DataFrame):
    """
    平均线
    指标计算外层封装
    input按标的拆分
    output按标的组合
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_MA(group_df,5,10,20,60)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_MA_VOL(DataFrame):
    """
    平均成交量
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_MA_VOL(group_df,5,10,20,60)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_EMA(DataFrame):
    """
    EMA
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_EMA(group_df,5,10,20,60)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_SMA(DataFrame):
    """
    SMA
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_SMA(group_df,5,10,20,60)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic


def build_indicator_MACD(DataFrame):
    """
    MACD
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_MACD(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_DMI(DataFrame):
    """
    DMI
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_DMI(group_df)
        # df_symbol_indic = df_symbol_indic.diff() # 取差值 之后再累积
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic


def build_indicator_PBX(DataFrame):
    """
    DMI
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_PBX(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_DMA(DataFrame):
    """
    DMA
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_DMA(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_MTM(DataFrame):
    """
    '动量线'
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_MTM(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_CHO(DataFrame):
    """
    DMA
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_CHO(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_KDJ(DataFrame):
    """
    KDJ
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_KDJ(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_BIAS(DataFrame):
    """
    KDJ
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_BIAS(group_df,6,12,24) # BIAS指标有三条指标线，N的参数一般设置为6日、12日、24日。
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_ROC(DataFrame):
    """
    变动率指标
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_ROC(group_df) # BIAS指标有三条指标线，N的参数一般设置为6日、12日、24日。
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_CCI(DataFrame):
    """
    CCI
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_CCI(group_df) # BIAS指标有三条指标线，N的参数一般设置为6日、12日、24日。
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_WR(DataFrame):
    """
    威廉指标
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_WR(group_df,14,30) # BIAS指标有三条指标线，N的参数一般设置为6日、12日、24日。
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic


def build_indicator_OSC(DataFrame):
    """

    震荡量指标OSC，也叫变动速率线。属于超买超卖类指标,是从移动平均线原理派生出来的一种分析指标。

    它反应当日收盘价与一段时间内平均收盘价的差离值,从而测出股价的震荡幅度。

    按照移动平均线原理，根据OSC的值可推断价格的趋势，如果远离平均线，就很可能向平均线回归。
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_OSC(group_df) # BIAS指标有三条指标线，N的参数一般设置为6日、12日、24日。
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_RSI(DataFrame):
    """
    '相对强弱指标RSI1:SMA(MAX(CLOSE-LC,0),N1,1)/SMA(ABS(CLOSE-LC),N1,1)*100;'
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_RSI(group_df) # BIAS指标有三条指标线，N的参数一般设置为6日、12日、24日。
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic


def build_indicator_ADTM(DataFrame):
    """
    '动态买卖气指标'
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_ADTM(group_df) # BIAS指标有三条指标线，N的参数一般设置为6日、12日、24日。
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic


def build_indicator_VR(DataFrame):
    """
    容量指标 VR
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_VR(group_df) # BIAS指标有三条指标线，N的参数一般设置为6日、12日、24日。
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic


def build_indicator_VRSI(DataFrame):
    """
    VRSI
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_VRSI(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic


def build_indicator_ARBR(DataFrame):
    """
    ARBR
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_ARBR(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic


def build_indicator_CR(DataFrame):
    """
    CR
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_CR(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_VSTD(DataFrame):
    """
    VSTD
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_VSTD(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic


def build_indicator_ASI(DataFrame):
    """
    ASI
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_ASI(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_PVT(DataFrame):
    """
    PVT
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_PVT(group_df)
        df_symbol_indic = df_symbol_indic.diff()
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic


def build_indicator_OBV(DataFrame):
    """
    OBV
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_OBV(group_df)
        df_symbol_indic = df_symbol_indic.diff()
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_VPT(DataFrame):
    """
    VPT
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_VPT(group_df)
        df_symbol_indic = df_symbol_indic.diff()
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic


def build_indicator_BOLL(DataFrame):
    """
    布林线
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_BOLL(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_MIKE(DataFrame):
    """
    MIKE指标
    指标说明
    MIKE是另外一种形式的路径指标。
    买卖原则
    1  WEAK-S，MEDIUM-S，STRONG-S三条线代表初级、中级、强力支撑。
    2  WEAK-R，MEDIUM-R，STRONG-R三条线代表初级、中级、强力压力。
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_MIKE(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_BBI(DataFrame):
    """
    '多空指标'
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_BBI(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_MFI(DataFrame):
    """
    资金指标
    TYP := (HIGH + LOW + CLOSE)/3;
    V1:=SUM(IF(TYP>REF(TYP,1),TYP*VOL,0),N)/SUM(IF(TYP<REF(TYP,1),TYP*VOL,0),N);
    MFI:100-(100/(1+V1));
    赋值: (最高价 + 最低价 + 收盘价)/3
    V1赋值:如果TYP>1日前的TYP,返回TYP*成交量(手),否则返回0的N日累和/如果TYP<1日前的TYP,返回TYP*成交量(手),否则返回0的N日累和
    输出资金流量指标:100-(100/(1+V1))
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_MFI(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic


def build_indicator_ATR(DataFrame):
    """
    输出TR:(最高价-最低价)和昨收-最高价的绝对值的较大值和昨收-最低价的绝对值的较大值
    输出真实波幅:TR的N日简单移动平均
    算法：今日振幅、今日最高与昨收差价、今日最低与昨收差价中的最大值，为真实波幅，求真实波幅的N日移动平均

    参数：N　天数，一般取14
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_ATR(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic


def build_indicator_SKDJ(DataFrame):
    """
    输出TR:(最高价-最低价)和昨收-最高价的绝对值的较大值和昨收-最低价的绝对值的较大值
    输出真实波幅:TR的N日简单移动平均
    算法：今日振幅、今日最高与昨收差价、今日最低与昨收差价中的最大值，为真实波幅，求真实波幅的N日移动平均

    参数：N　天数，一般取14
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_SKDJ(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_DDI(DataFrame):
    """
    '方向标准离差指数'
    分析DDI柱状线，由红变绿(正变负)，卖出信号参考；由绿变红，买入信号参考。
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_DDI(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic

def build_indicator_shadow(DataFrame):
    """
    '方向标准离差指数'
    分析DDI柱状线，由红变绿(正变负)，卖出信号参考；由绿变红，买入信号参考。
    """
    symbols = DataFrame['symbol'].unique()  # 把一列所有取值提出来成为一个Array
    df_indic = pd.DataFrame()
    for symbol in symbols:
        group_df = DataFrame[DataFrame['symbol'].isin([symbol])]  # 按照列的某个值取一个dataframe
        df_symbol_indic =  QA_indicator_shadow(group_df)
        df_symbol_indic['symbol'] = symbol
        df_symbol_indic['date'] = group_df['date']
        df_symbol_indic = df_symbol_indic.tail(1)
        df_indic = pd.concat([df_indic, df_symbol_indic], ignore_index=True)
    return df_indic


def stock_indic_flow():
    symbols = get_Astocks()
    trade_date = get_s_tdays(start_date="20240601", end_date="20240903")
    trade_date = trade_date['date']
    quotes = get_quote_datas(code = symbols,trade_date = trade_date)
    # df_indic = build_indicator_MA(quotes)
    # df_indic = build_indicator_MA_VOL(quotes)
    # df_indic = build_indicator_EMA(quotes)
    # df_indic = build_indicator_SMA(quotes)
    # df_indic = build_indicator_MACD(quotes)
    # df_indic = build_indicator_DMI(quotes)
    # df_indic = build_indicator_PBX(quotes)
    # df_indic = build_indicator_DMA(quotes)
    # df_indic = build_indicator_MTM(quotes)
    # df_indic = build_indicator_CHO(quotes) # 累积量
    # df_indic = build_indicator_KDJ(quotes)
    # df_indic = build_indicator_BIAS(quotes)
    # df_indic = build_indicator_ROC(quotes)
    # df_indic = build_indicator_CCI(quotes)
    # df_indic = build_indicator_WR(quotes)
    # df_indic = build_indicator_OSC(quotes)
    # df_indic = build_indicator_RSI(quotes)
    # df_indic = build_indicator_ADTM(quotes)
    # df_indic_ = build_indicator_VR(quotes)
    # df_indic = build_indicator_VRSI(quotes)
    # df_indic = build_indicator_ARBR(quotes)
    df_indic = build_indicator_CR(quotes)
    # df_indic = build_indicator_VSTD(quotes)
    # df_indic = build_indicator_ASI(quotes)
    # df_indic = build_indicator_PVT(quotes) # 累积量
    # df_indic = build_indicator_OBV(quotes) # 累积量
    # df_indic = build_indicator_VPT(quotes) # 累积量
    # df_indic = build_indicator_BOLL(quotes)
    # df_indic = build_indicator_MIKE(quotes)
    # df_indic = build_indicator_BBI(quotes)
    # df_indic = build_indicator_MFI(quotes)
    # df_indic = build_indicator_ATR(quotes)
    # df_indic = build_indicator_SKDJ(quotes)
    # df_indic = build_indicator_DDI(quotes)
    # df_indic = build_indicator_shadow(quotes)
    print(df_indic)


stock_indic_flow()