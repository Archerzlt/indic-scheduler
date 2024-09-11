from fetch.sdk import *


if __name__ == '__main__':
    # demo3
    index_component = get_index_indexcomponent(symbols=["000002","399107"])
    # index_component = get_index_indexcomponent(symbols=["000688"])
    print("index_component_length:",len(index_component['symbol']))
    # symbols = ["600745","603986"]
    symbols = index_component['symbol']
    report_date1 =['20231231','20221231','20211231','20201231','20191231']
    # end_date = '20240520'
    # if start_date != end_date:
    #     trade_date = get_s_tdays(start_date=start_date,end_date=end_date)
    #     report_date = get_s_tdays(start_date=start_date, end_date=end_date, period='q')
    # else:
    #     trade_date = end_date
    #     report_date = end_date
    field = ["bizinco","bizcost","netprofit","totasset","declaredate"]
    functions_and_args = [   ("bizinco", "get_s_bizinco",
                              {"code": symbols, "report_date": report_date1, "report_type": '1'}),
                             ("bizcost", "get_s_bizcost",
                              {"code": symbols, "report_date": report_date1, "report_type": '1'}),
                             ("netprofit", "get_s_netprofit",
                              {"code": symbols, "report_date": report_date1, "report_type": '1'}),
                             ("totasset", "get_s_totasset",
                              {"code": symbols, "report_date": report_date1, "report_type": '1'}),
                              ("declaredate", "get_s_declaredate",
                            {"code": symbols, "report_date": report_date2, "report_type": '1'})
                             ]
    import time
    start_time = time.time()
    result = get_query_builder(field,functions_and_args)
    result['symbol']=result.apply(lambda x:str(x['symbol']),axis=1)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"ALLQUERY执行时间：{execution_time} 秒")
    result  = result.dropna()
    report_date2 = ['20231231','20221231','20211231','20201231','20191231']
    field2 = ["declaredate"]
    functions_and_args2 = [("declaredate", "get_s_declaredate",
                            {"code": symbols, "report_date": report_date2, "report_type": '1'})
                           ]
    result['ATO'] = result['bizinco']/result['totasset']
    result['GP'] = (result['bizinco'] - result['bizcost'])/result['totasset']
    result['GPM'] = (result['bizinco'] - result['bizcost'])/result['bizinco']
    result['ROA'] = result['netprofit'] / result['totasset']
    factor = result.loc[:, ['symbol','date','ATO','GP','GPM','ROA',"declaredate"]]
    # 合并发布日期
    #declaredate_result = get_query_builder(field2, functions_and_args2)
    #declaredate_result['symbol'] = declaredate_result .apply(lambda x: str(x['symbol']), axis=1)
    #factor = pd.merge(factor,declaredate_result,on = ['date','symbol'])
    print(factor)

