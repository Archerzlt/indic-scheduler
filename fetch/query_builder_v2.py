import pandas as pd
from fetch.sdk import *
import concurrent.futures
import numpy as np

def process_function(field_name, func_name, kwargs):
    func = globals().get(func_name)  # 获取函数对象
    if func is not None and callable(func):
        result = func(**kwargs)  # 调用函数
        result = dict(result)
        result['field_name'] = field_name
        return result
    else:
        print(f"Function '{func_name}' not found or is not callable.")
        return None

def get_query_builder(field,functions_and_args):
    
    functions_and_args = [(field_name, func_name, kwargs) for field_name, func_name, kwargs in functions_and_args if field_name in field]
    
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_function, field_name, func_name, kwargs) for field_name, func_name, kwargs in functions_and_args]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
                
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
        merged_temporal_data = pd.merge(merged_temporal_data,merged_crosssection_data,on=['symbol'], how='outer')
        return merged_temporal_data.fillna(np.nan)

if __name__ == '__main__':
    # Access key and secret key
    # 指数成分股
    index_component = get_index_indexcomponent(symbols=["000001","399107","000688"]) # symbols 可以为数组
    symbols = index_component['symbol']

    start_date ='20240301'
    # end_date = '20240331'
    end_date = '20240305'
    if start_date != end_date:
        trade_date = get_s_tdays(start_date=start_date,end_date=end_date)
        report_date = get_s_tdays(start_date=start_date, end_date=end_date, period='q')
        
        trade_date = trade_date['date']
        report_date = report_date['date']
    else:
        trade_date = end_date
        report_date = end_date
    print("dates:",trade_date,report_date)
    field = ["pchg","amount","close","open","high","low"]
    # field = ["pchg"] 
    functions_and_args = [
                    ("pchg","get_s_price_pchg_d", {"code": symbols, "trade_date": trade_date, "count_type": '1'}),
                    ("amount","get_s_price_amount_d", {"code": symbols, "trade_date": trade_date, "count_type": '1'}),
                    ("close","get_s_price_close_d", {"code": symbols, "trade_date": trade_date, "count_type": '1', "adjust_type": 'B'}),
                    ("open", "get_s_price_open_d",{"code": symbols, "trade_date": trade_date, "count_type": '1', "adjust_type": 'B'}),
                    ("high", "get_s_price_high_d",{"code": symbols, "trade_date": trade_date, "count_type": '1', "adjust_type": 'B'}),
                    ("low", "get_s_price_low_d",{"code": symbols, "trade_date": trade_date, "count_type": '1', "adjust_type": 'B'}),
                        ]
    import time
    start_time = time.time()
    result = get_query_builder(field,functions_and_args)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"ALLQUERY执行时间：{execution_time} 秒")
    # print(result.loc[:10,:])
    result = result.astype(float)
    result.to_feather("quota.feather")
    # print(len(result['date'].unique()))
    print(result)
