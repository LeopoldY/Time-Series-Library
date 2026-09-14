import pandas as pd
import mysql.connector
from mysql.connector import errorcode
import os

# 连接 MySQL 数据库
try:
    cnx = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="123456a?",
        database="warn2016"
    )
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
    else:
        print(err)
    exit(1)

# 从 MySQL 读取数据到 DataFrame
query = "SELECT * FROM warn_2016_filter"
df = pd.read_sql(query, cnx)
# print("Data read from MySQL:")
# print(df)

df['timestamp'] = pd.to_datetime(df['HAPPEN_TIME'], format='%m/%d/%Y %H:%M:%S')
df.sort_values(by='timestamp',inplace=True)

min_time = df['timestamp'].min()
max_time = df['timestamp'].max()



for window in [60]:
    middle_timedelta = pd.Timedelta(days=0, hours=0, minutes=window/2)
    timedelta = pd.Timedelta(days=0, hours=0, minutes=window)
    for type in df['DEVICE_TYPE'].unique():
        dir_path = "/mnt/sdc1/skx/Time-Series-Library/dataset/fault/"+type
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        
        for device in df[df['DEVICE_TYPE'] == type]['DEVICE'].unique():
            if not os.path.exists(dir_path+'/'+device):
                os.mkdir(dir_path+'/'+device)
            deviceData = df[(df['DEVICE_TYPE'] == type) & (df['DEVICE'] == device)]
                        
            print(f'{dir_path}/{device} original data shape: {deviceData.shape}')
            
            curr_timestamp = min_time
            data = []
            while curr_timestamp <= max_time:
                start_timestamp = curr_timestamp
                middle_timestamp = curr_timestamp + middle_timedelta
                end_timestamp = curr_timestamp + timedelta
                df_window = deviceData[(deviceData['timestamp']>=start_timestamp) & (deviceData['timestamp']<end_timestamp)]
                data.append([str(middle_timestamp),len(df_window[df_window['WARN_LEVEL']=='提示']),len(df_window[df_window['WARN_LEVEL']=='次要']),len(df_window[df_window['WARN_LEVEL']=='重要']),len(df_window[df_window['WARN_LEVEL']=='紧急'])])
                curr_timestamp = end_timestamp
            window_df = pd.DataFrame(data, columns=['date', 'w_level1', 'w_level2', 'w_level3', 'w_level4'])
            window_df.to_csv(dir_path+'/'+device+f'/window_{window}_data2.csv',index=False)
# 关闭数据库连接
cnx.close()