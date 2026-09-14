import pandas as pd

# 1. 读取数据
file_path = "dataset/traffic_predict/50场景业务数据-5.xlsx"
df = pd.read_excel(file_path)

# 2. 提取并设置正确的列名
# 第一行(iloc[0])包含了真实的链路名，我们取第1列之后的所有内容
real_names = df.iloc[0, 1:].values
# 截取数据部分：去掉第一行(名称行)和第一列(网口名称)
df_clean = df.iloc[1:, 1:].copy()
# 赋值新列名
df_clean.columns = real_names

# 3. 强制转换为数值类型 (处理潜在的非数字字符)
df_clean = df_clean.apply(pd.to_numeric, errors='coerce')

# 4. 缺失值处理 (使用前向填充，假设流量是连续的)
df_clean = df_clean.fillna(method='ffill').fillna(0)

# 5. 【关键】生成时间索引
# 请根据实际情况修改 start (开始时间) 和 freq (采集频率, 如 '5T' 代表5分钟)
df_clean.index = pd.date_range(start='2024-01-01 00:00', periods=len(df_clean), freq='5T')

# 6. 查看结果
print(df_clean.head())

# 7. 保存为新的 CSV 文件
df_clean.to_csv("dataset/traffic_predict/cleaned_traffic_data.csv")