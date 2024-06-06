import pandas as pd

# 加载航空数据
chengdu_flights = pd.read_csv('Chengdu_flights_from_hefei.csv')
chongqing_flights = pd.read_csv('Chongqing_flights_from_hefei.csv')
shanghai_flights = pd.read_csv('Shanghai_flights_from_hefei.csv')

# 合并所有航空数据到一个DataFrame
all_flights = pd.concat([chengdu_flights, chongqing_flights, shanghai_flights], ignore_index=True)

# 保存合并后的数据到CSV文件
output_flights_path = 'all_flights.csv'
all_flights.to_csv(output_flights_path, index=False)

print(f"Flight data saved to {output_flights_path}")
