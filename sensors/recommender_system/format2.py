
import pandas as pd


input_file = './data/decomp_table.csv'
output_file = './data/decomp_table_format.csv'

df = pd.read_csv(input_file)

# create a new column called 'user' with a unique number for each user
df['user'] = pd.factorize(df['user'])[0] + 1

# format consumption data from L/h to L, considering 1 min sampling time
df['value'] = round(df['value'] / 60, 2)

df.to_csv(output_file)
