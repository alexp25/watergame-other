
import pandas as pd


input_file = './data/res_vol.csv'
output_file = './data/res_vol_format.csv'

# read the csv file
df = pd.read_csv(input_file)

# transpose the dataframe
df_transposed = df.transpose()

# save the transposed dataframe to a new csv file
df_transposed.to_csv(output_file)
