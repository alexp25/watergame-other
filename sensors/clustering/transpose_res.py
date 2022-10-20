

# import our modules
from modules import loader
import yaml
config = yaml.safe_load(open("config.yml"))


reverse = False

root_data_folder = "./data"
# read the data from the csv file

rolling_filter = False
# rolling_filter = True

result_name = root_data_folder + "/res"
if rolling_filter:
    result_name += "_rf"

if reverse:
    result_name_file = result_name + "_t.csv"
else:
    result_name_file = result_name + ".csv"

df = loader.load_dataset_pd(result_name_file)
df = df.transpose()

if reverse:
    result_name_file = result_name + ".csv"
else:
    result_name_file = result_name + "_t.csv"
loader.write_dataset_pd(df, result_name_file)
print(df)