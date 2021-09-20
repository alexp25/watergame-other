import pandas as pd
import numpy as np
import csv


filename = "./data/Smart Water Meter/smart__water_meter_trialA.csv"

df = pd.read_csv(filename)
userdict = {}

all_data = True

for index, row in df.iterrows():
    if index < 1000 or all_data:
        if row["user key"] in userdict:
            userdict[row["user key"]]["meter"].append(row["meter reading"])
            userdict[row["user key"]]["data"].append(row["diff"])
            userdict[row["user key"]]["time"].append(row["datetime"])
        else:
            userdict[row["user key"]] = {
                "meter": [],
                "data": [],
                "time": []
            }
    else:
        break

print(df)
print(len(list(userdict)))

print("processing")

head = ["id", "uid", "time", "data", "meter"]

for key in userdict:
    datalen = len(userdict[key]["data"])
    # npdata = np.zeros((datalen, 4), dtype="S")

    # npdata[:, 0] = np.array([0] * datalen)
    # npdata[:, 1] = np.array(userdict[key]["time"])
    # npdata[:, 2] = np.array(userdict[key]["data"])
    # npdata[:, 3] = np.array(userdict[key]["meter"])

    npdata = []

    for i in range(datalen):
        npdata.append([i, key, userdict[key]["time"][i], userdict[key]["data"][i], userdict[key]["meter"][i]])

    # opening the csv file in 'w+' mode
    file = open('./data/Smart Water Meter/user_' + key + ".csv", 'w', newline ='')
    
    # writing the data into the file
    with file:    
        file.write(",".join(head) + "\n")
        write = csv.writer(file)
        write.writerows(npdata)

print("done")