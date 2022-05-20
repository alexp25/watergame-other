import pandas as pd
import numpy as np
import csv


filename = "./data/Shower water/amphiro_trialA.csv"
output_folder = "./data/Shower water/"

df = pd.read_csv(filename)
userdict = {}

all_data = True

spec = {
    "uid": "device.key",
    "time": "Alicante.local.date.time",
    "data": "flow",  
    "temp": "temperature",  
    "volume": "volume",
    "energy": "energy"
}

for index, row in df.iterrows():
    if index < 1000 or all_data:
        if row[spec["uid"]] in userdict:
            for speckey in spec:
                userdict[row[spec["uid"]]][speckey].append(row[spec[speckey]])
        else:
            userdict[row[spec["uid"]]] = {}
            for speckey in spec:
                userdict[row[spec["uid"]]][speckey] = []
    else:
        break

print(df)
print(len(list(userdict)))
print("processing")
head = ["id"] + list(spec.keys())

for key in userdict:
    datalen = len(userdict[key]["data"])
    npdata = []
    for i in range(datalen):
        datarow = [i]
        for speckey in spec:
            try:
                dataspec = userdict[key][speckey][i]
            except:
                dataspec = userdict[key][speckey]
            datarow.append(dataspec)
        npdata.append(datarow)

    # opening the csv file in 'w+' mode
    file = open(output_folder + '/user_' + key + ".csv", 'w', newline ='')
    
    # writing the data into the file
    with file:    
        file.write(",".join(head) + "\n")
        write = csv.writer(file)
        write.writerows(npdata)

print("done")