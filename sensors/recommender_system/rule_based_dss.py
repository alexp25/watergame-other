import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

with open("output_collab.json", "r") as f:
    all_recommendations = json.loads(f.read())
    print(all_recommendations)


df = pd.read_csv('./output_collab.csv')
print(df)


with open("ruleset.json", "r") as f:
    ruleset = json.loads(f.read())

# print(df.to_numpy().tolist())
# quit()
recommendations = []

# loop over each user's consumption data
for index, row in df.iterrows():
    user_id = row['user_id']
    outlet = row['outlet']
    actual_value = row['actual_value']
    recommended_value = row['recommended_value']   
   
    for rule in ruleset['water_saving_actions']:
        if rule['location'] == outlet:
            if actual_value - recommended_value > rule['threshold']:
                 recommendations.append({'user_id': user_id,
                                'action': rule['description'],
                                'actual_value': actual_value,
                                'recommended_value': recommended_value,
                                'threshold': rule['threshold'],
                                'unit': rule['unit']})
   
print(recommendations)