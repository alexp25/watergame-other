import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the data into a Pandas DataFrame
# data = pd.read_csv('./data/test_data.csv')
data = pd.read_csv('./data/decomp_table_format.csv')

# Split the data into training and testing sets
train = data.sample(frac=0.8, random_state=1)
test = data.drop(train.index)

# Calculate the mean consumption for each user and source in the training set
mean_consumption = train.groupby(['user', 'source'])['value'].mean().reset_index()

print(mean_consumption)

# Calculate the deviation of each consumption value from its respective mean in the training set
deviation = train.copy()
deviation['mean'] = deviation.groupby(['user', 'source'])['value'].transform('mean')
deviation['deviation'] = deviation['value'] - deviation['mean']

print(deviation)

# Reshape the deviation DataFrame to create a matrix where each row represents a user, each column represents a source, and the cells contain the deviation of the corresponding consumption value from the user and source mean
deviation_matrix = deviation.pivot_table(index='user', columns='source', values='deviation').fillna(0)

# Calculate the similarity between users based on their consumption patterns
similarity = pd.DataFrame(cosine_similarity(deviation_matrix))

# For each user and source in the testing set, find the K most similar users and calculate the weighted average of their deviations
K = 3
rec = {}
for _, row in test.iterrows():
    user = row['user']
    source = row['source']
    key = str(user) + source
    if key in rec:
        continue    
    mean = mean_consumption[(mean_consumption['user'] == user) & (mean_consumption['source'] == source)]['value'].iloc[0]
    dev = deviation[(deviation['user'] != user) & (deviation['source'] == source)]
    dev = pd.merge(dev, similarity[user-1].reset_index(), left_on='user', right_on='index')
    dev['weighted_deviation'] = dev['deviation'] * dev[user-1]
    recommended_value = mean + (dev['weighted_deviation'].sum() / dev[user-1].sum())
    # print(f"Recommended consumption for user {user} and source {source}: {recommended_value}")
    rec[key] = {
        'user': user,
        'source': source,
        'rec': recommended_value
    }

for r in rec:
    print(rec[r])