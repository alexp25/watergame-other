import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# Load the data into a pandas dataframe
data = pd.read_csv('./data/test_data.csv')

# data = pd.read_csv('./data/decomp_table_format.csv')
# sources_filter = ['chiuveta', 'toaleta', 'dus']
# data = data[data['source'].isin(sources_filter)]

# Compute the consumption matrix
consumption_matrix = pd.pivot_table(data, values='value', index='user', columns='source', fill_value=0)

print(consumption_matrix)

# Compute the user similarity matrix using cosine similarity
user_similarity = pd.DataFrame(1 - pairwise_distances(consumption_matrix, metric='cosine'), index=consumption_matrix.index, columns=consumption_matrix.index)
# user_similarity = pd.DataFrame(cosine_similarity(consumption_matrix))

# Compute the predicted consumption values for each user and source
predicted_consumption = pd.DataFrame(index=data.index, columns=['user', 'source', 'actual_consumption', 'predicted_consumption'])
for i, row in data.iterrows():
    user = row['user']
    source = row['source']
    consumption = row['value']
    average_consumption = consumption_matrix[source].mean()
    if not pd.isna(user_similarity[user]).any():
        # Compute the weighted average of the consumption values for similar users
        similar_users = user_similarity[user].drop(user).sort_values(ascending=False)[:5]
        similar_consumption = consumption_matrix.loc[similar_users.index, source]
        weights = similar_users.values.reshape(-1, 1)
        weighted_consumption = (similar_consumption.values * weights).sum(axis=0) / weights.sum()
        # Compute the predicted consumption value for the current user and source
        predicted_consumption.loc[i] = [user, source, average_consumption, weighted_consumption[0]]
    else:
        # If there are no similar users, use the average consumption value for the current source        
        predicted_consumption.loc[i] = [user, source, average_consumption, average_consumption]


res = predicted_consumption.groupby(['user', 'source'])

mat = []
for key, value in res:
    print(value.to_numpy()[0])
    mat.append(value.to_numpy()[0].tolist())

# print(mat)
data = np.array(mat)

print(data)

# extract the unique user and source values
users = np.unique(data[:, 0])
sources = np.unique(data[:, 1])

# create the bar chart
fig, ax = plt.subplots()
bar_width = 0.4
opacity = 1
opacity_rec = 0.5
offset = bar_width / 2
width = bar_width / 8

colors = ['b','g','r'] * 3

for i, user in enumerate(users):
    for j, source in enumerate(sources):
        vals = data[(data[:, 0] == user) & (data[:, 1] == source), 2:4][0]
        ax.bar(i + offset * j, float(vals[0]), width=width, alpha=opacity, color=colors[j], label=source)
        ax.bar(i + offset * j +width, float(vals[1]), width=width, alpha=opacity_rec, color=colors[j])


# set the axis labels and title
ax.set_xlabel('User')
ax.set_ylabel('Values')
ax.set_title('Bar Chart')
ax.set_xticks(np.arange(len(users)))
ax.set_xticklabels(users)

# add legend
# ax.legend()

# show the plot
plt.show()
