import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# Load data
# data = pd.read_csv('./data/test_data.csv')
data = pd.read_csv('./data/decomp_table_format.csv')

# sources_filter = ['chiuveta', 'toaleta', 'dus']
# data = data[data['source'].isin(sources_filter)]

# Transform data into a matrix
matrix = pd.pivot_table(data, values='value', index=['user'], columns=['source'], fill_value=0)

sources = list(matrix.columns)

print(matrix)
# quit()

# Calculate similarity between users using cosine similarity
user_similarity = cosine_similarity(matrix)

# Calculate similarity between sources using cosine similarity
source_similarity = cosine_similarity(matrix.T)

print("user similarity: ", user_similarity)
print("source similarity: ", source_similarity)

# Define a function to make recommendations
def make_recommendation(user_id, source):
    # Find similar users
    similar_users = np.flip(np.argsort(user_similarity[user_id-1]))[1:]
    # Find similar sources
    similar_sources = np.flip(np.argsort(source_similarity[source]))[1:]
    # Calculate weighted average of consumption for similar users and sources
    weighted_sum = 0
    similarity_sum = 0
    for user in similar_users:
        for s in similar_sources:
            if not pd.isnull(matrix.iloc[user][s]):
                weighted_sum += user_similarity[user][user_id-1] * source_similarity[source][s] * matrix.iloc[user][s]
                similarity_sum += user_similarity[user][user_id-1] * source_similarity[source][s]
    if similarity_sum == 0:
        return np.nan
    else:
        return weighted_sum / similarity_sum

# Make recommendations for all user-source combinations
recommendations = pd.DataFrame(index=matrix.index, columns=matrix.columns)
for user in matrix.index:
    for source in matrix.columns:
        source_index = sources.index(source)
        recommendations.loc[user, source] = make_recommendation(user, source_index)


print(matrix)
# Print recommendations
print(recommendations)


quit()
recommendations_matrix = recommendations.to_numpy()



# create arrays for user, source, current consumption, and recommended consumption
users = np.unique(data['user'])
sources = np.unique(data['source'])
current_consumptions = np.zeros((len(users), len(sources)))
recommended_consumptions = np.zeros((len(users), len(sources)))

for i, user in enumerate(users):
    for j, source in enumerate(sources):
        mask = (data['user'] == user) & (data['source'] == source) 
        current_consumptions[i, j] = np.sum(data[mask]['value']) 
        recommended_consumptions[i, j] = recommendations_matrix[user-1, j]


print(current_consumptions)
print(recommended_consumptions)


current_consumptions = current_consumptions.reshape(-1)
recommended_consumptions = recommended_consumptions.reshape(-1)



# create a bar chart comparing the current and recommended consumption for each user and source
x = np.arange(len(users) * len(sources))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, current_consumptions, width, label='Current Consumption')
rects2 = ax.bar(x + width/2, recommended_consumptions, width, label='Recommended Consumption')

# add labels and titles
ax.set_ylabel('Water Consumption')
ax.set_xlabel('User and Source')
ax.set_xticks(x)
ax.set_xticklabels([f'User {user}, {source}' for user in users for source in sources])
ax.legend()
plt.title('Comparison of Current and Recommended Consumption by User and Source')

# display the chart
plt.show()