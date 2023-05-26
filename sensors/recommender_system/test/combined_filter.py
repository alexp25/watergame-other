
import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split

import matplotlib.pyplot as plt


# Load consumption data
df = pd.read_csv('./data/decomp_table_format.csv')
# df = pd.read_csv('./data/test_data.csv')

# Define the Surprise Reader object
reader = Reader(rating_scale=(0, 10))

# Load the consumption data into a Surprise Dataset object
# data = Dataset.load_from_df(df[['user_id', 'outlet_id', 'consumption']], reader)
data = Dataset.load_from_df(df[['user', 'source', 'value']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, shuffle=False)

# Train a Singular Value Decomposition (SVD) algorithm on the training data
model = SVD()
model.fit(trainset)

# Define a function to generate recommendations for a given user
def get_collaborative_filtering_recommendations(user_id, num_items=5):
    # Predict the ratings for all items for the given user

    print("unique sources: ", df['source'].unique())

    all_predictions = []
    for outlet_id in df['source'].unique():
        all_predictions.append(model.predict(user_id, outlet_id))

    # Convert the predictions to a dataframe and sort them by rating
    # print("predictions: ", [p.est for p in all_predictions])

    print("predictions: ", len(all_predictions))
    

    top_predictions = [p.est for p in all_predictions]
    predictions_iids = [p.iid for p in all_predictions]

    print("predictions est: ", top_predictions)


    # https://realpython.com/build-recommendation-engine-collaborative-filtering/


    # predictions_df = 
    # predictions_df = pd.DataFrame.from_records([p.est for p in all_predictions], columns=['rating'])
    # # predictions_df = pd.DataFrame.from_records([p.est for p in all_predictions])

    # predictions_df['source'] = df['source'].unique()
    # predictions_df = predictions_df.sort_values('rating', ascending=False).reset_index(drop=True)

    # # Get the top n highest rated items
    # top_n = predictions_df.head(num_items)

    # Return the outlet ids of the recommended items
    # return list(top_n['source'])

    return top_predictions, predictions_iids


def sorted_enumerate_key(seq):
    return [x for x, y in sorted(enumerate(seq), key=lambda x: x[1])]


# Define the rules-based recommender function
def get_rule_based_recommendations(df, user_id):
    # Calculate the mean consumption for each outlet
    gb = df.groupby('source')
    gb_sources = [key[0] for key in gb['source']]

    mean_consumption = gb['value'].mean()
    # print("mean consumption by outlet: ", mean_consumption)

    # Calculate the standard deviation of consumption for each outlet
    std_consumption = gb['value'].std()
    # print("stdev consumption by outlet: ", std_consumption)

    # ------------- Calculate the dynamic thresholds for each outlet
    thresholds = mean_consumption + 0.5 * std_consumption
    print("thresholds by outlet: ", thresholds)

    # print("thresholds loc: ", thresholds.loc['chiuveta'])

    # Filter the consumption data for the given user
    user_consumption = df[df['user'] == user_id]

    gb = user_consumption.groupby('source')
    gb_sources = [key[0] for key in gb['source']]   
    user_mean_consumption = gb['value'].mean()
    print("user mean consumption: ", user_mean_consumption)

    # print("thresholds locs: ", thresholds.loc[user_consumption['source']])

    # Get the list of recommended outlets based on the collaborative filtering algorithm
    cf_recommendations, cf_iids = get_collaborative_filtering_recommendations(user_id)

    print("collaborative filter recommendations: ", cf_recommendations)

    sorted_idx_orig = sorted_enumerate_key(gb_sources)
    # print(sorted_idx_orig)

    sorted_idx_cf = sorted_enumerate_key(cf_iids)
    # print(sorted_idx_cf)

    gb_sources = [gb_sources[i] for i in sorted_idx_orig] 
    cf_iids = [cf_iids[i] for i in sorted_idx_cf]

    cf_recommendations = [cf_recommendations[i] for i in sorted_idx_cf]
    user_mean_consumption = user_mean_consumption.to_list()
    user_mean_consumption = [user_mean_consumption[i] for i in sorted_idx_orig]

    cf_recommendations = cf_recommendations[:len(gb_sources)]

    rec = {
        "collab": cf_recommendations,
        "current": user_mean_consumption,
        "sources": gb_sources,
        # "sources_collab": cf_iids
    }

     # Identify the outlets where the user's consumption exceeds the dynamic threshold
    # exceeds_threshold = user_consumption.to_numpy()[user_consumption['value'].to_numpy() > thresholds.loc[user_consumption['source'].to_numpy()]]

    # print(exceeds_threshold)

    # Filter the recommendations to exclude outlets that the user has already exceeded the threshold for
    # filtered_recommendations = [r for r in cf_recommendations if r not in list(exceeds_threshold['source'])]

    # Return the filtered recommendations
    # return filtered_recommendations

    return rec

# Generate recommendations for each user in the dataset
all_recommendations = {}
users = []
for user_id in df['user'].unique():
    recommendations = get_rule_based_recommendations(df, user_id)
    all_recommendations[user_id] = recommendations
    users.append(recommendations)

# Print the recommendations for each user
for user_id, recommendations in all_recommendations.items():
    print(f"User {user_id}: {recommendations}")


# plot comparative recommendations
collab_values = []
current_values = []

legend = ['collab', 'current']

for user in users:
    # collab_values.append(sum(user['collab']))
    # current_values.append(sum(user['current']))
    collab_values.append(user['collab'][0])
    current_values.append(user['current'][0])
    legend = ['collab ' + user['sources'][0], 'current ' + user['sources'][0]]

x = range(len(users))

plt.bar(x, collab_values, color='b', width=0.3)
plt.bar([i+0.3 for i in x], current_values, color='g', width=0.3)

plt.legend(legend)
plt.xticks([i+0.15 for i in x], ['User {}'.format(i+1) for i in x])

plt.show()
