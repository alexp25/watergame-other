import pandas as pd
import numpy as np
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import cross_validate, train_test_split
import matplotlib.pyplot as plt
import matplotlib
import json

# Load the data into a pandas dataframe


# df = pd.read_csv('./data/test_data.csv')

# df = pd.read_csv('./data/decomp_table_format.csv')
# sources_filter = ['chiuveta_rece', 'chiuveta_calda', 'toaleta']
# df = df[df['source'].isin(sources_filter)]

key_target = "value"
key_target = "duration"

label_y = "Consumption (L)"
label_y = "Duration (min)"


df = pd.read_csv('./data/res_evt_vol.csv')
# df = pd.read_csv('./data/res_evt.csv')
sources_filter = ['sink_cold', 'sink_hot', 'toilet']
df = df[df['source'].isin(sources_filter)]

type = "volume"
type = "duration"


# Pivot the data to create the user-item matrix
pivot_table = pd.pivot_table(df, values=key_target, index='user', columns='source', aggfunc="mean")

# print(pivot_table)
# quit()

# Define the rating scale
reader = Reader(rating_scale=(0, 100))

# Load the data into a Surprise dataset
data = Dataset.load_from_df(df[['user', 'source', key_target]], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, shuffle=False)

# Create the collaborative filtering model
model = SVD()

# Train the model using cross-validation
# cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train the model using the full training set
model.fit(trainset)

users = np.unique(df['user'])
sources = np.unique(df['source'])


# Generate recommendations for a user
for user_id in users:
    print('Recommended consumption for user', user_id, ':')
    for source in sources:
        est = model.predict(user_id, source).est
        print(source, est)


def get_rule_based_recommendations(df, user_id):
    # Filter the consumption data for the given user
    user_consumption = df[df['user'] == user_id]
    sources = np.unique(user_consumption['source'])
    filtered_recommendations = []

    for source in sources:
        est = model.predict(user_id, source).est       
        actual = user_consumption[user_consumption['source'] == source][key_target].mean()
        print(source, actual, est)
        # if actual > est:
        filtered_recommendations.append([source, actual, est, actual > est])

    # Filter the recommendations to exclude outlets that the user has already exceeded the threshold for
    # filtered_recommendations = [r for r in cf_recommendations if r not in list(exceeds_threshold['source'])]

    # Return the filtered recommendations
    return filtered_recommendations

# Generate recommendations for each user in the dataset
all_recommendations = {}
all_recommendations_vect = []
for user_id in df['user'].unique():
    recommendations = get_rule_based_recommendations(df, user_id)
    all_recommendations[int(user_id)] = recommendations
    all_recommendations_vect.append(recommendations)

# Print the recommendations for each user
for user_id, recommendations in all_recommendations.items():
    print(f"User {user_id}: {recommendations}")


with open("output_collab.json", "w") as f:
    s = json.dumps(all_recommendations, indent=2, default=str)
    f.write(s)

with open("output_collab.csv", "w") as f:
    s = 'user_id,outlet,actual_value,recommended_value,exceeds\n'
    for user_id in all_recommendations:
        for outlet in all_recommendations[user_id]:
            rec = outlet
            s += str(user_id) + ',' + str(rec[0]) + ',' + str(rec[1]) + ',' + str(rec[2]) + ',' + str(rec[3]) + '\n'
    f.write(s)

# print(all_recommendations_vect)
# quit()

# create the bar chart
fig, ax = plt.subplots()
bar_width = 0.4
opacity = 0.8
opacity_rec = 0.5
offset = bar_width / 2
width = bar_width / 8

FSIZE_TITLE = 16
FSIZE_LABEL = 14
FSIZE_LABEL_S = 14
FSIZE_LABEL_XS = 12
OPACITY = 0.9
OPACITY = 1

colors = ['b','g','r','c'] * 3

j_offset  = int(len(sources) / 2)
show_once = True
show_actual = True

ax.grid(zorder=0)

exceeds_threshold = {}

for i, user_id in enumerate(users):
    for j, source in enumerate(sources):
        est = model.predict(user_id, source).est
        actual = np.mean((df[df['source'] == source][df['user'] == user_id][key_target].to_numpy()))     
        ax.bar(i + offset * (j-j_offset) - width/2,  actual, width=width, alpha=opacity, color=colors[j], label='actual (' + source + ')' if show_actual else '_nolegend_', zorder=3)
        # ax.bar(i + offset * (j-j_offset) + width - width/2, est, width=width, alpha=opacity_rec, color=colors[j])
        ax.bar(i + offset * (j-j_offset) + width - width/2, est, width=width, alpha=opacity, color='orange', label='recommended' if show_once and not show_actual else '_nolegend_', zorder=3)
        if not show_actual:
            show_once = False
    show_actual = False


# set the axis labels and title
ax.set_xlabel('User (outlet id)', fontsize=FSIZE_LABEL)
ax.set_ylabel(label_y, fontsize=FSIZE_LABEL)
ax.set_title('Collaborative filtering results', fontsize=FSIZE_TITLE, pad=10)
ax.set_xticks(np.arange(len(users)))
ax.set_xticklabels(users)

# We change the fontsize of minor ticks label 
ax.tick_params(axis='y', which='major', labelsize=FSIZE_LABEL_S)
ax.tick_params(axis='y', which='minor', labelsize=FSIZE_LABEL_S)

# add legend
# ax.legend(['actual', 'recommended'])
ax.legend(loc=2, prop={'size': FSIZE_LABEL_S})
fig.set_size_inches(10.,8.)


plt.savefig('sample_collab_results_' + type + '.png', dpi=300)
# show the plot
plt.show()
