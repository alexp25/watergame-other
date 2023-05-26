import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./data/res_evt_vol.csv')
sources_filter = ['sink_cold', 'sink_hot', 'toilet']
df = df[df['source'].isin(sources_filter)]

volumes = df["value"]
durations = df["duration"]


# Create a pivot table
pivot_table = pd.pivot_table(df, values='value', index='user', columns='source', aggfunc='mean')

# Display the pivot table
print(pivot_table)


# create the bar chart
fig, ax = plt.subplots()

FSIZE_TITLE = 16
FSIZE_LABEL = 14
FSIZE_LABEL_S = 14
FSIZE_LABEL_XS = 12
OPACITY = 0.9
OPACITY = 1

ax.grid(zorder=0)

ax.set_xlabel('Duration', fontsize=FSIZE_LABEL)
ax.set_ylabel('Volume', fontsize=FSIZE_LABEL)


# We change the fontsize of minor ticks label 
ax.tick_params(axis='y', which='major', labelsize=FSIZE_LABEL_S)
ax.tick_params(axis='y', which='minor', labelsize=FSIZE_LABEL_S)

ax.tick_params(axis='x', which='major', labelsize=FSIZE_LABEL_S)
ax.tick_params(axis='x', which='minor', labelsize=FSIZE_LABEL_S)

# ax.set_title('Collaborative filtering results', fontsize=FSIZE_TITLE, pad=10)

# add legend
# ax.legend(['actual', 'recommended'])
# ax.legend(loc=2, prop={'size': FSIZE_LABEL_S})
fig.set_size_inches(10.,8.)

# Create the plot
plt.scatter(durations, volumes, zorder=3)
plt.xlabel('Duration (min)')
plt.ylabel('Volume (L)')
plt.title('Water Consumption Events', fontsize=FSIZE_TITLE, pad=10)

plt.savefig('consumption_events.png', dpi=300)
# show the plot

plt.show()




