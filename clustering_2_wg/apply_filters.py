
import pandas as pd


def apply_filter_labels(df, filter_labels):
    if len(filter_labels) > 0:
        boolean_series = df['label'].isin(filter_labels)
        df = df[boolean_series]
    print(len(df))
    return df


def apply_balancing(df, filter_labels):
    dff = pd.DataFrame()
    min_count = None
    print("apply balancing")
    for label in filter_labels:
        try:
            is_label = df['label'] == label
            # print(list(is_label))
            count_label = sum(
                [1 if el == True else 0 for el in list(is_label)])
            print(count_label)
            if min_count is None or count_label < min_count:
                min_count = count_label
        except:
            print("label not found: ", label)
    for label in filter_labels:
        try:
            boolean_series = df['label'] == label
            print("fs: ", len(df[boolean_series]))
            df_sample = df[boolean_series].sample(n=min_count)
            print("dfs: ", len(df_sample))
            dff = dff.append(df_sample)
        except:
            print("label not found 2: ", label)
    print(min_count)
    print(len(dff))
    return dff
