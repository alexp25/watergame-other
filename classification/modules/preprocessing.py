
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import math


class Preprocessing:
    def __init__(self):
        self.encoder = None

    # adapt input, from list of list of binary values to list of string encoded binary values
    def adapt_input(self, data):
        data_str = []
        for row in data:
            row_str = self.adapt_input_core(row)
            data_str.append([row_str])
        # print(data_str)
        return data_str

    # adapt input from list of binary values to a string encoded binary value
    def adapt_input_core(self, data):
        return "".join([str(int(e)) for e in data])

    # create one hot encoder
    def create_encoder(self, data):
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        # enc = OrdinalEncoder()
        self.encoder.fit(data)

        # print(self.encoder.categories_)
        # td = enc.transform(data).toarray()
        td = self.encoder.transform(data)
        # print("encoded:")
        # print(td)
        return td

    # encode via one hot encoder
    def encode(self, data):
        encoded = self.encoder.transform(data)
        # print(encoded)
        s = np.shape(data)
        rows = s[0]
        cols = s[1]
        print("encode > encoded shape: ", rows, cols)
        print(encoded[0:10])
        return encoded

    # decode via one hot encoder
    # adapt unknown encodings
    def decode(self, data):
        s = np.shape(data)
        rows = s[0]
        cols = s[1]
        cols_data = cols
        cols_decoded = int(math.log(cols_data, 2))
        # print("decode > encoded shape: ", rows, cols)
        decoded = self.encoder.inverse_transform(data)
        s = np.shape(decoded)
        rows = s[0]
        cols = s[1]
        # print("decode > decoded shape: ", rows, cols)
        i = 0
        for r in range(rows):
            if decoded[r][0] is None:
                decoded[r][0] = "".join([str(e) for e in [0]*cols_decoded])
                # print(decoded[r])
            else:
                if i == 0:
                    # print(decoded[r])
                    i += 1
                # np.zeros((cols,), dtype=int)
                # print("decoded unknown: ", decoded[r])
        # print(decoded)
        return decoded

    # decode from string encoded binary value into an int
    def binary_to_int(self, val_s):
        int_b = 0
        val_s = val_s[::-1]
        p = 0
        for c in val_s:
            int_b1 = int(c)
            if int_b1 == 1:
                int_b += pow(2, p)
            p += 1
        return int_b

    # decode from list of lists of binary values into a list of ints
    def decode_int(self, data):
        ints = []
        for b in data:
            b = b[0]
            # print(b)
            int_b = self.binary_to_int(b)
            ints.append(int_b)
        return ints

    # decode from list of lists of one hot encoded binary values into a list of ints
    def decode_int_onehot(self, data):
        ints = []
        print("decoding: ", data)
        s = np.shape(data)
        print("shape: ", s[0], s[1])
        for b in data:
            # self.adapt_input_core(b)
            # b = b[0]
            # print("data: ", b)
            s = len(b)
            # print("shape: ", s)
            decoded_binary = self.decode([b])[0]
            # print("decoded binary: ", decoded_binary)
            int_b = self.binary_to_int(decoded_binary[0])
            # print("decoded int: ", int_b)
            ints.append(int_b)
        return ints

    def str_to_list(self, data):
        res = []
        invalid_count = 0
        for d in data:
            # print(d)
            if d[0] is not None:
                res.append([int(e) for e in d[0]])
            else:
                res.append(d)
                invalid_count += 1
        print("invalid count: " + str(invalid_count))
        return res


def get_samples_nbins(a, bins):
    skip = int(len(a) / bins)
    b = get_samples_skip(a, skip)
    return b[:bins]

def get_samples_skip(a, skip):
    b = []
    for i, e in enumerate(a):
        if i % skip == 0:
            b.append(e)
    return b

