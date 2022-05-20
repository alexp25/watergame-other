
import numpy as np
import generator, preprocessing
import math

if __name__ == "__main__":
    prep = preprocessing.Preprocessing()
    # data = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    # data = [["A"], ["B"], ["C"]]
    data = [["000"], ["001"], ["010"], ["011"],
            ["100"], ["101"], ["110"], ["111"]]

    binv = generator.generate_binary(6)
    print("binv:")
    print(binv)
    binv = prep.adapt_input(binv)
    print("adapted:")
    print(binv)
    print("to list")
    # data = prep.str_to_list(binv)
    data = binv
    encoded = prep.create_encoder(data)

    print("encoded: ")
    print(encoded)

    encoded1 = prep.encode([["000000"], ["000001"]])

    print("binary to int")
    print(prep.binary_to_int("000110"))

    # print("data to int")
    print(prep.decode_int(data))

    print("encoded sample")
    print(encoded1)

    print("one hot encoded decoded")
    print(prep.decode(encoded1))

    quit()

    print("one hot encoded to list")
    print(prep.decode_int_onehot(encoded1))

    decoded = prep.decode(encoded)
    print("decoded")
    print(decoded)
    decoded = prep.str_to_list(decoded)
    print("decoded")
    print(decoded)
