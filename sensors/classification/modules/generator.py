

# This function generates all n bit Gray
# codes and prints the generated codes
def generate_gray(n):

    # base case
    if (n <= 0):
        return

    # 'arr' will store all generated codes
    arr = list()

    # start with one-bit pattern
    arr.append([0])
    arr.append([1])

    # Every iteration of this loop generates
    # 2*i codes from previously generated i codes.
    i = 2
    j = 0
    while(True):

        if i >= 1 << n:
            break

        # Enter the prviously generated codes
        # again in arr[] in reverse order.
        # Nor arr[] has double number of codes.
        for j in range(i - 1, -1, -1):
            arr.append(arr[j])

        # append 0 to the first half
        for j in range(i):
            arr[j] = [0] + arr[j]

        # append 1 to the second half
        for j in range(i, 2 * i):
            arr[j] = [1] + arr[j]
        i = i << 1

    # prcontents of arr[]
    for i in range(len(arr)):
        print(arr[i])

    return arr

def generate_binary(n):
    arr = []
    for i in range(1 << n):
        s = bin(i)[2:]
        s = '0'*(n-len(s))+s
        arr.append(s)
        # print(list(s))
    return arr

if __name__ == "__main__":
    n = 3
    generate_gray(n)
    arr = generate_binary(n)
    print(arr)