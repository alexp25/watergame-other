import os

path = '.\\data'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))

accs = []
acc_min = 100
acc_max = 0

for f in files:
    print(f)
    with open(f, "r") as file:
        acc = file.read()
        acc_lines = acc.split("\n")
        if "accuracy" in acc_lines[0]:

            acc_line = acc_lines[0]
            acc1 = acc_line.split(": ")[1]
            print(acc1)

            acc1 = float(acc1) * 100
            accs.append(acc1)
            if acc1 < acc_min:
                acc_min = acc1
            if acc1 > acc_max:
                acc_max = acc1

print("\nexp accs: ")
for i, acc in enumerate(accs):
    print(i+1, "\t", acc)

print("\nmin: ", acc_min)
print("max: ", acc_max)
