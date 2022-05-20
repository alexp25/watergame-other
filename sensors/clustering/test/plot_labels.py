import matplotlib.pyplot as plt
w = 4
h = 3
d = 70
plt.figure(figsize=(w, h), dpi=d)
x = [0, 1, 2, 3, 4, 5]
y = [0, 2, 3, 5, 7, 9]
positions = (2, 3, 4)
labels = ("B", "C", "D")
plt.xticks(positions, labels)
plt.plot(x, y)
plt.show()
