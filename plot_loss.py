import os
import numpy as np
import matplotlib.pyplot as plt

def update_loss():
    losses = []
    with open(os.path.join(os.path.dirname(__file__), "train.log"), "r", encoding="utf-8") as f:
        fulltext = f.read()
        lines = fulltext.split("\n")
        for line in lines:
            fields = line.split("|")
            if len(fields) >= 4 and fields[3].strip()[0:5] == "Loss:":
                loss = float(fields[3].strip()[6:])
                losses.append(loss)
    x = np.array(list(range(0, len(losses))))
    y = np.array(losses)
    return x, y

def show_loss(ax):
    x, y = update_loss()
    ax.clear()
    ax.grid()
    ax.plot(x, y, linewidth=1.0, color="#ff0000")
    ax.figure.canvas.draw()

fig, ax = plt.subplots()

timer = fig.canvas.new_timer(interval=100)
timer.add_callback(show_loss, ax)
timer.start()

plt.show()
