import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from io import StringIO

def safe_loadtxt(filename, fill_value=-1e9):
    def parse_line(line):
        return (line.replace('-nan(ind)', str(fill_value))
                    .replace('nan', str(fill_value))
                    .replace('NaN', str(fill_value)))
    with open(filename, 'r') as f:
        lines = [parse_line(l) for l in f]
    return np.loadtxt(StringIO(''.join(lines)))

root = os.getcwd()
cases = [d for d in os.listdir(root) if os.path.isdir(d) and "case" in d]

if len(cases) == 0:
    print("No case folders found")
    sys.exit(1)

print("Available cases:")
for i, c in enumerate(cases):
    print(i, c)

idx = int(input("Select case index: "))
case = cases[idx]

# -------------------- Files --------------------
x_file     = os.path.join(case, "mesh.txt")
time_file  = os.path.join(case, "time.txt")
wall_file  = os.path.join(case, "T_wall.txt")
na_file    = os.path.join(case, "T_sodium.txt")

for f in [x_file, time_file, wall_file, na_file]:
    if not os.path.isfile(f):
        print("Missing file:", f)
        sys.exit(1)

# -------------------- Load data --------------------
x    = safe_loadtxt(x_file)
time = safe_loadtxt(time_file)
T_w  = safe_loadtxt(wall_file)      # 2D: [time, space]
T_Na = safe_loadtxt(na_file)

names = ["Wall temperature", "Na temperature"]
units = ["[K]", "[K]"]
Y = [T_w, T_Na]

# -------------------- Utils --------------------
def robust_ylim(y):
    if y.ndim > 1:
        vals = y[50:, :].flatten()
    else:
        vals = y
    lo, hi = np.percentile(vals, [1, 99])
    if lo == hi:
        lo, hi = np.min(vals), np.max(vals)
    margin = 0.1 * (hi - lo)
    return lo - margin, hi + margin

def time_to_index(t):
    return np.searchsorted(time, t, side='left')

def index_to_time(i):
    return time[i]

# -------------------- Figure --------------------
fig, ax = plt.subplots(figsize=(11, 6))
plt.subplots_adjust(left=0.08, bottom=0.25, right=0.75)

line, = ax.plot([], [], lw=2)
ax.set_xlabel("Axial length [m]")
ax.grid(True)

ax_slider = plt.axes([0.08, 0.10, 0.50, 0.03])
slider = Slider(ax_slider, "Time [s]", time.min(), time.max(), valinit=time[0])

# -------------------- Buttons --------------------
buttons = []
button_names = ["Wall", "Na"]
current_idx = 0

for j, name in enumerate(button_names):
    bx = plt.axes([0.80, 0.75 - j*0.10, 0.15, 0.08])
    btn = Button(bx, name)
    buttons.append(btn)

# -------------------- State --------------------
current_frame = [0]

ax.set_title(f"{names[current_idx]} {units[current_idx]}")
ax.set_xlim(x.min(), x.max())
ax.set_ylim(*robust_ylim(Y[current_idx]))

# -------------------- Draw --------------------
def draw_frame(i, update_slider=True):
    y = Y[current_idx]
    line.set_data(x, y[i, :])

    if update_slider:
        slider.disconnect(slider_cid)
        slider.set_val(index_to_time(i))
        connect_slider()

    return line,

def slider_update(val):
    i = time_to_index(val)
    current_frame[0] = i
    draw_frame(i, update_slider=False)
    fig.canvas.draw_idle()

def connect_slider():
    global slider_cid
    slider_cid = slider.on_changed(slider_update)

connect_slider()

def change_variable(idx):
    global current_idx
    current_idx = idx
    ax.set_title(f"{names[idx]} {units[idx]}")
    ax.set_ylim(*robust_ylim(Y[idx]))
    current_frame[0] = 0
    draw_frame(0)
    fig.canvas.draw_idle()

for j, btn in enumerate(buttons):
    btn.on_clicked(lambda event, k=j: change_variable(k))

plt.show()
