import os
import numpy as np
import matplotlib.pyplot as plt

def load_csv_line(path):
    with open(path, "r") as f:
        line = f.read().strip()
    parts = [p.strip() for p in line.split(",")]
    parts = [p for p in parts if p]
    return np.array([float(p) for p in parts])

# -------------------- Select case --------------------
root = os.getcwd()
cases = [d for d in os.listdir(root) if os.path.isdir(d) and "case" in d]

print("Available cases:")
for i, c in enumerate(cases):
    print(i, c)

idx = int(input("Select case index: "))
case = cases[idx]

# -------------------- Files --------------------
f_mesh = os.path.join(case, "mesh.txt")
f_wall = os.path.join(case, "T_wall.txt")
f_sod  = os.path.join(case, "T_sodium.txt")

for f in (f_mesh, f_wall, f_sod):
    if not os.path.isfile(f):
        print("Missing file:", f)
        raise SystemExit

mesh  = load_csv_line(f_mesh)
T_wall = load_csv_line(f_wall)
T_sod  = load_csv_line(f_sod)

# -------------------- Plot --------------------
fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(mesh, T_wall, lw=2, label="T_wall")
ax.plot(mesh, T_sod,  lw=2, label="T_sodium")

ax.set_xlabel("Axial position [m]")
ax.set_ylabel("Temperature [K]")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()
