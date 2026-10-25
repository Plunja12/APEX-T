import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import PowerNorm

# načtení dat
data = np.loadtxt("Mat2.csv", delimiter=",")
masked = np.ma.masked_where(data == 0, data)
# dostupné palety
cmaps = ["viridis_r", "magma_r"]

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.45)
ax.set_title("Mat2")

# počáteční hodnoty
init_cmap = cmaps[0]
init_gamma = 1
init_vmax = data.max()
init_vmin = data.min()

# první vykreslení
norm = PowerNorm(gamma=init_gamma, vmin=data.min(), vmax=init_vmax)

img = ax.imshow(masked, cmap=init_cmap, norm=norm)
cbar = plt.colorbar(img)

# --- SLIDER PRO MAXIMUM ---
ax_vmax = plt.axes([0.2, 0.35, 0.6, 0.05])
slider_vmax = Slider(ax_vmax, "Max", data.min(), data.max(), valinit=init_vmax)

# --- SLIDER PRO MIMIMUM ---
ax_vmin = plt.axes([0.2, 0.25, 0.6, 0.05])
slider_vmin = Slider(ax_vmin, "Min", data.min(), data.max(), valinit=init_vmin)


# --- SLIDER PRO PALETU ---
ax_cmap = plt.axes([0.2, 0.15, 0.6, 0.05])
slider_cmap = Slider(ax_cmap, "Paleta", 0, len(cmaps)-1, valinit=0, valstep=1)

# --- SLIDER PRO NORMALIZACI (GAMMA) ---
ax_gamma = plt.axes([0.2, 0.05, 0.6, 0.05])
slider_gamma = Slider(ax_gamma, "Gamma", 0.1, 1, valinit=init_gamma)

# funkce pro aktualizaci
def update(val):
    vmax = slider_vmax.val
    vmin = slider_vmin.val
    cmap = cmaps[int(slider_cmap.val)]
    gamma = slider_gamma.val

    # aktualizace normalizace
    norm = PowerNorm(gamma=gamma)

    img.set_norm(norm)
    img.set_cmap(cmap)
    img.set_clim(vmin=vmin, vmax=vmax)

    fig.canvas.draw_idle()

slider_vmax.on_changed(update)
slider_vmin.on_changed(update)
slider_cmap.on_changed(update)
slider_gamma.on_changed(update)

def update(val):
    vmin = slider_vmin.val
    vmax = slider_vmax.val
    cmap = cmaps[int(slider_cmap.val)]
    gamma = slider_gamma.val

    if vmin >= vmax:
        return

    # maskování hodnot pod minimem
    masked = np.ma.masked_where(data <= vmin, data)

    # nová normalizace
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    img.set_norm(norm)
    img.set_cmap(cmap)
    img.set_data(masked)

    fig.canvas.draw_idle()
    

plt.show()
