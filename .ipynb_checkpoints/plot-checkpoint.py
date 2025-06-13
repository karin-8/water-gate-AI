import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import numpy as np

# import streamlit as st
# from matplotlib.animation import FuncAnimation
import tempfile
import os

from matplotlib.colors import LinearSegmentedColormap

# Define two custom blues (RGB or hex)
light_blue =  (0.2, 0.6, 1.0)
dark_blue  =  (0.0, 0.25, 0.8)

# Create a colormap
custom_cmap = LinearSegmentedColormap.from_list("custom_blue", [light_blue, dark_blue])


# def plot_gates(gate_names, gate_heights, gate_positions, water_levels, qs, y_min=6, y_max=12):
#     # Plotting
#     fig, ax = plt.subplots(figsize=(12, 5))
#     bar_width = 1.0
#     bars = ax.bar(gate_positions, water_levels, width=bar_width, color='deepskyblue', align='edge')
    
#     # Max/Min lines
#     ax.axhline(y_max, color='red', linestyle='-', label='Max')
#     ax.axhline(y_min, color='orange', linestyle='-', label='Min')
    
#     # Add gates and callouts
#     for i, height in zip(gate_positions, gate_heights):
#         if i != 0:
#             gate_height_m = height
#             gate_open_ratio = height / 2
    
#             gate = Rectangle((i - 0.05, 0), 0.1, gate_height_m, facecolor='deepskyblue')
#             gatewall = Rectangle((i - 0.05, gate_height_m), 0.1, y_max - gate_height_m, edgecolor='black', facecolor='grey')
#             ax.add_patch(gate)
#             ax.add_patch(gatewall)
    
#             # Callout
#             ax.annotate(f"{round(height,2)} m\n{gate_open_ratio:.2%}",
#                         xy=(i, gate_height_m),
#                         xytext=(i + 0.2, gate_height_m + 2),
#                         arrowprops=dict(facecolor='black', arrowstyle='->'),
#                         ha='right', fontsize=9,
#                         bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black", lw=0.5))
    
#             # White dotted arrow
#             arrow_y = gate_height_m / 2
#             ax.annotate("",
#                         xy=(i + 0.08, arrow_y), xytext=(i - 0.06, arrow_y),
#                         arrowprops=dict(arrowstyle="->", linestyle=":", color='white', lw=2))
    
#     # Water level labels
#     for i, h in zip(gate_positions, water_levels):
#         if i != 0:
#             ax.text(i + 0.5, h - 0.5, f"+{round(h,2)} m", ha='center', color='white', fontsize=10)
    
#     # X labels
#     ax.set_xticks(gate_positions)
#     ax.set_xticklabels([f"{gate_names[i]}\nQ = {round(qs[i],2)} cms" for i in range(len(qs))])
    
#     # Aesthetic
#     ax.set_ylim(0, 16)
#     ax.set_ylabel("Height (m)")
#     ax.set_title("Water Levels and Gate Openings")
#     ax.legend()
    
#     plt.grid(axis='y', linestyle='--', alpha=0.4)
#     plt.tight_layout()

#     return fig

def plot_gates(gate_names, gate_heights, gate_positions, water_levels, qs, current_levels=None, y_min=6, y_max=12):
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 5))
    bar_width = 1.0
    bars = ax.bar(gate_positions, water_levels, width=bar_width, color='deepskyblue', align='edge')
    
    # Max/Min lines
    ax.axhline(y_max, color='red', linestyle='-', label='Max')
    ax.axhline(y_min, color='orange', linestyle='-', label='Min')
    
    # Current water levels as grey dashed lines
    if current_levels:
        for i, level in zip(gate_positions, current_levels):
            ax.hlines(level, i+1, i+1 + bar_width, colors='grey', linestyles='--', linewidth=2, label='Current Level' if i == gate_positions[1] else "")
    
    # Add gates and callouts
    for i, height in zip(gate_positions, gate_heights):
        if i != 0:
            gate_height_m = height
            gate_open_ratio = height / 2
    
            gate = Rectangle((i - 0.05, 0), 0.1, gate_height_m, facecolor='deepskyblue')
            gatewall = Rectangle((i - 0.05, gate_height_m), 0.1, y_max - gate_height_m, edgecolor='black', facecolor='grey')
            ax.add_patch(gate)
            ax.add_patch(gatewall)
    
            # Callout
            ax.annotate(f"{round(height,2)} m\n{gate_open_ratio:.2%}",
                        xy=(i, gate_height_m),
                        xytext=(i + 0.2, gate_height_m + 2),
                        arrowprops=dict(facecolor='black', arrowstyle='->'),
                        ha='right', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black", lw=0.5))
    
            # White dotted arrow
            arrow_y = gate_height_m / 2
            ax.annotate("",
                        xy=(i + 0.08, arrow_y), xytext=(i - 0.06, arrow_y),
                        arrowprops=dict(arrowstyle="->", linestyle=":", color='white', lw=2))
    
    # Water level labels
    for i, h in zip(gate_positions, water_levels):
        if i != 0:
            ax.text(i + 0.5, h - 0.5, f"+{round(h,2)} m", ha='center', color='white', fontsize=10)
    
    # X labels
    ax.set_xticks(gate_positions)
    ax.set_xticklabels([f"{gate_names[i]}\nQ = {round(qs[i],2)} cms" for i in range(len(qs))])
    
    # Aesthetic
    ax.set_ylim(0, 16)
    ax.set_ylabel("Height (m)")
    ax.set_title("Water Levels and Gate Openings")
    ax.legend()
    
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    return fig


# def animate_ripple_color(gate_names, gate_heights, gate_positions, water_levels, qs, y_min=6, y_max=12):
#     fig, ax = plt.subplots(figsize=(12, 5))
#     bar_width = 1.0

#     # Base gate bars (transparent — we'll fill with ripples instead)
#     ax.axhline(y_max, color='red', linestyle='-', label='Max')
#     ax.axhline(y_min, color='orange', linestyle='-', label='Min')

#     # Draw gate structures and callouts
#     for i, height in zip(gate_positions, gate_heights):
#         if i != 0:
#             gate_height_m = height
#             gate_open_ratio = height / 2

#             gate = Rectangle((i - 0.05, 0), 0.1, gate_height_m, facecolor='deepskyblue')
#             gatewall = Rectangle((i - 0.05, gate_height_m), 0.1, y_max - gate_height_m,
#                                  edgecolor='black', facecolor='grey')
#             ax.add_patch(gate)
#             ax.add_patch(gatewall)

#             ax.annotate(f"{round(height,2)} m\n{gate_open_ratio:.2%}",
#                         xy=(i, gate_height_m),
#                         xytext=(i + 0.2, gate_height_m + 2),
#                         arrowprops=dict(facecolor='black', arrowstyle='->'),
#                         ha='right', fontsize=9,
#                         bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black", lw=0.5))

#             arrow_y = gate_height_m / 2
#             ax.annotate("",
#                         xy=(i + 0.08, arrow_y), xytext=(i - 0.06, arrow_y),
#                         arrowprops=dict(arrowstyle="->", linestyle=":", color='white', lw=2))

#     # Water level labels and ripple containers
#     img_objs = []
#     grid_x = np.linspace(0, 1, 100).reshape(1, -1)
#     for i, h in enumerate(water_levels):
#         # Add level label
#         ax.text(gate_positions[i] + 0.5, h - 0.5, f"+{round(h,2)} m", ha='center', color='white', fontsize=10)

#         # Simulated blue ripple
#         base_wave = 0.5 * (np.sin(20 * np.pi * grid_x * 4) + 1)
#         extent = (gate_positions[i], gate_positions[i] + bar_width, 0, h)
#         im = ax.imshow(base_wave, extent=extent, aspect='auto',
#                        cmap=custom_cmap, vmin=0, vmax=0.5, alpha=0.9)
#         img_objs.append(im)

#     # X-axis labels
#     ax.set_xticks(gate_positions)
#     ax.set_xticklabels([f"{gate_names[i]}\nQ = {round(qs[i], 2)} cms" for i in range(len(qs))])

#     # Aesthetic settings
#     ax.set_ylim(0, 16)
#     ax.set_ylabel("Height (m)")
#     ax.set_title("Water Levels and Gate Openings (Ripple Fill)")
#     ax.legend()
#     plt.grid(axis='y', linestyle='--', alpha=0.4)
#     plt.tight_layout()

#     # Animate ripple color only (not shape or height)
#     def update(frame):
#         for im in img_objs:
#             phase = frame / 10.0
#             wave = 0.1 * (np.sin(2 * np.pi * grid_x * 4 - phase) + 1)
#             im.set_data(wave)

#     ani = FuncAnimation(fig, update, frames=60, interval=100, repeat=True)
#     plt.show()
#     # return ani

#     tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
#     ani.save(tmpfile.name, writer='pillow')
#     plt.close(fig)
#     return tmpfile.name
    # return ani

# def animate_ripple_color(gate_names, gate_heights, gate_positions, water_levels, qs, y_min=6, y_max=12):
#     fig, ax = plt.subplots(figsize=(12, 5))
#     bar_width = 1.0
#     ax.set_xlim(-1, max(gate_positions) + 2)
#     ax.set_ylim(0, 16)
#     ax.axis('off')

#     # Optional custom blue gradient (no white!)
#     custom_cmap = LinearSegmentedColormap.from_list("ocean_blue", ["#0077be", "#00bfff", "#1ca9c9"])

#     # Max/Min lines
#     ax.axhline(y_max, color='red', linestyle='-', label='Max')
#     ax.axhline(y_min, color='orange', linestyle='-', label='Min')

#     # Draw gate structures
#     for i, height in zip(gate_positions, gate_heights):
#         if i != 0:
#             gate = Rectangle((i - 0.05, 0), 0.1, height, facecolor='deepskyblue')
#             gatewall = Rectangle((i - 0.05, height), 0.1, y_max - height, edgecolor='black', facecolor='grey')
#             ax.add_patch(gate)
#             ax.add_patch(gatewall)

#             # Callout
#             ax.annotate(f"{round(height,2)} m\n{height / 2:.2%}",
#                         xy=(i, height),
#                         xytext=(i + 0.2, height + 2),
#                         arrowprops=dict(facecolor='black', arrowstyle='->'),
#                         ha='right', fontsize=9,
#                         bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black", lw=0.5))

#             ax.annotate("",
#                         xy=(i + 0.08, height / 2), xytext=(i - 0.06, height / 2),
#                         arrowprops=dict(arrowstyle="->", linestyle=":", color='white', lw=2))

#     # Water level labels and animated ripple fills
#     img_objs = []
#     grid_x = np.linspace(0, 1, 100).reshape(1, -1)
#     for i, h in enumerate(water_levels):
#         ax.text(gate_positions[i] + 0.5, h - 0.5, f"+{round(h,2)} m", ha='center', color='white', fontsize=10)

#         wave = 0.1 * (np.sin(2 * np.pi * grid_x * 4) + 1)
#         extent = (gate_positions[i], gate_positions[i] + bar_width, 0, h)
#         im = ax.imshow(wave, extent=extent, aspect='auto',
#                        cmap=custom_cmap, vmin=0, vmax=0.2, alpha=0.95)
#         img_objs.append(im)

#     # X-axis
#     ax.set_xticks(gate_positions)
#     ax.set_xticklabels([f"{gate_names[i]}\nQ = {round(qs[i], 2)} cms" for i in range(len(qs))])
#     ax.set_ylabel("Height (m)")
#     ax.set_title("Water Levels and Gate Openings (Ripple Fill)")
#     ax.legend()
#     plt.grid(axis='y', linestyle='--', alpha=0.4)
#     plt.tight_layout()

#     def update(frame):
#         phase = frame / 5.0  # faster phase change = smoother animation
#         wave = 0.1 * (np.sin(2 * np.pi * grid_x * 4 - phase) + 1)
#         for im in img_objs:
#             im.set_data(wave)

#     ani = FuncAnimation(fig, update, frames=60, interval=60, repeat=True)

#     # Save as temp GIF
#     tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
#     ani.save(tmpfile.name, writer='pillow', fps=15)
#     plt.close(fig)
#     return tmpfile.name


