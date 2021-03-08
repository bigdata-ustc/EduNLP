# coding: utf-8
# 2021/3/8 @ tongshiwei

import matplotlib.pyplot as plt

ax = plt.gca()
ax.clear()
ax.set_axis_off()
ax_width = ax.get_window_extent().width
ax_height = ax.get_window_extent().height
ax.text(0, 0, "123")
ax.text(
    0, 0, "Direction", ha="center", va="center", rotation=45, size=15,
    bbox=dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="b", lw=2)
)
plt.show()