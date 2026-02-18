"""
地形の3D表示 - 河川との関係を確認
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from noise import pnoise2
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator

def generate_ridge_terrain(nrows, ncols, seed=42):
    terrain = np.zeros((nrows, ncols))
    for octave in range(6):
        freq = 2 ** octave
        amp = 0.5 ** octave
        for i in range(nrows):
            for j in range(ncols):
                val = pnoise2(i * freq / 150.0, j * freq / 150.0, base=seed + octave)
                terrain[i, j] += (1 - abs(val)) * amp
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    return terrain

# ========================================
# 地形生成
# ========================================
nrows, ncols = 100, 100  # 少し小さめで高速に
dx = 10.0

terrain_2d = generate_ridge_terrain(nrows, ncols, seed=42) * 300.0

# 盆地効果
center_y, center_x = nrows // 2, ncols // 2
for i in range(nrows):
    for j in range(ncols):
        dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        terrain_2d[i, j] *= 1 - 0.3 * (dist / max_dist)

# Landlab設定
mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
z = mg.add_zeros("topographic__elevation", at="node")
z += terrain_2d.ravel()

mg.set_closed_boundaries_at_grid_edges(True, True, True, False)

# 流路計算
fa = FlowAccumulator(mg, flow_director='D8')
fa.run_one_step()

drainage = mg.at_node["drainage_area"].reshape((nrows, ncols))
elevation = z.reshape((nrows, ncols))

# 河川マスク（上位5%）
river_threshold = np.percentile(drainage, 95)
river_mask = drainage > river_threshold

# ========================================
# 3D表示
# ========================================
fig = plt.figure(figsize=(16, 12))

# メッシュグリッド作成
X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))

# ----- 3D地形 (視点1) -----
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
surf = ax1.plot_surface(X, Y, elevation, cmap='terrain', alpha=0.9,
                        linewidth=0, antialiased=True)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Elevation [m]')
ax1.set_title('3D Terrain (view 1)')
ax1.view_init(elev=45, azim=45)

# ----- 3D地形 + 河川 (視点2) -----
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.plot_surface(X, Y, elevation, cmap='terrain', alpha=0.7,
                 linewidth=0, antialiased=True)

# 河川を赤い点で表示（高さを少し上げて見やすく）
river_y, river_x = np.where(river_mask)
river_z = elevation[river_mask] + 5  # 少し浮かせる
ax2.scatter(river_x, river_y, river_z, c='blue', s=2, alpha=0.8, label='Rivers')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Elevation [m]')
ax2.set_title('3D Terrain + Rivers (blue dots)')
ax2.view_init(elev=60, azim=30)
ax2.legend()

# ----- 断面図 -----
ax3 = fig.add_subplot(2, 2, 3)
mid_row = nrows // 2
ax3.fill_between(range(ncols), 0, elevation[mid_row, :], color='brown', alpha=0.5, label='Terrain')
ax3.plot(range(ncols), elevation[mid_row, :], 'k-', linewidth=2)

# この断面上の河川位置をマーク
river_in_section = river_mask[mid_row, :]
river_x_section = np.where(river_in_section)[0]
river_z_section = elevation[mid_row, river_in_section]
ax3.scatter(river_x_section, river_z_section, c='blue', s=50, zorder=5, label='Rivers')

ax3.set_xlabel('X')
ax3.set_ylabel('Elevation [m]')
ax3.set_title(f'Cross section at Y={mid_row}\n(rivers should be in valleys)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ----- 2D鳥瞰図 -----
ax4 = fig.add_subplot(2, 2, 4)
im = ax4.imshow(elevation, cmap='terrain', origin='lower')
# 河川を青で重ねる
river_display = np.ma.masked_where(~river_mask, np.ones_like(elevation))
ax4.imshow(river_display, cmap='Blues', origin='lower', alpha=0.9)
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_title('2D view: Terrain + Rivers')
plt.colorbar(im, ax=ax4, label='Elevation [m]')

plt.tight_layout()
plt.savefig("terrain_3d.png", dpi=150)
print("Saved: terrain_3d.png")
plt.show()
