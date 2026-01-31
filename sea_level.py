"""
海面レベル (Sea Level) を基準にした地形・湖・海のシミュレーション
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
nrows, ncols = 120, 120
dx = 10.0

# リッジ地形
terrain_2d = generate_ridge_terrain(nrows, ncols, seed=42)

# スケーリング: -50m（海底）〜 +250m（山頂）
# 海面レベル = 0m
terrain_2d = terrain_2d * 300 - 50  # 範囲: -50 ~ +250

# 島っぽくするため、端を下げる
center_y, center_x = nrows // 2, ncols // 2
for i in range(nrows):
    for j in range(ncols):
        dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        # 端に近いほど低くする（海に沈む）
        edge_depression = 150 * (dist / max_dist) ** 1.5
        terrain_2d[i, j] -= edge_depression

# ========================================
# 海面レベル設定
# ========================================
SEA_LEVEL = 0.0  # 海面 = 0m

print(f"=== Sea Level Simulation ===")
print(f"Sea Level: {SEA_LEVEL} m")
print(f"Terrain range: {terrain_2d.min():.1f} ~ {terrain_2d.max():.1f} m")

# 海（海面より低い部分）
sea_mask = terrain_2d < SEA_LEVEL
land_mask = ~sea_mask

print(f"Sea cells: {np.sum(sea_mask)} ({100*np.sum(sea_mask)/terrain_2d.size:.1f}%)")
print(f"Land cells: {np.sum(land_mask)} ({100*np.sum(land_mask)/terrain_2d.size:.1f}%)")

# ========================================
# Landlab設定 - 境界を海面レベルに
# ========================================
mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
z = mg.add_zeros("topographic__elevation", at="node")
z += terrain_2d.ravel()

# 境界ノード（端）を海面レベルに設定
# これにより水は端（海）に向かって流れる
boundary_nodes = mg.boundary_nodes
z[boundary_nodes] = SEA_LEVEL

# 境界を固定値として設定
mg.status_at_node[boundary_nodes] = mg.BC_NODE_IS_FIXED_VALUE

print(f"Boundary nodes set to sea level ({SEA_LEVEL}m)")

# ========================================
# 流路計算
# ========================================
fa = FlowAccumulator(mg, flow_director='D8')
fa.run_one_step()

drainage = mg.at_node["drainage_area"].reshape((nrows, ncols))
elev_2d = z.reshape((nrows, ncols))

# ========================================
# 水面計算
# ========================================
# 海: 地形が海面より低いところ
water_surface = np.maximum(elev_2d, SEA_LEVEL)  # 水面は最低でも海面レベル
water_depth = water_surface - elev_2d
water_depth[elev_2d >= SEA_LEVEL] = 0  # 陸地は水深0

# 海岸線
coastline = np.zeros_like(elev_2d, dtype=bool)
from scipy.ndimage import binary_dilation
coastline = binary_dilation(sea_mask) & land_mask

# ========================================
# 可視化
# ========================================
fig = plt.figure(figsize=(16, 12))

# ----- 1. 地形（海面を境界に） -----
ax1 = fig.add_subplot(2, 3, 1)
im = ax1.imshow(elev_2d, cmap='terrain', origin='lower', 
                vmin=-100, vmax=250)  # 海底〜山頂
ax1.contour(elev_2d, levels=[SEA_LEVEL], colors='blue', linewidths=2)  # 海岸線
ax1.set_title(f'Terrain\n(blue line = coastline at {SEA_LEVEL}m)')
plt.colorbar(im, ax=ax1, label='Elevation [m]')

# ----- 2. 陸地/海 マスク -----
ax2 = fig.add_subplot(2, 3, 2)
land_sea = np.where(land_mask, 1, 0)
ax2.imshow(land_sea, cmap='RdYlBu', origin='lower')
ax2.set_title('Land (red) / Sea (blue)')

# ----- 3. 水深 -----
ax3 = fig.add_subplot(2, 3, 3)
ax3.imshow(elev_2d, cmap='terrain', origin='lower', alpha=0.3,
           vmin=-100, vmax=250)
depth_display = np.ma.masked_where(water_depth < 1, water_depth)
im = ax3.imshow(depth_display, cmap='Blues', origin='lower')
ax3.set_title('Water Depth')
plt.colorbar(im, ax=ax3, label='Depth [m]')

# ----- 4. 3D表示 -----
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))

# 地形（海底も含む）
ax4.plot_surface(X, Y, elev_2d, cmap='terrain', alpha=0.8, 
                 linewidth=0, antialiased=True, vmin=-100, vmax=250)

# 海面（平らな面）
sea_surface = np.full_like(elev_2d, SEA_LEVEL)
sea_surface[land_mask] = np.nan  # 陸地は表示しない
ax4.plot_surface(X, Y, sea_surface, color='blue', alpha=0.4, linewidth=0)

ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Elevation [m]')
ax4.set_title(f'3D: Terrain + Sea (level={SEA_LEVEL}m)')
ax4.view_init(elev=40, azim=45)

# ----- 5. 河川 + 海 -----
ax5 = fig.add_subplot(2, 3, 5)
ax5.imshow(elev_2d, cmap='terrain', origin='lower', vmin=-100, vmax=250)

# 河川（陸地のみ）
river_thresh = np.percentile(drainage[land_mask], 90)
river_mask = (drainage > river_thresh) & land_mask
river_disp = np.ma.masked_where(~river_mask, np.ones_like(elev_2d))
ax5.imshow(river_disp, cmap='cool', origin='lower', alpha=0.7)

# 海
sea_disp = np.ma.masked_where(~sea_mask, np.ones_like(elev_2d))
ax5.imshow(sea_disp, cmap='Blues', origin='lower', alpha=0.6)

# 海岸線
ax5.contour(elev_2d, levels=[SEA_LEVEL], colors='darkblue', linewidths=1.5)

ax5.set_title('Terrain + Rivers + Sea')

# ----- 6. 断面図 -----
ax6 = fig.add_subplot(2, 3, 6)
mid_row = nrows // 2

# 地形
ax6.fill_between(range(ncols), -150, elev_2d[mid_row, :], 
                  color='brown', alpha=0.6, label='Ground')
ax6.plot(range(ncols), elev_2d[mid_row, :], 'k-', linewidth=2)

# 海面
ax6.axhline(y=SEA_LEVEL, color='blue', linestyle='-', linewidth=2, label=f'Sea Level ({SEA_LEVEL}m)')
ax6.fill_between(range(ncols), elev_2d[mid_row, :], SEA_LEVEL,
                  where=elev_2d[mid_row, :] < SEA_LEVEL,
                  color='blue', alpha=0.5, label='Sea')

ax6.set_xlabel('X')
ax6.set_ylabel('Elevation [m]')
ax6.set_title(f'Cross section at Y={mid_row}')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_ylim(-150, 200)

plt.tight_layout()
plt.savefig("terrain_sea_level.png", dpi=150)
print(f"\nSaved: terrain_sea_level.png")
plt.show()
