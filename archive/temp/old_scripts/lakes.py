"""
湖・窪地に水が溜まる地形シミュレーション
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from noise import pnoise2
from landlab import RasterModelGrid
from landlab.components import (
    FlowAccumulator,
    DepressionFinderAndRouter,
    LakeMapperBarnes,
)

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
# 地形生成（窪地ができやすいように調整）
# ========================================
nrows, ncols = 100, 100
dx = 10.0

terrain_2d = generate_ridge_terrain(nrows, ncols, seed=42) * 300.0

# 盆地効果を強めに（中央が低くなりやすい）
center_y, center_x = nrows // 2, ncols // 2
for i in range(nrows):
    for j in range(ncols):
        dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        terrain_2d[i, j] *= 1 - 0.4 * (dist / max_dist)  # 0.3→0.4に強化

# 人工的に窪地を追加（湖ができる場所）
# 窪地1
terrain_2d[30:40, 30:40] -= 50
# 窪地2
terrain_2d[60:75, 50:65] -= 80

print(f"Terrain range: {terrain_2d.min():.1f} - {terrain_2d.max():.1f} m")

# ========================================
# Landlab設定
# ========================================
mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
z = mg.add_zeros("topographic__elevation", at="node")
z += terrain_2d.ravel()

# 境界：一部のみ開放（水が溜まりやすくする）
mg.set_closed_boundaries_at_grid_edges(True, True, True, False)

# ========================================
# 方法1: DepressionFinderAndRouter
# ========================================
print("\n=== DepressionFinderAndRouter ===")

fa1 = FlowAccumulator(mg, flow_director='D8')
fa1.run_one_step()

df = DepressionFinderAndRouter(mg)
df.map_depressions()

# 窪地情報を取得
depression_depth = mg.at_node.get("depression__depth", np.zeros(mg.number_of_nodes))
flood_status = mg.at_node.get("flood_status_code", np.zeros(mg.number_of_nodes))

print(f"Depression depths: min={depression_depth.min():.1f}, max={depression_depth.max():.1f}")

# ========================================
# 方法2: LakeMapperBarnes（より詳細な湖計算）
# ========================================
print("\n=== LakeMapperBarnes ===")

# 新しいグリッドで試す
mg2 = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
z2 = mg2.add_zeros("topographic__elevation", at="node")
z2 += terrain_2d.ravel()
mg2.set_closed_boundaries_at_grid_edges(True, True, True, False)

fa2 = FlowAccumulator(mg2, flow_director='D8')
fa2.run_one_step()

lmb = LakeMapperBarnes(mg2, method='D8', fill_flat=False, 
                        surface='topographic__elevation',
                        fill_surface='topographic__elevation',
                        redirect_flow_steepest_descent=True,
                        track_lakes=True)
lmb.run_one_step()

# 湖の情報
lake_map = mg2.at_node.get('lake__id', np.full(mg2.number_of_nodes, -1))
water_surface = mg2.at_node.get('water_surface__elevation', z2.copy())

# 湖のID一覧
unique_lakes = np.unique(lake_map[lake_map >= 0])
print(f"Number of lakes detected: {len(unique_lakes)}")

# 水深計算（水面 - 地形）
water_depth = water_surface - z2
water_depth[water_depth < 0.01] = 0  # ほぼ0は0に

print(f"Water depth: max={water_depth.max():.1f} m")

# ========================================
# 可視化
# ========================================
fig = plt.figure(figsize=(16, 12))

# 2Dに変換
elev_2d = z2.reshape((nrows, ncols))
water_depth_2d = water_depth.reshape((nrows, ncols))
lake_map_2d = lake_map.reshape((nrows, ncols))
water_surface_2d = water_surface.reshape((nrows, ncols))
drainage_2d = mg2.at_node["drainage_area"].reshape((nrows, ncols))

# ----- 1. 元の地形 -----
ax1 = fig.add_subplot(2, 3, 1)
im = ax1.imshow(elev_2d, cmap='terrain', origin='lower')
ax1.set_title('Original Terrain')
plt.colorbar(im, ax=ax1, label='Elevation [m]')

# ----- 2. 湖のマップ -----
ax2 = fig.add_subplot(2, 3, 2)
ax2.imshow(elev_2d, cmap='terrain', origin='lower', alpha=0.5)
lake_display = np.ma.masked_where(lake_map_2d < 0, lake_map_2d)
im = ax2.imshow(lake_display, cmap='Blues', origin='lower', alpha=0.8)
ax2.set_title(f'Lakes detected ({len(unique_lakes)} lakes)')
plt.colorbar(im, ax=ax2, label='Lake ID')

# ----- 3. 水深 -----
ax3 = fig.add_subplot(2, 3, 3)
ax3.imshow(elev_2d, cmap='terrain', origin='lower', alpha=0.5)
depth_display = np.ma.masked_where(water_depth_2d < 0.1, water_depth_2d)
im = ax3.imshow(depth_display, cmap='Blues', origin='lower')
ax3.set_title('Water Depth')
plt.colorbar(im, ax=ax3, label='Depth [m]')

# ----- 4. 3D地形 + 湖 -----
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))

# 地形
ax4.plot_surface(X, Y, elev_2d, cmap='terrain', alpha=0.7, linewidth=0)

# 水面（湖がある場所のみ）
water_surface_display = np.where(water_depth_2d > 0.1, water_surface_2d, np.nan)
ax4.plot_surface(X, Y, water_surface_display, color='blue', alpha=0.5, linewidth=0)

ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Elevation [m]')
ax4.set_title('3D Terrain + Lakes (blue)')
ax4.view_init(elev=45, azim=45)

# ----- 5. 断面図 -----
ax5 = fig.add_subplot(2, 3, 5)
mid_row = 65  # 窪地2を通る行

ax5.fill_between(range(ncols), 0, elev_2d[mid_row, :], color='brown', alpha=0.5, label='Ground')
ax5.plot(range(ncols), elev_2d[mid_row, :], 'k-', linewidth=2)

# 水面
ax5.fill_between(range(ncols), elev_2d[mid_row, :], water_surface_2d[mid_row, :], 
                  where=water_depth_2d[mid_row, :] > 0.1,
                  color='blue', alpha=0.5, label='Lake')
ax5.plot(range(ncols), water_surface_2d[mid_row, :], 'b-', linewidth=1, alpha=0.7)

ax5.set_xlabel('X')
ax5.set_ylabel('Elevation [m]')
ax5.set_title(f'Cross section at Y={mid_row}')
ax5.legend()
ax5.grid(True, alpha=0.3)

# ----- 6. 河川 + 湖 -----
ax6 = fig.add_subplot(2, 3, 6)
ax6.imshow(elev_2d, cmap='terrain', origin='lower', alpha=0.7)

# 河川
river_threshold = np.percentile(drainage_2d, 95)
river_mask = drainage_2d > river_threshold
river_display = np.ma.masked_where(~river_mask, np.ones_like(elev_2d))
ax6.imshow(river_display, cmap='cool', origin='lower', alpha=0.7)

# 湖
lake_display = np.ma.masked_where(water_depth_2d < 0.1, water_depth_2d)
ax6.imshow(lake_display, cmap='Blues', origin='lower', alpha=0.8)

ax6.set_title('Terrain + Rivers + Lakes')

plt.tight_layout()
plt.savefig("terrain_with_lakes.png", dpi=150)
print(f"\nSaved: terrain_with_lakes.png")
plt.show()
