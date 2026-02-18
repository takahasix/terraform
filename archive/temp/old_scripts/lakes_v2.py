"""
湖・窪地に水が溜まる地形シミュレーション（改良版）
- 完全に閉じた窪地を作成
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from noise import pnoise2
from landlab import RasterModelGrid
from landlab.components import (
    FlowAccumulator,
    DepressionFinderAndRouter,
)

# ========================================
# シンプルなテスト: 明確なクレーターを作る
# ========================================
nrows, ncols = 100, 100
dx = 10.0

# ベース地形（平坦 + 微小ノイズ）
np.random.seed(42)
terrain_2d = np.random.rand(nrows, ncols) * 5 + 50  # 50-55mのベース

# クレーター1（閉じた窪地）
cy1, cx1, r1 = 30, 30, 15
for i in range(nrows):
    for j in range(ncols):
        dist = np.sqrt((i - cy1)**2 + (j - cx1)**2)
        if dist < r1:
            # 縁を高くして中央を低くする
            terrain_2d[i, j] = 70 - 40 * (1 - dist/r1)**2  # 中央30m、縁70m

# クレーター2（より深い）
cy2, cx2, r2 = 70, 60, 20
for i in range(nrows):
    for j in range(ncols):
        dist = np.sqrt((i - cy2)**2 + (j - cx2)**2)
        if dist < r2:
            terrain_2d[i, j] = 80 - 60 * (1 - dist/r2)**2  # 中央20m、縁80m

# 山（水源）
cy3, cx3, r3 = 50, 80, 12
for i in range(nrows):
    for j in range(ncols):
        dist = np.sqrt((i - cy3)**2 + (j - cx3)**2)
        if dist < r3:
            terrain_2d[i, j] = 50 + 50 * (1 - dist/r3)  # 山頂100m

print(f"Terrain range: {terrain_2d.min():.1f} - {terrain_2d.max():.1f} m")

# ========================================
# Landlab設定
# ========================================
mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
z = mg.add_zeros("topographic__elevation", at="node")
z += terrain_2d.ravel()

# 全境界を閉じる（島のような状態）→ 一箇所だけ開ける
mg.set_closed_boundaries_at_grid_edges(True, True, True, True)
# 左下角だけ開放（海への出口）
mg.status_at_node[0] = mg.BC_NODE_IS_FIXED_VALUE

print(f"\nBoundary: all closed except node 0 (outlet)")

# ========================================
# 流路計算 + 窪地検出
# ========================================
fa = FlowAccumulator(mg, flow_director='D8', depression_finder='DepressionFinderAndRouter')
fa.run_one_step()

# 窪地情報
if 'depression__depth' in mg.at_node:
    depression_depth = mg.at_node['depression__depth']
    print(f"Depression depth: max={depression_depth.max():.1f} m")
else:
    print("No depression depth calculated")

if 'depression__outlet_node' in mg.at_node:
    outlet = mg.at_node['depression__outlet_node']
    in_depression = outlet != -1
    depression_count = np.sum(in_depression)
    print(f"Nodes in depressions: {depression_count}")
else:
    in_depression = np.zeros(mg.number_of_nodes, dtype=bool)

# ========================================
# 水を溜める（手動シミュレーション）
# ========================================
print("\n=== Manual lake filling ===")

# 窪地を見つける: 周囲より低いローカルミニマム
from scipy.ndimage import minimum_filter, label

elev_2d = z.reshape((nrows, ncols))

# ローカルミニマム検出
local_min = minimum_filter(elev_2d, size=5) == elev_2d
# 境界は除外
local_min[0, :] = False
local_min[-1, :] = False
local_min[:, 0] = False
local_min[:, -1] = False

min_points = np.where(local_min)
print(f"Local minima found: {len(min_points[0])}")

# 各窪地を水で満たす（単純化: 縁の最低点まで水を入れる）
water_level = elev_2d.copy()
lake_mask = np.zeros((nrows, ncols), dtype=bool)

for i, (my, mx) in enumerate(zip(min_points[0], min_points[1])):
    min_elev = elev_2d[my, mx]
    
    # 周囲を広げながら、流出点（縁の最低点）を見つける
    visited = np.zeros((nrows, ncols), dtype=bool)
    to_visit = [(my, mx)]
    basin_cells = []
    rim_elevations = []
    
    while to_visit:
        cy, cx = to_visit.pop()
        if visited[cy, cx]:
            continue
        visited[cy, cx] = True
        
        # この点より低いか同じ高さなら盆地の一部
        if elev_2d[cy, cx] <= min_elev + 30:  # 閾値
            basin_cells.append((cy, cx))
            # 隣接セルを追加
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < nrows and 0 <= nx < ncols and not visited[ny, nx]:
                    to_visit.append((ny, nx))
        else:
            rim_elevations.append(elev_2d[cy, cx])
    
    if rim_elevations:
        spill_level = min(rim_elevations)  # 溢れる水位
        if spill_level > min_elev + 5:  # 十分な深さがある
            for cy, cx in basin_cells:
                if elev_2d[cy, cx] < spill_level:
                    water_level[cy, cx] = spill_level
                    lake_mask[cy, cx] = True

water_depth = water_level - elev_2d
water_depth[water_depth < 0.5] = 0
lake_mask = water_depth > 0.5

print(f"Lake cells: {np.sum(lake_mask)}")
print(f"Max water depth: {water_depth.max():.1f} m")

# ========================================
# 可視化
# ========================================
fig = plt.figure(figsize=(16, 12))

drainage = mg.at_node["drainage_area"].reshape((nrows, ncols))

# ----- 1. 地形 -----
ax1 = fig.add_subplot(2, 3, 1)
im = ax1.imshow(elev_2d, cmap='terrain', origin='lower')
ax1.set_title('Terrain (with craters)')
plt.colorbar(im, ax=ax1, label='Elevation [m]')

# ----- 2. 集水面積（河川） -----
ax2 = fig.add_subplot(2, 3, 2)
im = ax2.imshow(np.log10(drainage + 1), cmap='Blues', origin='lower')
ax2.set_title('Drainage Area')
plt.colorbar(im, ax=ax2, label='log10(area)')

# ----- 3. 湖 -----
ax3 = fig.add_subplot(2, 3, 3)
ax3.imshow(elev_2d, cmap='terrain', origin='lower', alpha=0.7)
lake_display = np.ma.masked_where(~lake_mask, water_depth)
im = ax3.imshow(lake_display, cmap='Blues', origin='lower')
ax3.set_title(f'Lakes (depth)')
plt.colorbar(im, ax=ax3, label='Depth [m]')

# ----- 4. 3D -----
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))
ax4.plot_surface(X, Y, elev_2d, cmap='terrain', alpha=0.8, linewidth=0)

# 水面
water_surface = np.where(lake_mask, water_level, np.nan)
ax4.plot_surface(X, Y, water_surface, color='blue', alpha=0.6, linewidth=0)

ax4.set_title('3D: Terrain + Lakes')
ax4.view_init(elev=50, azim=45)

# ----- 5. 断面図 -----
ax5 = fig.add_subplot(2, 3, 5)
section_row = 70  # クレーター2を通る

ax5.fill_between(range(ncols), 0, elev_2d[section_row, :], color='brown', alpha=0.5, label='Ground')
ax5.plot(range(ncols), elev_2d[section_row, :], 'k-', linewidth=2)

# 水面
for j in range(ncols):
    if lake_mask[section_row, j]:
        ax5.fill_between([j-0.5, j+0.5], 
                         [elev_2d[section_row, j], elev_2d[section_row, j]],
                         [water_level[section_row, j], water_level[section_row, j]],
                         color='blue', alpha=0.7)

ax5.set_xlabel('X')
ax5.set_ylabel('Elevation [m]')
ax5.set_title(f'Cross section at Y={section_row}')
ax5.legend()
ax5.grid(True, alpha=0.3)

# ----- 6. 全部まとめ -----
ax6 = fig.add_subplot(2, 3, 6)
ax6.imshow(elev_2d, cmap='terrain', origin='lower', alpha=0.7)

# 河川
river_thresh = np.percentile(drainage, 95)
river_mask = drainage > river_thresh
river_disp = np.ma.masked_where(~river_mask, np.ones_like(elev_2d))
ax6.imshow(river_disp, cmap='cool', origin='lower', alpha=0.5)

# 湖
ax6.imshow(np.ma.masked_where(~lake_mask, water_depth), cmap='Blues', origin='lower', alpha=0.9)

ax6.set_title('Terrain + Rivers + Lakes')

plt.tight_layout()
plt.savefig("terrain_with_lakes_v2.png", dpi=150)
print(f"\nSaved: terrain_with_lakes_v2.png")
plt.show()
