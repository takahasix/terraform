"""
海面レベル + 連結された河川ネットワーク
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

def trace_rivers(mg, drainage, threshold_percentile=90):
    """
    流向を使って河川を上流から下流までトレース
    集水面積が閾値を超えたセルから、海/境界まで連続した線を描く
    """
    nrows, ncols = drainage.shape
    flow_receiver = mg.at_node["flow__receiver_node"]
    
    # 河川の始点（集水面積が閾値を超えるセル）
    threshold = np.percentile(drainage, threshold_percentile)
    river_starts = np.where(drainage.ravel() > threshold)[0]
    
    # 河川マスク
    river_mask = np.zeros(mg.number_of_nodes, dtype=bool)
    
    # 各始点から下流へトレース
    for start_node in river_starts:
        node = start_node
        visited = set()
        
        while node not in visited:
            visited.add(node)
            river_mask[node] = True
            
            receiver = flow_receiver[node]
            
            # 自分自身に流れる（境界/シンク）なら終了
            if receiver == node:
                break
            
            node = receiver
    
    return river_mask.reshape((nrows, ncols))


def extract_river_lines(mg, drainage, threshold_percentile=85):
    """
    河川をライン（座標のリスト）として抽出
    """
    flow_receiver = mg.at_node["flow__receiver_node"]
    threshold = np.percentile(drainage, threshold_percentile)
    
    # 河川の上流端を見つける（閾値を超えるが、上流に閾値を超えるセルがない）
    drainage_flat = drainage.ravel()
    is_river = drainage_flat > threshold
    
    # 各河川セルから下流へトレース
    river_lines = []
    traced = np.zeros(mg.number_of_nodes, dtype=bool)
    
    # 集水面積が大きい順にソート（下流から処理）
    sorted_nodes = np.argsort(drainage_flat)[::-1]
    
    for start in sorted_nodes:
        if not is_river[start] or traced[start]:
            continue
        
        # この点から上流に遡って始点を見つける
        line_x = []
        line_y = []
        
        node = start
        while True:
            if traced[node]:
                # すでにトレース済みの河川に合流
                line_x.append(mg.x_of_node[node])
                line_y.append(mg.y_of_node[node])
                break
            
            line_x.append(mg.x_of_node[node])
            line_y.append(mg.y_of_node[node])
            traced[node] = True
            
            receiver = flow_receiver[node]
            if receiver == node:  # 境界/シンク
                break
            node = receiver
        
        if len(line_x) > 3:  # 短すぎる線は無視
            river_lines.append((line_x, line_y))
    
    return river_lines


# ========================================
# 地形生成
# ========================================
nrows, ncols = 120, 120
dx = 10.0

terrain_2d = generate_ridge_terrain(nrows, ncols, seed=42)
terrain_2d = terrain_2d * 300 - 50

center_y, center_x = nrows // 2, ncols // 2
for i in range(nrows):
    for j in range(ncols):
        dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        edge_depression = 150 * (dist / max_dist) ** 1.5
        terrain_2d[i, j] -= edge_depression

SEA_LEVEL = 0.0

# ========================================
# Landlab設定
# ========================================
mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
z = mg.add_zeros("topographic__elevation", at="node")
z += terrain_2d.ravel()

boundary_nodes = mg.boundary_nodes
z[boundary_nodes] = SEA_LEVEL
mg.status_at_node[boundary_nodes] = mg.BC_NODE_IS_FIXED_VALUE

# 流路計算
fa = FlowAccumulator(mg, flow_director='D8')
fa.run_one_step()

drainage = mg.at_node["drainage_area"].reshape((nrows, ncols))
elev_2d = z.reshape((nrows, ncols))

sea_mask = elev_2d < SEA_LEVEL
land_mask = ~sea_mask

print(f"=== River Network ===")

# ========================================
# 河川抽出（連結版）
# ========================================
# 方法1: マスクベース（トレースで連結）
river_mask = trace_rivers(mg, drainage, threshold_percentile=88)
river_mask_land = river_mask & land_mask  # 陸地のみ

# 方法2: ラインベース
river_lines = extract_river_lines(mg, drainage, threshold_percentile=85)
print(f"River segments extracted: {len(river_lines)}")

# ========================================
# 可視化
# ========================================
fig, axes = plt.subplots(2, 3, figsize=(16, 11))

# ----- 1. 元の細切れ河川（比較用） -----
ax = axes[0, 0]
ax.imshow(elev_2d, cmap='terrain', origin='lower', vmin=-100, vmax=200)

# 単純な閾値（細切れ）
simple_thresh = np.percentile(drainage[land_mask], 90)
simple_river = (drainage > simple_thresh) & land_mask
ax.imshow(np.ma.masked_where(~simple_river, simple_river), 
          cmap='Blues', origin='lower')
ax.imshow(np.ma.masked_where(~sea_mask, sea_mask), 
          cmap='Blues', origin='lower', alpha=0.5)
ax.set_title('Original (fragmented)')

# ----- 2. 連結河川（マスク版） -----
ax = axes[0, 1]
ax.imshow(elev_2d, cmap='terrain', origin='lower', vmin=-100, vmax=200)
ax.imshow(np.ma.masked_where(~river_mask_land, river_mask_land), 
          cmap='Blues', origin='lower')
ax.imshow(np.ma.masked_where(~sea_mask, sea_mask), 
          cmap='Blues', origin='lower', alpha=0.5)
ax.set_title('Connected (mask-based)')

# ----- 3. 連結河川（ライン版） -----
ax = axes[0, 2]
ax.imshow(elev_2d, cmap='terrain', origin='lower', vmin=-100, vmax=200)

# 河川を線で描画（太さは集水面積に応じて）
for line_x, line_y in river_lines:
    # 座標をグリッドインデックスに変換
    ix = [int(x/dx) for x in line_x]
    iy = [int(y/dx) for y in line_y]
    
    # 陸地部分のみ描画
    land_points_x = []
    land_points_y = []
    for x, y, i, j in zip(line_x, line_y, ix, iy):
        if 0 <= i < ncols and 0 <= j < nrows and land_mask[j, i]:
            land_points_x.append(x / dx)
            land_points_y.append(y / dx)
        elif land_points_x:
            ax.plot(land_points_x, land_points_y, 'b-', linewidth=1, alpha=0.8)
            land_points_x = []
            land_points_y = []
    
    if land_points_x:
        ax.plot(land_points_x, land_points_y, 'b-', linewidth=1, alpha=0.8)

ax.imshow(np.ma.masked_where(~sea_mask, sea_mask), 
          cmap='Blues', origin='lower', alpha=0.5)
ax.set_title('Connected (line-based)')
ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows)

# ----- 4. 河川の太さを集水面積で変える -----
ax = axes[1, 0]
ax.imshow(elev_2d, cmap='terrain', origin='lower', vmin=-100, vmax=200)

for line_x, line_y in river_lines:
    ix = [int(x/dx) for x in line_x]
    iy = [int(y/dx) for y in line_y]
    
    # 始点の集水面積で太さを決定
    if ix and iy and 0 <= ix[0] < ncols and 0 <= iy[0] < nrows:
        area = drainage[iy[0], ix[0]]
        # 対数スケールで太さを調整
        width = 0.5 + 2.0 * np.log10(area + 1) / np.log10(drainage.max() + 1)
        width = min(width, 4)
        
        points_x = [x/dx for x, i, j in zip(line_x, ix, iy) 
                    if 0 <= i < ncols and 0 <= j < nrows and land_mask[j, i]]
        points_y = [y/dx for y, i, j in zip(line_y, ix, iy) 
                    if 0 <= i < ncols and 0 <= j < nrows and land_mask[j, i]]
        
        if len(points_x) > 1:
            ax.plot(points_x, points_y, 'b-', linewidth=width, alpha=0.7)

ax.imshow(np.ma.masked_where(~sea_mask, sea_mask), 
          cmap='Blues', origin='lower', alpha=0.5)
ax.set_title('Rivers with variable width')
ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows)

# ----- 5. 3D表示 -----
ax = axes[1, 1]
ax = fig.add_subplot(2, 3, 5, projection='3d')
X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))

ax.plot_surface(X, Y, elev_2d, cmap='terrain', alpha=0.8, 
                linewidth=0, vmin=-100, vmax=200)

# 海面
sea_surface = np.full_like(elev_2d, SEA_LEVEL)
sea_surface[land_mask] = np.nan
ax.plot_surface(X, Y, sea_surface, color='blue', alpha=0.4, linewidth=0)

# 河川を3Dで描画
for line_x, line_y in river_lines[:50]:  # 主要な河川のみ
    ix = [int(x/dx) for x in line_x]
    iy = [int(y/dx) for y in line_y]
    
    zs = []
    xs = []
    ys = []
    for x, y, i, j in zip(line_x, line_y, ix, iy):
        if 0 <= i < ncols and 0 <= j < nrows and land_mask[j, i]:
            xs.append(i)
            ys.append(j)
            zs.append(elev_2d[j, i] + 2)  # 少し浮かせる
    
    if len(xs) > 1:
        ax.plot(xs, ys, zs, 'b-', linewidth=1, alpha=0.8)

ax.set_title('3D with rivers')
ax.view_init(elev=45, azim=45)

# ----- 6. 最終版 -----
ax = axes[1, 2]
ax.imshow(elev_2d, cmap='terrain', origin='lower', vmin=-100, vmax=200)

# 海
ax.imshow(np.ma.masked_where(~sea_mask, np.ones_like(elev_2d)), 
          cmap='Blues', origin='lower', alpha=0.6)

# 河川（太さ可変）
for line_x, line_y in river_lines:
    ix = [int(x/dx) for x in line_x]
    iy = [int(y/dx) for y in line_y]
    
    if ix and iy and 0 <= ix[0] < ncols and 0 <= iy[0] < nrows:
        area = drainage[iy[0], ix[0]]
        width = 0.3 + 2.5 * np.log10(area + 1) / np.log10(drainage.max() + 1)
        
        points_x = [x/dx for x, i, j in zip(line_x, ix, iy) 
                    if 0 <= i < ncols and 0 <= j < nrows and land_mask[j, i]]
        points_y = [y/dx for y, i, j in zip(line_y, ix, iy) 
                    if 0 <= i < ncols and 0 <= j < nrows and land_mask[j, i]]
        
        if len(points_x) > 1:
            ax.plot(points_x, points_y, color='royalblue', 
                    linewidth=width, alpha=0.8, solid_capstyle='round')

# 海岸線
ax.contour(elev_2d, levels=[SEA_LEVEL], colors='darkblue', linewidths=1)

ax.set_title('Final: Terrain + Sea + Rivers')
ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows)

plt.tight_layout()
plt.savefig("rivers_connected.png", dpi=150)
print(f"\nSaved: rivers_connected.png")
plt.show()
