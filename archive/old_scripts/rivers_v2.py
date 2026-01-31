"""
海面レベル + 連結された河川ネットワーク（修正版）
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
flow_receiver = mg.at_node["flow__receiver_node"]

sea_mask = elev_2d < SEA_LEVEL
land_mask = ~sea_mask

# ========================================
# 河川トレース（シンプル版）
# ========================================
def trace_river_from_node(start_node, flow_receiver, num_nodes):
    """1点から下流へトレース"""
    path = [start_node]
    node = start_node
    visited = set()
    
    while node not in visited:
        visited.add(node)
        receiver = flow_receiver[node]
        if receiver == node or receiver < 0 or receiver >= num_nodes:
            break
        path.append(receiver)
        node = receiver
    
    return path

# 河川の始点を選定（集水面積が大きいセル）
drainage_flat = drainage.ravel()
threshold = np.percentile(drainage_flat[drainage_flat > 0], 85)
river_sources = np.where(drainage_flat > threshold)[0]

print(f"River source nodes: {len(river_sources)}")

# 各始点からトレース
all_river_nodes = set()
river_paths = []

# 集水面積順にソートして、大きい川から処理
sorted_sources = sorted(river_sources, key=lambda n: drainage_flat[n], reverse=True)

for source in sorted_sources[:200]:  # 上位200本
    path = trace_river_from_node(source, flow_receiver, mg.number_of_nodes)
    
    # 新しいノードが含まれている場合のみ追加
    new_nodes = [n for n in path if n not in all_river_nodes]
    if len(new_nodes) > 2:
        river_paths.append(path)
        all_river_nodes.update(path)

print(f"River paths traced: {len(river_paths)}")

# ========================================
# 可視化
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

# ----- 1. 元の細切れ -----
ax = axes[0, 0]
ax.imshow(elev_2d, cmap='terrain', origin='lower', vmin=-100, vmax=200)

# 単純閾値
simple_river = (drainage > threshold) & land_mask
ax.imshow(np.ma.masked_where(~simple_river, np.ones_like(elev_2d)), 
          cmap='Blues', origin='lower')
ax.imshow(np.ma.masked_where(~sea_mask, np.ones_like(elev_2d)), 
          cmap='Blues', origin='lower', alpha=0.5)
ax.set_title('Fragmented (threshold only)')

# ----- 2. 連結版（マスク） -----
ax = axes[0, 1]
ax.imshow(elev_2d, cmap='terrain', origin='lower', vmin=-100, vmax=200)

# 河川マスク
river_mask_2d = np.zeros((nrows, ncols), dtype=bool)
for node in all_river_nodes:
    row = node // ncols
    col = node % ncols
    if 0 <= row < nrows and 0 <= col < ncols:
        river_mask_2d[row, col] = True

river_mask_land = river_mask_2d & land_mask
ax.imshow(np.ma.masked_where(~river_mask_land, np.ones_like(elev_2d)), 
          cmap='Blues', origin='lower')
ax.imshow(np.ma.masked_where(~sea_mask, np.ones_like(elev_2d)), 
          cmap='Blues', origin='lower', alpha=0.5)
ax.set_title('Connected (traced)')

# ----- 3. ライン描画（太さ可変） -----
ax = axes[1, 0]
ax.imshow(elev_2d, cmap='terrain', origin='lower', vmin=-100, vmax=200)

# 海
ax.imshow(np.ma.masked_where(~sea_mask, np.ones_like(elev_2d)), 
          cmap='Blues', origin='lower', alpha=0.5)

# 河川をラインで描画
for path in river_paths:
    xs = []
    ys = []
    
    for node in path:
        row = node // ncols
        col = node % ncols
        
        # 陸地のみ
        if 0 <= row < nrows and 0 <= col < ncols and land_mask[row, col]:
            xs.append(col)
            ys.append(row)
        elif xs:  # 海に入ったら一旦描画
            if len(xs) > 1:
                # 始点の集水面積で太さ決定
                start_drainage = drainage_flat[path[0]]
                width = 0.5 + 2.5 * np.log10(start_drainage + 1) / np.log10(drainage_flat.max() + 1)
                ax.plot(xs, ys, 'b-', linewidth=width, alpha=0.8, solid_capstyle='round')
            xs = []
            ys = []
    
    # 残り
    if len(xs) > 1:
        start_drainage = drainage_flat[path[0]]
        width = 0.5 + 2.5 * np.log10(start_drainage + 1) / np.log10(drainage_flat.max() + 1)
        ax.plot(xs, ys, 'b-', linewidth=width, alpha=0.8, solid_capstyle='round')

ax.contour(elev_2d, levels=[SEA_LEVEL], colors='navy', linewidths=1)
ax.set_title('Rivers as lines (width = drainage area)')
ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows)

# ----- 4. 最終版 -----
ax = axes[1, 1]
# 陰影
from matplotlib.colors import LightSource
ls = LightSource(azdeg=315, altdeg=45)
rgb = ls.shade(elev_2d, cmap=plt.cm.terrain, vert_exag=2, 
               blend_mode='soft', vmin=-100, vmax=200)
ax.imshow(rgb, origin='lower')

# 海（半透明）
ax.imshow(np.ma.masked_where(~sea_mask, np.ones_like(elev_2d)), 
          cmap='Blues', origin='lower', alpha=0.6)

# 河川
for path in river_paths:
    xs = []
    ys = []
    
    for node in path:
        row = node // ncols
        col = node % ncols
        
        if 0 <= row < nrows and 0 <= col < ncols and land_mask[row, col]:
            xs.append(col)
            ys.append(row)
        elif xs:
            if len(xs) > 1:
                start_drainage = drainage_flat[path[0]]
                width = 0.3 + 3.0 * np.log10(start_drainage + 1) / np.log10(drainage_flat.max() + 1)
                ax.plot(xs, ys, color='royalblue', linewidth=width, 
                        alpha=0.9, solid_capstyle='round')
            xs = []
            ys = []
    
    if len(xs) > 1:
        start_drainage = drainage_flat[path[0]]
        width = 0.3 + 3.0 * np.log10(start_drainage + 1) / np.log10(drainage_flat.max() + 1)
        ax.plot(xs, ys, color='royalblue', linewidth=width, 
                alpha=0.9, solid_capstyle='round')

ax.contour(elev_2d, levels=[SEA_LEVEL], colors='darkblue', linewidths=1.5)
ax.set_title('Final: Shaded terrain + Sea + Rivers')
ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows)

plt.tight_layout()
plt.savefig("rivers_connected_v2.png", dpi=150)
print(f"\nSaved: rivers_connected_v2.png")
plt.show()
