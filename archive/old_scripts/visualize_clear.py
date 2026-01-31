"""
地形と河川の分かりやすい可視化
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from noise import pnoise2
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator

def generate_ridge_terrain(nrows, ncols, seed=42):
    """リッジノイズで山脈地形を生成"""
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
nrows, ncols = 150, 150
dx = 10.0

mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)

# リッジ地形
terrain = generate_ridge_terrain(nrows, ncols, seed=42) * 500.0

# 盆地効果（端を低く）
center_y, center_x = nrows // 2, ncols // 2
for i in range(nrows):
    for j in range(ncols):
        dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        terrain[i, j] *= 1 - 0.3 * (dist / max_dist)

z = mg.add_zeros("topographic__elevation", at="node")
z += terrain[::-1, :].ravel()  # 上下反転してLandlab座標系に合わせる

mg.set_closed_boundaries_at_grid_edges(True, True, True, False)

# 流路計算
fa = FlowAccumulator(mg, flow_director='D8')
fa.run_one_step()

drainage = mg.at_node["drainage_area"].reshape((nrows, ncols))
elevation = z.reshape((nrows, ncols))

# ========================================
# 分かりやすい可視化
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# ----- 1. 標高マップ（グレースケール）-----
ax = axes[0, 0]
im = ax.imshow(elevation, cmap='gray', origin='lower')
ax.set_title('Elevation (gray = higher)')
plt.colorbar(im, ax=ax, label='Elevation [m]')

# ----- 2. 標高マップ（陰影付き）-----
ax = axes[0, 1]
ls = LightSource(azdeg=315, altdeg=45)
hillshade = ls.hillshade(elevation)
ax.imshow(hillshade, cmap='gray', origin='lower')
ax.set_title('Hillshade (3D relief)')

# ----- 3. 河川だけ -----
ax = axes[1, 0]
# 背景を白に
ax.set_facecolor('white')
# 河川（集水面積が大きい場所）を青い線で表示
river_levels = sorted([
    np.percentile(drainage, 95),   # 中程度の川
    np.percentile(drainage, 97),   # 大きな川
    np.percentile(drainage, 99),   # 最大の川
])
ax.contour(drainage, levels=river_levels, colors=['lightblue', 'blue', 'darkblue'], 
           linewidths=[1, 1.5, 2], origin='lower')
ax.set_title('Rivers only (blue lines)')
ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows)
ax.set_aspect('equal')

# ----- 4. 標高 + 河川オーバーレイ -----
ax = axes[1, 1]
# 陰影図をベースに
ax.imshow(hillshade, cmap='gray', origin='lower')
# 標高で色付け（半透明）
ax.imshow(elevation, cmap='YlOrBr', origin='lower', alpha=0.4)
# 河川を青線で重ねる
ax.contour(drainage, levels=river_levels, colors=['blue', 'blue', 'lightblue'], 
           linewidths=[2.5, 1.5, 0.8], origin='lower')
ax.set_title('Terrain + Rivers (blue lines = rivers)')

plt.tight_layout()
plt.savefig("terrain_rivers_clear.png", dpi=150)
print("Saved: terrain_rivers_clear.png")
plt.show()
