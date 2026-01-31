"""
自然な地形生成 - 谷が細く、尾根が広い
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from noise import pnoise2
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator

def generate_mountain_terrain(nrows, ncols, seed=42):
    """
    自然な山岳地形を生成
    - 通常のパーリンノイズベース（尾根が広い）
    - 谷を細くするため、低い部分だけを強調
    """
    terrain = np.zeros((nrows, ncols))
    
    # 複数スケールのパーリンノイズを重ねる
    for octave in range(6):
        freq = 2 ** octave
        amp = 0.5 ** octave
        for i in range(nrows):
            for j in range(ncols):
                val = pnoise2(
                    i * freq / 100.0,
                    j * freq / 100.0,
                    base=seed + octave
                )
                terrain[i, j] += val * amp
    
    # 正規化
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    
    # 谷を深くする（低い部分を二乗で強調）
    # これで「広い山頂・尾根」と「細い谷」ができる
    terrain = terrain ** 0.7  # べき乗で調整（0.5-1.0）
    
    return terrain

# ========================================
# 地形生成
# ========================================
nrows, ncols = 150, 150
dx = 10.0

mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)

# 山岳地形（谷が細い）
terrain = generate_mountain_terrain(nrows, ncols, seed=42) * 500.0

# 排水のため、南端に向かって下げる（谷が南に流れる）
for i in range(nrows):
    # 南に行くほど低くなる傾斜を追加
    slope_factor = i / nrows * 50  # 0-50mの傾斜
    terrain[i, :] += slope_factor

z = mg.add_zeros("topographic__elevation", at="node")
z += terrain.ravel()

# 境界条件：南側のみ開放
mg.set_closed_boundaries_at_grid_edges(True, True, True, False)

# 流路計算
fa = FlowAccumulator(mg, flow_director='D8')
fa.run_one_step()

drainage = mg.at_node["drainage_area"].reshape((nrows, ncols))
elevation = z.reshape((nrows, ncols))

# ========================================
# 確認
# ========================================
river_threshold = np.percentile(drainage, 95)
river_mask = drainage > river_threshold
river_elevations = elevation[river_mask]
non_river_elevations = elevation[~river_mask]

print("=== 地形情報 ===")
print(f"標高範囲: {elevation.min():.1f} - {elevation.max():.1f} m")
print(f"河川部分の平均標高: {river_elevations.mean():.1f} m")
print(f"非河川部分の平均標高: {non_river_elevations.mean():.1f} m")

# ========================================
# 可視化
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# 1. 陰影図
ax = axes[0, 0]
ls = LightSource(azdeg=315, altdeg=45)
hillshade = ls.hillshade(elevation)
ax.imshow(hillshade, cmap='gray', origin='lower')
ax.set_title('Hillshade (white=high, dark=valley)')

# 2. 標高カラー
ax = axes[0, 1]
im = ax.imshow(elevation, cmap='YlOrBr', origin='lower')
ax.set_title('Elevation (yellow=low, brown=high)')
plt.colorbar(im, ax=ax, label='Elevation [m]')

# 3. 河川のみ
ax = axes[1, 0]
ax.set_facecolor('lightgray')
# 河川レベル
levels = sorted([np.percentile(drainage, p) for p in [90, 95, 98]])
cs = ax.contour(drainage, levels=levels, colors=['lightblue', 'blue', 'darkblue'],
                linewidths=[0.5, 1, 2], origin='lower')
ax.set_title('Rivers only')
ax.set_aspect('equal')

# 4. 地形 + 河川
ax = axes[1, 1]
# 陰影 + カラー
rgb = ls.shade(elevation, cmap=plt.cm.terrain, vert_exag=3, blend_mode='overlay')
ax.imshow(rgb, origin='lower')
# 河川を重ねる
ax.contour(drainage, levels=levels, colors=['cyan', 'blue', 'darkblue'],
           linewidths=[0.5, 1.5, 2.5], origin='lower')
ax.set_title('Terrain + Rivers\n(rivers should flow in valleys)')

plt.tight_layout()
plt.savefig("natural_terrain.png", dpi=150)
print(f"\nSaved: natural_terrain.png")
plt.show()
