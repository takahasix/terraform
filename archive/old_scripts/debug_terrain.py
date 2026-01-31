"""
標高と河川の関係をデバッグ
"""

import numpy as np
import matplotlib.pyplot as plt
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
z += terrain.ravel()

mg.set_closed_boundaries_at_grid_edges(True, True, True, False)

# 流路計算
fa = FlowAccumulator(mg, flow_director='D8')
fa.run_one_step()

drainage = mg.at_node["drainage_area"].reshape((nrows, ncols))
elevation = z.reshape((nrows, ncols))

# ========================================
# デバッグ: 標高と集水面積の関係を確認
# ========================================
print("=== デバッグ情報 ===")
print(f"標高の範囲: {elevation.min():.1f} - {elevation.max():.1f} m")
print(f"集水面積の範囲: {drainage.min():.1f} - {drainage.max():.1f}")

# 河川部分（集水面積上位5%）の標高を確認
river_threshold = np.percentile(drainage, 95)
river_mask = drainage > river_threshold
river_elevations = elevation[river_mask]
non_river_elevations = elevation[~river_mask]

print(f"\n河川部分の標高: 平均 {river_elevations.mean():.1f} m, 範囲 {river_elevations.min():.1f} - {river_elevations.max():.1f} m")
print(f"非河川部分の標高: 平均 {non_river_elevations.mean():.1f} m, 範囲 {non_river_elevations.min():.1f} - {non_river_elevations.max():.1f} m")

if river_elevations.mean() > non_river_elevations.mean():
    print("\n⚠️ 問題発見: 河川が高いところを流れている！")
else:
    print("\n✅ 正常: 河川は低いところを流れている")

# ========================================
# 可視化
# ========================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 標高（数値で確認できるように）
ax = axes[0, 0]
im = ax.imshow(elevation, cmap='viridis', origin='lower')
ax.set_title(f'Elevation\nmin={elevation.min():.0f}, max={elevation.max():.0f}')
plt.colorbar(im, ax=ax)

# 2. 集水面積
ax = axes[0, 1]
im = ax.imshow(np.log10(drainage + 1), cmap='Blues', origin='lower')
ax.set_title('Drainage Area (log10)')
plt.colorbar(im, ax=ax)

# 3. 河川マスク
ax = axes[0, 2]
ax.imshow(river_mask, cmap='binary', origin='lower')
ax.set_title('River mask (white=river)')

# 4. 断面図（中央を横切る）
ax = axes[1, 0]
mid_row = nrows // 2
ax.plot(elevation[mid_row, :], 'b-', label='Elevation')
ax2 = ax.twinx()
ax2.plot(np.log10(drainage[mid_row, :] + 1), 'r-', alpha=0.7, label='Drainage (log)')
ax.set_xlabel('X position')
ax.set_ylabel('Elevation [m]', color='b')
ax2.set_ylabel('Drainage (log10)', color='r')
ax.set_title('Cross-section (middle row)')

# 5. 標高vs集水面積の散布図
ax = axes[1, 1]
sample_idx = np.random.choice(len(elevation.ravel()), 5000)
ax.scatter(elevation.ravel()[sample_idx], np.log10(drainage.ravel()[sample_idx] + 1), 
           alpha=0.3, s=1)
ax.set_xlabel('Elevation [m]')
ax.set_ylabel('Drainage Area (log10)')
ax.set_title('Elevation vs Drainage\n(should be negative correlation)')

# 6. 3D表示風
ax = axes[1, 2]
from matplotlib.colors import LightSource
ls = LightSource(azdeg=315, altdeg=45)
rgb = ls.shade(elevation, cmap=plt.cm.terrain, vert_exag=2)
ax.imshow(rgb, origin='lower')
# 河川を重ねる
river_display = np.ma.masked_where(~river_mask, np.ones_like(elevation))
ax.imshow(river_display, cmap='Blues', origin='lower', alpha=0.8)
ax.set_title('3D terrain + Rivers')

plt.tight_layout()
plt.savefig("debug_terrain.png", dpi=150)
print(f"\nSaved: debug_terrain.png")
plt.show()
