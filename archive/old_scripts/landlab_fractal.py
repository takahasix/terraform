"""
Landlab 地形進化シミュレーション - フラクタル初期地形版
パーリンノイズで自然な初期地形を生成し、河川侵食をシミュレート
"""

import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2
from landlab import RasterModelGrid, imshow_grid
from landlab.components import (
    FlowAccumulator,
    LinearDiffuser,
    StreamPowerEroder,
)


def generate_fractal_terrain(nrows, ncols, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=42):
    """
    パーリンノイズを使ったフラクタル地形生成
    
    Parameters:
    -----------
    scale : float
        ノイズのスケール（大きいほど滑らか）
    octaves : int
        重ね合わせる周波数の数（多いほど細かいディテール）
    persistence : float
        高周波成分の振幅減衰率（0.5が自然）
    lacunarity : float
        周波数の増加率（2.0が自然）
    """
    terrain = np.zeros((nrows, ncols))
    
    for i in range(nrows):
        for j in range(ncols):
            terrain[i, j] = pnoise2(
                i / scale,
                j / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=seed
            )
    
    # 0-1に正規化してから標高にスケーリング
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    return terrain


def generate_ridge_terrain(nrows, ncols, seed=42):
    """
    尾根と谷のある地形を生成（リッジノイズ）
    絶対値を取ることで尾根状の地形になる
    """
    np.random.seed(seed)
    terrain = np.zeros((nrows, ncols))
    
    # 複数のオクターブを重ね合わせ
    for octave in range(6):
        freq = 2 ** octave
        amp = 0.5 ** octave
        
        for i in range(nrows):
            for j in range(ncols):
                # リッジノイズ: 1 - |noise| で尾根を作る
                val = pnoise2(
                    i * freq / 150.0,
                    j * freq / 150.0,
                    base=seed + octave
                )
                terrain[i, j] += (1 - abs(val)) * amp
    
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    return terrain


# ========================================
# 1. グリッドの作成
# ========================================
nrows, ncols = 150, 150
dx = 10.0  # セルサイズ [m]

mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
print(f"Grid: {nrows}x{ncols} cells, resolution {dx}m")

# ========================================
# 2. フラクタル初期地形の生成
# ========================================
print("\n=== Generating fractal terrain ===")

# 方法1: 通常のパーリンノイズ
perlin_terrain = generate_fractal_terrain(nrows, ncols, scale=80, octaves=6, seed=42)

# 方法2: リッジノイズ（山脈っぽい）
ridge_terrain = generate_ridge_terrain(nrows, ncols, seed=42)

# 今回はリッジ地形を使用（山脈っぽくなる）
# 標高スケール: 0-500m
initial_elevation = ridge_terrain * 500.0

# 排水のため、端に向かって少し下げる（盆地を作る）
center_y, center_x = nrows // 2, ncols // 2
for i in range(nrows):
    for j in range(ncols):
        dist_from_center = np.sqrt((i - center_y)**2 + (j - center_x)**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        # 端に近いほど低くなる
        edge_factor = 1 - 0.3 * (dist_from_center / max_dist)
        initial_elevation[i, j] *= edge_factor

# グリッドに標高を設定
z = mg.add_zeros("topographic__elevation", at="node")
z += initial_elevation.ravel()

# 境界条件：下端を開放（川が流れ出る）
mg.set_closed_boundaries_at_grid_edges(
    right_is_closed=True,
    top_is_closed=True,
    left_is_closed=True,
    bottom_is_closed=False,  # 南側のみ開放
)

print(f"Elevation range: {z.min():.1f} - {z.max():.1f} m")

# ========================================
# 3. コンポーネントの初期化
# ========================================
K_sp = 5.0e-5   # 河川侵食係数
m_sp = 0.5
n_sp = 1.0
K_hs = 0.05     # 斜面拡散係数

fa = FlowAccumulator(mg, flow_director='D8')
sp = StreamPowerEroder(mg, K_sp=K_sp, m_sp=m_sp, n_sp=n_sp)
ld = LinearDiffuser(mg, linear_diffusivity=K_hs)

# ========================================
# 4. 比較用プロット準備
# ========================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 初期地形
ax = axes[0, 0]
im = ax.imshow(z.reshape((nrows, ncols)), cmap='terrain', origin='lower')
ax.set_title('Initial Terrain (Fractal)')
plt.colorbar(im, ax=ax, label='Elevation [m]')

# パーリンノイズ参考表示
ax = axes[0, 1]
im = ax.imshow(perlin_terrain * 500, cmap='terrain', origin='lower')
ax.set_title('Perlin Noise (Reference)')
plt.colorbar(im, ax=ax, label='Elevation [m]')

# リッジノイズ参考表示
ax = axes[0, 2]
im = ax.imshow(ridge_terrain * 500, cmap='terrain', origin='lower')
ax.set_title('Ridge Noise (Reference)')
plt.colorbar(im, ax=ax, label='Elevation [m]')

# ========================================
# 5. シミュレーション実行
# ========================================
dt = 1000       # タイムステップ [yr]
tmax = 50000    # 総時間 [yr]（5万年）
total_time = 0

print(f"\n=== Running simulation: dt={dt}yr, tmax={tmax}yr ===")

while total_time < tmax:
    ld.run_one_step(dt)     # 斜面拡散
    fa.run_one_step()       # 流路計算
    sp.run_one_step(dt)     # 河川侵食
    
    total_time += dt
    
    if total_time % 50000 == 0:
        print(f"  {total_time:,} years...")

print(f"Simulation complete: {total_time:,} years")
print(f"Final elevation range: {z.min():.1f} - {z.max():.1f} m")

# ========================================
# 6. 結果の表示
# ========================================
# 最終地形
ax = axes[1, 0]
im = ax.imshow(z.reshape((nrows, ncols)), cmap='terrain', origin='lower')
ax.set_title(f'Final Terrain ({total_time:,} years)')
plt.colorbar(im, ax=ax, label='Elevation [m]')

# 集水面積（河川ネットワーク）
ax = axes[1, 1]
drainage = mg.at_node["drainage_area"].reshape((nrows, ncols))
im = ax.imshow(np.log10(drainage + 1), cmap='Blues', origin='lower')
ax.set_title('Drainage Area (log10)')
plt.colorbar(im, ax=ax, label='log10(area)')

# 河川ネットワークのオーバーレイ
ax = axes[1, 2]
ax.imshow(z.reshape((nrows, ncols)), cmap='terrain', origin='lower', alpha=0.7)
# 大きな集水面積（河川）のみ表示
river_threshold = np.percentile(drainage, 95)
rivers = drainage > river_threshold
ax.imshow(np.ma.masked_where(~rivers, rivers), cmap='Blues', origin='lower', alpha=0.8)
ax.set_title('Terrain + Rivers')

plt.tight_layout()
plt.savefig("landlab_fractal_result.png", dpi=150)
print(f"\nResult saved to landlab_fractal_result.png")
plt.show()
