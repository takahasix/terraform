"""
リッジノイズ地形の問題を特定
"""

import numpy as np
import matplotlib.pyplot as plt
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

nrows, ncols = 100, 100
dx = 10.0

# リッジ地形生成
terrain_2d = generate_ridge_terrain(nrows, ncols, seed=42) * 500.0

# ========================================
# 問題の特定: 2D配列の変換方法を確認
# ========================================

mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
z = mg.add_zeros("topographic__elevation", at="node")

print("=== 2D配列からLandlab 1D配列への変換 ===")
print(f"terrain_2d.shape: {terrain_2d.shape}")
print(f"terrain_2d[0, 0] (配列の左上): {terrain_2d[0, 0]:.1f}")
print(f"terrain_2d[-1, 0] (配列の左下): {terrain_2d[-1, 0]:.1f}")

# 方法1: ravel() そのまま (現在のコード)
z_method1 = terrain_2d.ravel()
print(f"\n方法1 ravel(): z[0]={z_method1[0]:.1f}")

# 方法2: 上下反転してからravel
z_method2 = terrain_2d[::-1, :].ravel()
print(f"方法2 [::-1,:].ravel(): z[0]={z_method2[0]:.1f}")

# Landlabのノード0は左下
print(f"\nLandlabノード0の座標: x={mg.x_of_node[0]}, y={mg.y_of_node[0]} (左下)")

# ========================================
# 両方の方法で河川計算してみる
# ========================================
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for method_idx, (method_name, z_data) in enumerate([
    ("ravel() [CURRENT]", terrain_2d.ravel()),
    ("[::-1,:].ravel() [FIX?]", terrain_2d[::-1, :].ravel())
]):
    # グリッド作成
    mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
    z = mg.add_zeros("topographic__elevation", at="node")
    z[:] = z_data
    
    mg.set_closed_boundaries_at_grid_edges(True, True, True, False)
    
    fa = FlowAccumulator(mg, flow_director='D8')
    fa.run_one_step()
    
    elev_2d = z.reshape((nrows, ncols))
    drain_2d = mg.at_node["drainage_area"].reshape((nrows, ncols))
    
    # 標高
    ax = axes[method_idx, 0]
    im = ax.imshow(elev_2d, cmap='terrain', origin='lower')
    ax.set_title(f'{method_name}\nElevation')
    plt.colorbar(im, ax=ax)
    
    # 集水面積
    ax = axes[method_idx, 1]
    im = ax.imshow(np.log10(drain_2d + 1), cmap='Blues', origin='lower')
    ax.set_title('Drainage')
    plt.colorbar(im, ax=ax)
    
    # 元の2D配列との比較
    ax = axes[method_idx, 2]
    ax.imshow(terrain_2d, cmap='terrain', origin='lower')
    ax.set_title('Original terrain_2d\n(origin=lower)')
    
    # 河川と標高の関係
    river_mask = drain_2d > np.percentile(drain_2d, 95)
    river_elev = elev_2d[river_mask].mean()
    non_river_elev = elev_2d[~river_mask].mean()
    
    ax = axes[method_idx, 3]
    ax.axis('off')
    result = "OK" if river_elev < non_river_elev else "WRONG!"
    ax.text(0.1, 0.5, f"""
Method: {method_name}

River avg elev: {river_elev:.1f}m
Non-river avg: {non_river_elev:.1f}m

Result: {result}
""", fontsize=11, va='center', family='monospace')

plt.tight_layout()
plt.savefig("debug_ravel.png", dpi=150)
print("\nSaved: debug_ravel.png")
plt.show()
