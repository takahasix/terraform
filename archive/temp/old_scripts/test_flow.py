"""
超シンプルなテスト: 一方向の傾斜で水が正しく流れるか確認
"""

import numpy as np
import matplotlib.pyplot as plt
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator

# ========================================
# テスト1: 南から北に高くなる傾斜
# ========================================
nrows, ncols = 50, 50
dx = 10.0

mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)

# シンプルな傾斜: y（行番号）が大きいほど高い
z = mg.add_zeros("topographic__elevation", at="node")

# Landlabのノードインデックスと座標の関係を確認
print("=== Landlab座標系の確認 ===")
print(f"y_of_node[0] (ノード0のy座標): {mg.y_of_node[0]}")
print(f"y_of_node[{mg.number_of_nodes-1}] (最後のノードのy座標): {mg.y_of_node[-1]}")

# y座標に比例した標高（南が低く、北が高い）
z[:] = mg.y_of_node * 0.1  # 10%勾配

print(f"\n標高: ノード0 = {z[0]:.1f}m, ノード末尾 = {z[-1]:.1f}m")

# 境界: 南を開放（水が流れ出る）
mg.set_closed_boundaries_at_grid_edges(
    right_is_closed=True,
    top_is_closed=True,    # 北は閉鎖
    left_is_closed=True,
    bottom_is_closed=False  # 南は開放
)

# 流路計算
fa = FlowAccumulator(mg, flow_director='D8')
fa.run_one_step()

drainage = mg.at_node["drainage_area"]
elevation = z.copy()

# 2Dにreshape
elev_2d = elevation.reshape((nrows, ncols))
drain_2d = drainage.reshape((nrows, ncols))

print(f"\n=== reshape後の確認 ===")
print(f"elev_2d[0, 0] (左下): {elev_2d[0, 0]:.1f}m")
print(f"elev_2d[-1, 0] (左上): {elev_2d[-1, 0]:.1f}m")
print(f"elev_2d[0, -1] (右下): {elev_2d[0, -1]:.1f}m")
print(f"elev_2d[-1, -1] (右上): {elev_2d[-1, -1]:.1f}m")

print(f"\n集水面積:")
print(f"drain_2d[0, :] (下端): max={drain_2d[0, :].max():.1f}")
print(f"drain_2d[-1, :] (上端): max={drain_2d[-1, :].max():.1f}")

if drain_2d[0, :].max() > drain_2d[-1, :].max():
    print("\n✅ 正常: 下端（低い側）に水が集まっている")
else:
    print("\n⚠️ 問題: 上端（高い側）に水が集まっている！")

# ========================================
# 可視化
# ========================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 標高 (origin='lower')
ax = axes[0, 0]
im = ax.imshow(elev_2d, cmap='terrain', origin='lower')
ax.set_title('Elevation (origin=lower)\nbottom=low, top=high')
ax.set_xlabel('x')
ax.set_ylabel('y (row index)')
plt.colorbar(im, ax=ax)

# 標高 (origin='upper' - 通常の画像表示)
ax = axes[0, 1]
im = ax.imshow(elev_2d, cmap='terrain', origin='upper')
ax.set_title('Elevation (origin=upper)\ntop=low, bottom=high')
plt.colorbar(im, ax=ax)

# 集水面積 (origin='lower')
ax = axes[0, 2]
im = ax.imshow(np.log10(drain_2d + 1), cmap='Blues', origin='lower')
ax.set_title('Drainage (origin=lower)\nblue should be at BOTTOM')
plt.colorbar(im, ax=ax)

# 断面図: 左端の列
ax = axes[1, 0]
ax.plot(elev_2d[:, 0], 'b-', label='Elevation')
ax.set_xlabel('Row index (0=bottom in origin=lower)')
ax.set_ylabel('Elevation [m]')
ax.set_title('Elevation profile (left column)')
ax.legend()

# 断面図: 集水面積
ax = axes[1, 1]
ax.plot(drain_2d[:, ncols//2], 'r-', label='Drainage')
ax.set_xlabel('Row index')
ax.set_ylabel('Drainage area')
ax.set_title('Drainage profile (middle column)\nshould be HIGH at row 0')
ax.legend()

# テキスト情報
ax = axes[1, 2]
ax.axis('off')
info = f"""
=== テスト結果 ===

地形: 北（上）が高く、南（下）が低い

期待される動作:
- 水は北→南（上→下）に流れる
- 集水面積は南端（下端、row=0）で最大

実際の結果:
- 下端の最大集水面積: {drain_2d[0, :].max():.0f}
- 上端の最大集水面積: {drain_2d[-1, :].max():.0f}

判定: {'✅ 正常' if drain_2d[0, :].max() > drain_2d[-1, :].max() else '❌ 逆流!'}
"""
ax.text(0.1, 0.5, info, fontsize=12, family='monospace', va='center')

plt.tight_layout()
plt.savefig("flow_direction_test.png", dpi=150)
print("\nSaved: flow_direction_test.png")
plt.show()
