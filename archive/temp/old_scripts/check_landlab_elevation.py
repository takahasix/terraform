"""
Landlabの標高の扱いを確認
- topographic__elevation は標高？深度？
- FlowAccumulatorは高い→低いに流す？低い→高いに流す？
"""

import numpy as np
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator

# ========================================
# テスト: 明確な傾斜で確認
# ========================================
nrows, ncols = 10, 10
mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=10.0)

z = mg.add_zeros("topographic__elevation", at="node")

# パターン1: 値が大きい = 高い（標高）と仮定
# 北(top)を100、南(bottom)を0にする
for node in range(mg.number_of_nodes):
    y = mg.y_of_node[node]
    z[node] = y  # y座標に比例（北が高い）

print("=== テスト: 北が高い（z = y座標）===")
print(f"ノード0 (南西, y=0): 標高 = {z[0]:.1f}")
print(f"ノード{mg.number_of_nodes-1} (北東, y=max): 標高 = {z[-1]:.1f}")

# 境界: 南を開放
mg.set_closed_boundaries_at_grid_edges(True, True, True, False)

# 流路計算
fa = FlowAccumulator(mg, flow_director='D8')
fa.run_one_step()

drainage = mg.at_node["drainage_area"]
flow_receiver = mg.at_node["flow__receiver_node"]

print(f"\n集水面積: 南端(row0) max = {drainage[:ncols].max():.0f}")
print(f"集水面積: 北端(最終row) max = {drainage[-ncols:].max():.0f}")

# 流向を確認
print(f"\n流向確認:")
mid_col = ncols // 2
for row in range(nrows-1, -1, -1):
    node = row * ncols + mid_col
    receiver = flow_receiver[node]
    receiver_y = mg.y_of_node[receiver] if receiver != node else -1
    node_y = mg.y_of_node[node]
    direction = "↓(南)" if receiver_y < node_y else ("↑(北)" if receiver_y > node_y else "停止")
    print(f"  row{row} (y={node_y:.0f}, z={z[node]:.0f}) → {direction}")

if drainage[:ncols].max() > drainage[-ncols:].max():
    print("\n✅ 結論: 水は高い(大きい値)→低い(小さい値)に流れる")
    print("   つまり topographic__elevation は【標高】として扱われている")
else:
    print("\n❌ 結論: 水は低い(小さい値)→高い(大きい値)に流れる")
    print("   つまり topographic__elevation は【深度】として扱われている可能性")
