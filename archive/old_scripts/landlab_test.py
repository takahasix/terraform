"""
Landlab 地形進化シミュレーション テスト
ランダムな初期地形から河川侵食・斜面拡散をシミュレート
"""

import numpy as np
import matplotlib.pyplot as plt
from landlab import RasterModelGrid, imshow_grid
from landlab.components import (
    FlowAccumulator,
    LinearDiffuser,
    StreamPowerEroder,
)

# ========================================
# 1. グリッドの作成（100x100セル、各10m）
# ========================================
nrows, ncols = 100, 100
dx = 10.0  # セルサイズ [m]

mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
print(f"グリッド作成: {nrows}x{ncols} セル、解像度 {dx}m")

# ========================================
# 2. 初期地形の生成（ランダム + 傾斜）
# ========================================
np.random.seed(42)

# ベースとなる傾斜（南から北へ高くなる）
z = mg.add_zeros("topographic__elevation", at="node")
z += mg.y_of_node * 0.01  # 1%の傾斜

# ランダムなノイズを追加
z += np.random.rand(mg.number_of_nodes) * 5.0

# 境界条件：端は固定（開境界）
mg.set_closed_boundaries_at_grid_edges(
    right_is_closed=False,
    top_is_closed=True,
    left_is_closed=False,
    bottom_is_closed=False,
)

print("初期地形生成完了")

# ========================================
# 3. プロセスコンポーネントの初期化
# ========================================
# パラメータ設定
K_sp = 1.0e-4   # 河川侵食係数（大きめに設定して効果を見やすく）
m_sp = 0.5      # 集水面積の指数
n_sp = 1.0      # 勾配の指数
K_hs = 0.1      # 斜面拡散係数 [m²/yr]
uplift_rate = 0.001  # 隆起速度 [m/yr]

# コンポーネント初期化
fa = FlowAccumulator(mg)
sp = StreamPowerEroder(mg, K_sp=K_sp, m_sp=m_sp, n_sp=n_sp)
ld = LinearDiffuser(mg, linear_diffusivity=K_hs)

print("コンポーネント初期化完了")

# ========================================
# 4. 初期地形の表示
# ========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

plt.sca(axes[0])
imshow_grid(mg, "topographic__elevation", cmap="terrain")
axes[0].set_title("初期地形")

# ========================================
# 5. シミュレーション実行
# ========================================
dt = 500        # タイムステップ [yr]
total_time = 0
tmax = 50000    # 総時間 [yr]（5万年）

print(f"\nシミュレーション開始: dt={dt}yr, tmax={tmax}yr")

while total_time < tmax:
    # 地殻隆起
    z[mg.core_nodes] += uplift_rate * dt
    
    # 斜面拡散
    ld.run_one_step(dt)
    
    # 流路計算
    fa.run_one_step()
    
    # 河川侵食
    sp.run_one_step(dt)
    
    total_time += dt
    
    if total_time % 10000 == 0:
        print(f"  {total_time}年経過...")

print(f"シミュレーション完了: {total_time}年")

# ========================================
# 6. 結果の表示
# ========================================
# 最終地形
plt.sca(axes[1])
imshow_grid(mg, "topographic__elevation", cmap="terrain")
axes[1].set_title(f"最終地形 ({total_time}年後)")

# 集水面積（河川ネットワーク）
plt.sca(axes[2])
drainage_area = mg.at_node["drainage_area"]
imshow_grid(mg, np.log10(drainage_area + 1), cmap="Blues")
axes[2].set_title("集水面積 (log10)")

plt.tight_layout()
plt.savefig("landlab_result.png", dpi=150)
print("\n結果を landlab_result.png に保存しました")
plt.show()
