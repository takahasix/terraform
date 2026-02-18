"""
リッジノイズの地形構造を確認
問題: リッジノイズは「尾根が高い」のか「尾根が低い」のか？
"""

import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2

def generate_ridge_terrain(nrows, ncols, seed=42):
    terrain = np.zeros((nrows, ncols))
    for octave in range(6):
        freq = 2 ** octave
        amp = 0.5 ** octave
        for i in range(nrows):
            for j in range(ncols):
                val = pnoise2(i * freq / 150.0, j * freq / 150.0, base=seed + octave)
                # リッジノイズの計算
                terrain[i, j] += (1 - abs(val)) * amp
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    return terrain

# 生成
nrows, ncols = 100, 100
ridge = generate_ridge_terrain(nrows, ncols)

print("=== リッジノイズの構造 ===")
print(f"値の範囲: {ridge.min():.3f} - {ridge.max():.3f}")

# リッジノイズの計算式を分析
print("\n計算式: terrain += (1 - abs(pnoise2)) * amp")
print("  - pnoise2の出力: -1 ~ +1")
print("  - abs(pnoise2): 0 ~ 1")
print("  - 1 - abs(): 1 ~ 0")
print("  → pnoise2が0付近のとき最大値(1)")
print("  → pnoise2が±1付近のとき最小値(0)")

# パーリンノイズの「0」は等高線的には「中間」
# つまりリッジノイズは「パーリンノイズの0付近」が山頂になる
print("\n結論: リッジノイズの【高い部分】が尾根/山頂")

# 可視化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 通常のパーリンノイズ
perlin = np.zeros((nrows, ncols))
for i in range(nrows):
    for j in range(ncols):
        perlin[i, j] = pnoise2(i / 50.0, j / 50.0, base=42)

ax = axes[0]
im = ax.imshow(perlin, cmap='RdBu', origin='lower')
ax.set_title('Perlin noise\nred=positive, blue=negative')
plt.colorbar(im, ax=ax)

# リッジノイズ
ax = axes[1]
im = ax.imshow(ridge, cmap='terrain', origin='lower')
ax.set_title('Ridge noise\nhigh values = ridges')
plt.colorbar(im, ax=ax)

# 断面図
ax = axes[2]
mid = nrows // 2
ax.plot(perlin[mid, :], 'b-', label='Perlin', alpha=0.7)
ax.plot(ridge[mid, :], 'r-', label='Ridge (1-|perlin|)', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.legend()
ax.set_title('Cross section (middle row)')
ax.set_xlabel('x')
ax.set_ylabel('value')

plt.tight_layout()
plt.savefig("ridge_analysis.png", dpi=150)
print("\nSaved: ridge_analysis.png")
plt.show()
