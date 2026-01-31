"""
地形進化シミュレーション - Hatari Labs チュートリアル参考版
Basin Scale での地形進化（河川侵食・斜面拡散・地殻隆起）

参考: https://hatarilabs.com/ih-en/modeling-land-evolution-at-basin-scale-with-python-and-landlab-tutorial

パラメータ比較（Hatari Labs vs 現行版）:
------------------------------------------
| パラメータ | Hatari Labs | 現行版 (landlab_test.py) | 推奨値 |
|-----------|-------------|------------------------|--------|
| K_sp      | 1.0e-5      | 1.0e-4                 | 1.0e-5 |
| m_sp      | 0.5         | 0.5                    | 0.5    |
| n_sp      | 1.0         | 1.0                    | 1.0    |
| K_hs      | 0.05        | 0.1                    | 0.05   |
| uplift    | 0.001       | 0.001                  | 0.001  |
| dt        | 1000        | 500                    | 1000   |
| tmax      | 100000      | 50000                  | 100000 |
------------------------------------------

Stream Power Law (河川侵食):
    E = K_sp * A^m_sp * S^n_sp
    - K_sp: 侵食係数（岩石の種類・気候に依存）
    - A: 集水面積
    - m_sp: 集水面積の指数（通常 0.3-0.6）
    - n_sp: 勾配の指数（通常 0.7-1.0）

Linear Diffusion (斜面拡散):
    ∂z/∂t = K_hs * ∇²z
    - K_hs: 拡散係数 [m²/yr]（土壌クリープ、風化など）

Uplift (地殻隆起):
    ∂z/∂t = U
    - U: 隆起速度 [m/yr]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from noise import pnoise2
from landlab import RasterModelGrid, imshow_grid
from landlab.components import (
    FlowAccumulator,
    LinearDiffuser,
    StreamPowerEroder,
    DepressionFinderAndRouter,  # ピット（窪地）対策
)


# ========================================
# パラメータ設定（Hatari Labs準拠）
# ========================================
# グリッド設定
NROWS = 100      # 行数
NCOLS = 100      # 列数
DX = 50.0        # セルサイズ [m]（Hatari Labs: 90m）

# 地形進化パラメータ
UPLIFT_RATE = 0.001   # 隆起速度 [m/yr] - 1mm/年は活発な隆起域
K_SP = 1.0e-5         # 河川侵食係数（Hatari Labs推奨）
M_SP = 0.5            # 集水面積の指数
N_SP = 1.0            # 勾配の指数
K_HS = 0.05           # 斜面拡散係数 [m²/yr]（Hatari Labs推奨）

# 時間設定
DT = 1000             # タイムステップ [yr]
TMAX = 100000         # 総シミュレーション時間 [yr]（10万年）

# 初期地形
INITIAL_AMPLITUDE = 200.0  # 初期地形の振幅 [m]
BASE_SLOPE = 0.005         # 基本傾斜（排水のため）


def generate_perlin_terrain(nrows, ncols, seed=42, amplitude=200.0):
    """
    パーリンノイズを使った初期地形生成
    
    複数オクターブを重ねて自然な起伏を作る
    """
    terrain = np.zeros((nrows, ncols))
    
    for octave in range(5):
        freq = 2 ** octave
        amp = 0.5 ** octave
        for i in range(nrows):
            for j in range(ncols):
                val = pnoise2(
                    i * freq / 80.0,
                    j * freq / 80.0,
                    base=seed + octave
                )
                terrain[i, j] += val * amp
    
    # 正規化して振幅を適用
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    terrain *= amplitude
    
    return terrain


def run_land_evolution_simulation(
    nrows=NROWS,
    ncols=NCOLS,
    dx=DX,
    uplift_rate=UPLIFT_RATE,
    k_sp=K_SP,
    m_sp=M_SP,
    n_sp=N_SP,
    k_hs=K_HS,
    dt=DT,
    tmax=TMAX,
    seed=42,
    use_depression_finder=True,
    save_interval=10000,  # 中間結果の保存間隔 [yr]
):
    """
    地形進化シミュレーションを実行
    
    Parameters:
    -----------
    use_depression_finder: bool
        窪地（ピット）を検出して流路を通すか
        True: よりリアルだが計算が遅い
        False: 高速だが窪地に水が溜まる可能性
    """
    print("=" * 60)
    print("地形進化シミュレーション")
    print("=" * 60)
    print(f"グリッド: {nrows} x {ncols} セル, 解像度 {dx} m")
    print(f"隆起速度: {uplift_rate} m/yr ({uplift_rate*1000:.1f} mm/yr)")
    print(f"河川侵食係数 K_sp: {k_sp}")
    print(f"斜面拡散係数 K_hs: {k_hs} m²/yr")
    print(f"シミュレーション時間: {tmax/1000:.0f} 千年, dt={dt} yr")
    print(f"窪地処理: {'有効' if use_depression_finder else '無効'}")
    print("=" * 60)
    
    # ----- グリッド作成 -----
    mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
    
    # ----- 初期地形 -----
    terrain_2d = generate_perlin_terrain(nrows, ncols, seed=seed, amplitude=INITIAL_AMPLITUDE)
    
    # 排水のための傾斜を追加（南端が低い）
    for i in range(nrows):
        terrain_2d[i, :] += i * dx * BASE_SLOPE
    
    z = mg.add_zeros("topographic__elevation", at="node")
    z += terrain_2d.ravel()
    
    # 初期標高を保存
    z_initial = z.copy()
    
    # ----- 境界条件 -----
    # 南側のみ開放（水が流れ出る）
    mg.set_closed_boundaries_at_grid_edges(
        right_is_closed=True,
        top_is_closed=True,
        left_is_closed=True,
        bottom_is_closed=False,
    )
    
    # ----- コンポーネント初期化 -----
    fa = FlowAccumulator(mg, flow_director='D8')
    sp = StreamPowerEroder(mg, K_sp=k_sp, m_sp=m_sp, n_sp=n_sp, threshold_sp=0.0)
    ld = LinearDiffuser(mg, linear_diffusivity=k_hs, deposit=False)
    
    if use_depression_finder:
        df = DepressionFinderAndRouter(mg)
    
    # ----- 中間結果保存用 -----
    snapshots = []
    
    # ----- シミュレーションループ -----
    total_time = 0
    t_values = np.arange(0, tmax, dt)
    
    print(f"\nシミュレーション開始...")
    
    for ti in t_values:
        # 1. 地殻隆起
        z[mg.core_nodes] += uplift_rate * dt
        
        # 2. 斜面拡散（土壌クリープ）
        ld.run_one_step(dt)
        
        # 3. 流路計算
        fa.run_one_step()
        
        # 4. 窪地処理（オプション）
        if use_depression_finder:
            df.map_depressions()
        
        # 5. 河川侵食
        sp.run_one_step(dt)
        
        total_time += dt
        
        # 進捗表示
        if total_time % (tmax // 10) == 0:
            print(f"  {total_time/1000:.0f} 千年経過...")
        
        # スナップショット保存
        if total_time % save_interval == 0:
            snapshots.append({
                'time': total_time,
                'elevation': z.copy().reshape((nrows, ncols)),
                'drainage': mg.at_node["drainage_area"].copy().reshape((nrows, ncols)),
            })
    
    print(f"\nシミュレーション完了: {total_time/1000:.0f} 千年")
    
    # ----- 最終結果 -----
    z_final = z.reshape((nrows, ncols))
    drainage = mg.at_node["drainage_area"].reshape((nrows, ncols))
    
    # 統計
    z_change = z_final - z_initial.reshape((nrows, ncols))
    print(f"\n=== 結果サマリー ===")
    print(f"初期標高: {z_initial.min():.1f} - {z_initial.max():.1f} m")
    print(f"最終標高: {z_final.min():.1f} - {z_final.max():.1f} m")
    print(f"標高変化: {z_change.min():.1f} - {z_change.max():.1f} m")
    print(f"総隆起量: {uplift_rate * total_time:.1f} m")
    
    return {
        'grid': mg,
        'z_initial': z_initial.reshape((nrows, ncols)),
        'z_final': z_final,
        'drainage': drainage,
        'z_change': z_change,
        'snapshots': snapshots,
        'params': {
            'nrows': nrows, 'ncols': ncols, 'dx': dx,
            'uplift_rate': uplift_rate, 'k_sp': k_sp, 'k_hs': k_hs,
            'tmax': tmax, 'dt': dt,
        }
    }


def visualize_results(results, output_file="land_evolution_result.png"):
    """
    シミュレーション結果の可視化
    """
    z_initial = results['z_initial']
    z_final = results['z_final']
    drainage = results['drainage']
    z_change = results['z_change']
    params = results['params']
    
    nrows, ncols = z_final.shape
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ls = LightSource(azdeg=315, altdeg=45)
    
    # ----- 1. 初期地形 -----
    ax = axes[0, 0]
    rgb = ls.shade(z_initial, cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    ax.set_title('初期地形')
    
    # ----- 2. 最終地形 -----
    ax = axes[0, 1]
    rgb = ls.shade(z_final, cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    ax.set_title(f'最終地形 ({params["tmax"]/1000:.0f} 千年後)')
    
    # ----- 3. 標高変化 -----
    ax = axes[0, 2]
    vmax = max(abs(z_change.min()), abs(z_change.max()))
    im = ax.imshow(z_change, cmap='RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    ax.set_title('標高変化\n(青=侵食, 赤=堆積/隆起)')
    plt.colorbar(im, ax=ax, label='変化量 [m]')
    
    # ----- 4. 集水面積（河川） -----
    ax = axes[1, 0]
    log_drainage = np.log10(drainage + 1)
    im = ax.imshow(log_drainage, cmap='Blues', origin='lower')
    ax.set_title('集水面積 (log10)\n→ 河川ネットワーク')
    plt.colorbar(im, ax=ax, label='log10(面積)')
    
    # ----- 5. 地形 + 河川 -----
    ax = axes[1, 1]
    rgb = ls.shade(z_final, cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    
    # 河川を重ねる
    river_threshold = np.percentile(drainage, 92)
    levels = [np.percentile(drainage, p) for p in [92, 96, 98]]
    ax.contour(drainage, levels=levels, colors=['cyan', 'blue', 'darkblue'],
               linewidths=[0.5, 1, 2], origin='lower')
    ax.set_title('最終地形 + 河川')
    
    # ----- 6. パラメータ情報 -----
    ax = axes[1, 2]
    ax.axis('off')
    param_text = f"""
    === シミュレーションパラメータ ===
    
    グリッド: {params['nrows']} x {params['ncols']} セル
    解像度: {params['dx']} m
    
    隆起速度: {params['uplift_rate']} m/yr
            = {params['uplift_rate']*1000:.1f} mm/yr
    
    河川侵食係数 K_sp: {params['k_sp']:.1e}
    斜面拡散係数 K_hs: {params['k_hs']} m²/yr
    
    シミュレーション時間: {params['tmax']/1000:.0f} 千年
    タイムステップ: {params['dt']} yr
    
    === 結果 ===
    最終標高範囲: {z_final.min():.1f} - {z_final.max():.1f} m
    総隆起量: {params['uplift_rate'] * params['tmax']:.1f} m
    最大侵食量: {abs(z_change.min()):.1f} m
    """
    ax.text(0.1, 0.9, param_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\n結果を {output_file} に保存しました")
    
    return fig


def visualize_time_evolution(snapshots, output_file="land_evolution_timelapse.png"):
    """
    時間発展の可視化
    """
    if len(snapshots) < 2:
        print("スナップショットが少なすぎます")
        return
    
    n_snapshots = min(6, len(snapshots))
    indices = np.linspace(0, len(snapshots)-1, n_snapshots, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    ls = LightSource(azdeg=315, altdeg=45)
    
    for idx, ax_idx in enumerate(indices):
        ax = axes[idx]
        snap = snapshots[ax_idx]
        rgb = ls.shade(snap['elevation'], cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
        ax.imshow(rgb, origin='lower')
        ax.set_title(f"{snap['time']/1000:.0f} 千年")
        ax.axis('off')
    
    plt.suptitle('地形の時間発展', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"タイムラプスを {output_file} に保存しました")
    
    return fig


# ========================================
# メイン実行
# ========================================
if __name__ == "__main__":
    # シミュレーション実行
    results = run_land_evolution_simulation(
        nrows=100,
        ncols=100,
        dx=50.0,            # 50m解像度
        uplift_rate=0.001,  # 1mm/yr
        k_sp=1.0e-5,        # 河川侵食（Hatari Labs推奨）
        k_hs=0.05,          # 斜面拡散（Hatari Labs推奨）
        dt=1000,            # 1000年/ステップ
        tmax=100000,        # 10万年
        seed=42,
        use_depression_finder=False,  # 高速化のため無効
    )
    
    # 結果の可視化
    visualize_results(results)
    
    # 時間発展の可視化
    if results['snapshots']:
        visualize_time_evolution(results['snapshots'])
    
    plt.show()
