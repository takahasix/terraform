"""
沿岸地形進化シミュレーション - 海辺を含む地形生成

既存の land_evolution.py を拡張し、以下の沿岸プロセスを追加:
1. 海をベースレベル（固定境界）として設定
2. 海面変動（ユースタシー）のサポート
3. 海浜帯での高拡散（波による平滑化の近似）
4. 海浜帯での侵食係数の空間分布（未固結堆積物）

参考文献:
- Landlab SPACE コンポーネント（侵食・堆積両対応）
- 沿岸漂砂は厳密には別モデルが必要だが、高拡散で近似

海面変動シナリオ:
------------------------------------------
| シナリオ | 説明                          |
|---------|-------------------------------|
| static  | 海面固定（最もシンプル）         |
| rising  | 海面上昇（温暖化、沈降）        |
| falling | 海面低下（寒冷化、隆起）        |
| cycle   | 周期的変動（氷期-間氷期サイクル）|
------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LightSource
from noise import pnoise2
from landlab import RasterModelGrid
from landlab.components import (
    FlowAccumulator,
    LinearDiffuser,
    StreamPowerEroder,
    DepressionFinderAndRouter,
)

# オプション: SPACE（侵食・堆積両対応）
try:
    from landlab.components import SpaceLargeScaleEroder
    HAS_SPACE = True
except ImportError:
    HAS_SPACE = False
    print("注意: SpaceLargeScaleEroder が利用できません。StreamPowerEroder を使用します。")


# ========================================
# パラメータ設定
# ========================================
# グリッド設定
NROWS = 150       # 行数
NCOLS = 150       # 列数
DX = 50.0         # セルサイズ [m]

# 地形進化パラメータ
UPLIFT_RATE = 0.001   # 隆起速度 [m/yr]
K_SP = 1.0e-5         # 河川侵食係数
M_SP = 0.5            # 集水面積の指数
N_SP = 1.0            # 勾配の指数
K_HS = 0.05           # 斜面拡散係数 [m²/yr]

# 沿岸パラメータ
SEA_LEVEL_INITIAL = 0.0       # 初期海面 [m]
BEACH_ROWS = 5                # 海浜帯の行数（グリッド下端から）
BEACH_K_HS_MULTIPLIER = 5.0   # 海浜帯での拡散係数倍率
BEACH_K_SP_MULTIPLIER = 3.0   # 海浜帯での侵食係数倍率

# 時間設定
DT = 1000             # タイムステップ [yr]
TMAX = 500000         # 総シミュレーション時間 [yr]（50万年）

OUTPUT_DIR = Path(__file__).resolve().parent / "archive" / "trial_images"


def resolve_output_path(filename):
    """出力先を archive/trial_images に固定"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / Path(filename).name


# ========================================
# 海面変動シナリオ
# ========================================
class SeaLevelScenario:
    """海面変動を管理するクラス"""
    
    @staticmethod
    def static(t, base_level=0.0):
        """海面固定"""
        return base_level
    
    @staticmethod
    def rising(t, rate=0.0005, base_level=0.0):
        """
        海面上昇
        rate: 上昇速度 [m/yr]（デフォルト: 0.5 mm/yr）
        """
        return base_level + rate * t
    
    @staticmethod
    def falling(t, rate=0.0005, base_level=0.0):
        """
        海面低下
        rate: 低下速度 [m/yr]
        """
        return base_level - rate * t
    
    @staticmethod
    def cycle(t, amplitude=50.0, period=100000, base_level=0.0):
        """
        周期的海面変動（氷期-間氷期サイクル）
        amplitude: 振幅 [m]（デフォルト: 50m、氷期の典型値）
        period: 周期 [yr]（デフォルト: 10万年、ミランコビッチサイクル）
        """
        return base_level + amplitude * np.sin(2 * np.pi * t / period)
    
    @staticmethod
    def custom(t, func):
        """カスタム関数による海面変動"""
        return func(t)


def generate_coastal_terrain(nrows, ncols, seed=42, amplitude=300.0, 
                             land_fraction=0.7, coastal_slope=0.02):
    """
    海岸を含む初期地形を生成
    
    Parameters:
    -----------
    nrows, ncols : int
        グリッドサイズ
    seed : int
        乱数シード
    amplitude : float
        内陸部の最大標高 [m]
    land_fraction : float
        陸地の割合（0-1）。0.7 なら下端30%が海
    coastal_slope : float
        海岸線付近の傾斜（海に向かって下がる）
    
    Returns:
    --------
    terrain : ndarray
        初期標高（nrows x ncols）
    """
    terrain = np.zeros((nrows, ncols))
    
    # パーリンノイズで基本地形
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
    
    # 海岸に向かう傾斜を追加（南端が低い）
    # land_fraction より下は海面以下になるように調整
    sea_boundary_row = int(nrows * (1 - land_fraction))
    
    for i in range(nrows):
        if i < sea_boundary_row:
            # 海域：海面以下に沈める
            depth_factor = (sea_boundary_row - i) / sea_boundary_row
            terrain[i, :] -= amplitude * 0.3 * depth_factor  # 海底の深さ
            terrain[i, :] -= 50.0  # 海面より下に
        else:
            # 陸域：海岸から内陸に向かって上昇
            distance_from_coast = i - sea_boundary_row
            terrain[i, :] += distance_from_coast * DX * coastal_slope
    
    return terrain, sea_boundary_row


def setup_coastal_boundaries(mg, sea_boundary_row, sea_level=0.0):
    """
    沿岸の境界条件を設定
    
    海側（南端）を固定値境界、他を閉じた境界にする
    """
    # すべての境界を閉じる
    mg.set_closed_boundaries_at_grid_edges(
        right_is_closed=True,
        top_is_closed=True,
        left_is_closed=True,
        bottom_is_closed=True,  # まず閉じる
    )
    
    # 海側（bottom）を固定値境界に設定
    # bottom edge のノードを取得
    bottom_nodes = mg.nodes_at_bottom_edge
    
    # 固定値境界として設定
    mg.status_at_node[bottom_nodes] = mg.BC_NODE_IS_FIXED_VALUE
    
    # 海面高度を設定
    z = mg.at_node['topographic__elevation']
    z[bottom_nodes] = sea_level
    
    return bottom_nodes


def identify_beach_zone(mg, nrows, ncols, beach_rows=BEACH_ROWS):
    """
    海浜帯のノードを特定
    
    海岸線から beach_rows 行分を海浜帯として定義
    """
    beach_nodes = []
    
    # 最下端から beach_rows 行分を取得
    for row in range(beach_rows):
        start_node = row * ncols
        end_node = start_node + ncols
        beach_nodes.extend(range(start_node, end_node))
    
    return np.array(beach_nodes)


def setup_spatial_diffusivity(mg, beach_nodes, k_hs_base, multiplier=BEACH_K_HS_MULTIPLIER):
    """
    斜面拡散係数の空間分布を設定
    
    海浜帯では拡散を強くする（波による平滑化の近似）
    """
    # リンクごとの拡散係数フィールドを作成
    k_hs_field = mg.add_ones("linear_diffusivity", at="link", clobber=True) * k_hs_base
    
    # beach_nodes に隣接するリンクを特定してブースト
    for node in beach_nodes:
        links = mg.links_at_node[node]
        valid_links = links[links != -1]  # 無効なリンクを除外
        k_hs_field[valid_links] = multiplier * k_hs_base
    
    return k_hs_field


def setup_spatial_erodibility(mg, beach_nodes, k_sp_base, multiplier=BEACH_K_SP_MULTIPLIER):
    """
    侵食係数の空間分布を設定
    
    海浜帯は未固結堆積物なので侵食されやすい（K_sp が大きい）
    """
    # ノードごとの侵食係数フィールドを作成
    k_sp_field = mg.add_ones("K_sp", at="node", clobber=True) * k_sp_base
    
    # 海浜帯は侵食係数を大きく
    k_sp_field[beach_nodes] = multiplier * k_sp_base
    
    return k_sp_field


def run_coastal_evolution(
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
    # 沿岸固有パラメータ
    sea_level_scenario='static',
    sea_level_params=None,
    land_fraction=0.8,
    beach_rows=BEACH_ROWS,
    beach_k_hs_mult=BEACH_K_HS_MULTIPLIER,
    beach_k_sp_mult=BEACH_K_SP_MULTIPLIER,
    use_space=False,
    use_depression_finder=False,
    save_interval=50000,
):
    """
    沿岸地形進化シミュレーションを実行
    
    Parameters:
    -----------
    sea_level_scenario : str
        海面変動シナリオ（'static', 'rising', 'falling', 'cycle'）
    sea_level_params : dict
        海面変動パラメータ（シナリオに応じて）
    land_fraction : float
        陸地の割合（0.7 なら下端30%が初期海域）
    beach_rows : int
        海浜帯の幅（グリッド行数）
    beach_k_hs_mult : float
        海浜帯での拡散係数の倍率
    beach_k_sp_mult : float
        海浜帯での侵食係数の倍率
    use_space : bool
        SPACE（侵食-堆積モデル）を使用するか
    """
    print("=" * 60)
    print("沿岸地形進化シミュレーション")
    print("=" * 60)
    print(f"グリッド: {nrows} x {ncols} セル, 解像度 {dx} m")
    print(f"領域サイズ: {nrows * dx / 1000:.1f} km x {ncols * dx / 1000:.1f} km")
    print(f"陸地割合: {land_fraction * 100:.0f}%")
    print(f"海面変動: {sea_level_scenario}")
    print(f"海浜帯: {beach_rows} 行 (拡散 {beach_k_hs_mult}x, 侵食 {beach_k_sp_mult}x)")
    print(f"隆起速度: {uplift_rate} m/yr ({uplift_rate*1000:.1f} mm/yr)")
    print(f"シミュレーション時間: {tmax/1000:.0f} 千年")
    if use_space and not HAS_SPACE:
        print("警告: SPACE モジュールが見つかりません。StreamPowerEroder を使用します。")
        use_space = False
    print("=" * 60)
    
    # ----- グリッド作成 -----
    mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
    
    # ----- 初期地形 -----
    terrain_2d, sea_boundary_row = generate_coastal_terrain(
        nrows, ncols, seed=seed, 
        amplitude=300.0,
        land_fraction=land_fraction,
        coastal_slope=0.015,
    )
    
    z = mg.add_zeros("topographic__elevation", at="node")
    z += terrain_2d.ravel()
    z_initial = z.copy()
    
    # ----- 境界条件（海側を固定）-----
    sea_level_current = SEA_LEVEL_INITIAL
    fixed_nodes = setup_coastal_boundaries(mg, sea_boundary_row, sea_level_current)
    
    # ----- 海浜帯の設定 -----
    beach_nodes = identify_beach_zone(mg, nrows, ncols, beach_rows)
    
    # ----- 空間分布の拡散係数 -----
    k_hs_field = setup_spatial_diffusivity(mg, beach_nodes, k_hs, beach_k_hs_mult)
    
    # ----- 空間分布の侵食係数 -----
    k_sp_field = setup_spatial_erodibility(mg, beach_nodes, k_sp, beach_k_sp_mult)
    
    # ----- コンポーネント初期化 -----
    fa = FlowAccumulator(mg, flow_director='D8')
    ld = LinearDiffuser(mg, linear_diffusivity=k_hs_field, deposit=False)
    
    if use_space and HAS_SPACE:
        # SPACE: 侵食と堆積の両方を扱う
        sp = SpaceLargeScaleEroder(
            mg,
            K_sed=k_sp * 2,  # 堆積物の侵食係数
            K_br=k_sp,       # 基盤岩の侵食係数
            m_sp=m_sp,
            n_sp=n_sp,
            phi=0.3,         # 堆積物の間隙率
            H_star=1.0,      # 土壌生産のスケール
            v_s=1.0,         # 沈降速度
        )
    else:
        # StreamPowerEroder（侵食のみ）
        sp = StreamPowerEroder(mg, K_sp=k_sp_field, m_sp=m_sp, n_sp=n_sp, threshold_sp=0.0)
    
    if use_depression_finder:
        df = DepressionFinderAndRouter(mg)
    
    # ----- 海面変動関数の設定 -----
    sl_params = sea_level_params or {}
    if sea_level_scenario == 'static':
        sea_level_func = lambda t: SeaLevelScenario.static(t, **sl_params)
    elif sea_level_scenario == 'rising':
        sea_level_func = lambda t: SeaLevelScenario.rising(t, **sl_params)
    elif sea_level_scenario == 'falling':
        sea_level_func = lambda t: SeaLevelScenario.falling(t, **sl_params)
    elif sea_level_scenario == 'cycle':
        sea_level_func = lambda t: SeaLevelScenario.cycle(t, **sl_params)
    else:
        sea_level_func = lambda t: 0.0
    
    # ----- 中間結果保存用 -----
    snapshots = []
    sea_level_history = []
    
    # ----- シミュレーションループ -----
    total_time = 0
    t_values = np.arange(0, tmax, dt)
    
    print(f"\nシミュレーション開始...")
    
    for ti in t_values:
        # 1. 地殻隆起（陸側のみ、海浜帯を除く内陸部）
        core_nodes = mg.core_nodes
        inland_core = np.setdiff1d(core_nodes, beach_nodes)
        z[inland_core] += uplift_rate * dt
        
        # 2. 海面更新
        sea_level_current = sea_level_func(total_time)
        z[fixed_nodes] = sea_level_current
        
        # 3. 斜面拡散（海浜帯は強拡散）
        ld.run_one_step(dt)
        
        # 4. 流路計算
        fa.run_one_step()
        
        # 5. 窪地処理（オプション）
        if use_depression_finder:
            df.map_depressions()
        
        # 6. 河川侵食（または SPACE）
        sp.run_one_step(dt)
        
        # 7. 海面以下を強制（海が陸より高くならないように）
        z[z < sea_level_current - 100] = sea_level_current - 100  # 最大水深100m
        
        total_time += dt
        
        # 進捗表示
        if total_time % (tmax // 10) == 0:
            print(f"  {total_time/1000:.0f} 千年経過... (海面: {sea_level_current:.1f} m)")
        
        # スナップショット保存
        if total_time % save_interval == 0:
            snapshots.append({
                'time': total_time,
                'elevation': z.copy().reshape((nrows, ncols)),
                'drainage': mg.at_node["drainage_area"].copy().reshape((nrows, ncols)),
                'sea_level': sea_level_current,
            })
            sea_level_history.append((total_time, sea_level_current))
    
    print(f"\nシミュレーション完了: {total_time/1000:.0f} 千年")
    print(f"最終海面: {sea_level_current:.1f} m")
    
    # ----- 最終結果 -----
    z_final = z.reshape((nrows, ncols))
    drainage = mg.at_node["drainage_area"].reshape((nrows, ncols))
    z_change = z_final - z_initial.reshape((nrows, ncols))
    
    print(f"\n=== 結果サマリー ===")
    print(f"初期標高: {z_initial.min():.1f} - {z_initial.max():.1f} m")
    print(f"最終標高: {z_final.min():.1f} - {z_final.max():.1f} m")
    print(f"陸地最高点: {z_final[z_final > sea_level_current].max():.1f} m")
    print(f"海面: {sea_level_current:.1f} m")
    
    return {
        'grid': mg,
        'z_initial': z_initial.reshape((nrows, ncols)),
        'z_final': z_final,
        'drainage': drainage,
        'z_change': z_change,
        'snapshots': snapshots,
        'sea_level_history': sea_level_history,
        'final_sea_level': sea_level_current,
        'beach_nodes': beach_nodes,
        'params': {
            'nrows': nrows, 'ncols': ncols, 'dx': dx,
            'uplift_rate': uplift_rate, 'k_sp': k_sp, 'k_hs': k_hs,
            'tmax': tmax, 'dt': dt,
            'land_fraction': land_fraction,
            'beach_rows': beach_rows,
            'sea_level_scenario': sea_level_scenario,
        }
    }


def visualize_coastal_results(results, output_file="coastal_evolution_result.png"):
    """
    沿岸シミュレーション結果の可視化
    
    海陸境界、海浜帯、海面変動を考慮した表示
    """
    z_initial = results['z_initial']
    z_final = results['z_final']
    drainage = results['drainage']
    z_change = results['z_change']
    params = results['params']
    sea_level = results['final_sea_level']
    
    nrows, ncols = z_final.shape
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ls = LightSource(azdeg=315, altdeg=45)
    
    # 海のマスク
    sea_mask = z_final < sea_level
    
    # ----- 1. 初期地形 -----
    ax = axes[0, 0]
    rgb = ls.shade(z_initial, cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    ax.set_title('初期地形')
    
    # ----- 2. 最終地形（海を青で表示）-----
    ax = axes[0, 1]
    
    # 陸地と海で別々に色付け
    terrain_masked = np.ma.masked_where(sea_mask, z_final)
    sea_masked = np.ma.masked_where(~sea_mask, z_final)
    
    rgb_land = ls.shade(z_final, cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb_land, origin='lower')
    
    # 海を半透明の青でオーバーレイ
    ax.imshow(sea_mask, cmap='Blues', alpha=0.5, origin='lower')
    
    # 海岸線を描画
    ax.contour(z_final, levels=[sea_level], colors='navy', linewidths=2, origin='lower')
    ax.set_title(f'最終地形 ({params["tmax"]/1000:.0f} 千年後)\n海面: {sea_level:.1f} m')
    
    # ----- 3. 標高変化 -----
    ax = axes[0, 2]
    vmax = max(abs(z_change.min()), abs(z_change.max()))
    im = ax.imshow(z_change, cmap='RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    ax.contour(z_final, levels=[sea_level], colors='black', linewidths=1, linestyles='--', origin='lower')
    ax.set_title('標高変化\n(青=侵食, 赤=堆積/隆起)')
    plt.colorbar(im, ax=ax, label='変化量 [m]')
    
    # ----- 4. 集水面積（河川）-----
    ax = axes[1, 0]
    log_drainage = np.log10(drainage + 1)
    im = ax.imshow(log_drainage, cmap='Blues', origin='lower')
    ax.contour(z_final, levels=[sea_level], colors='red', linewidths=1, origin='lower')
    ax.set_title('集水面積 (log10)\n赤線 = 海岸線')
    plt.colorbar(im, ax=ax, label='log10(面積)')
    
    # ----- 5. 地形 + 河川 + 海岸線 -----
    ax = axes[1, 1]
    rgb = ls.shade(z_final, cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    
    # 海を表示
    ax.imshow(sea_mask, cmap='Blues', alpha=0.4, origin='lower')
    
    # 河川を重ねる（陸上のみ）
    river_threshold = np.percentile(drainage[~sea_mask], 90) if np.any(~sea_mask) else 100
    levels = [np.percentile(drainage[~sea_mask], p) for p in [90, 95, 98]] if np.any(~sea_mask) else [100, 1000, 10000]
    ax.contour(drainage, levels=levels, colors=['cyan', 'blue', 'darkblue'],
               linewidths=[0.5, 1, 2], origin='lower')
    
    # 海岸線
    ax.contour(z_final, levels=[sea_level], colors='yellow', linewidths=2, origin='lower')
    ax.set_title('最終地形 + 河川 + 海岸線\n(黄線 = 海岸線)')
    
    # ----- 6. パラメータ情報 -----
    ax = axes[1, 2]
    ax.axis('off')
    
    # 陸地面積の計算
    land_area_km2 = np.sum(~sea_mask) * (params['dx'] / 1000) ** 2
    total_area_km2 = nrows * ncols * (params['dx'] / 1000) ** 2
    
    param_text = f"""
    === 沿岸地形シミュレーション ===
    
    グリッド: {params['nrows']} x {params['ncols']} セル
    解像度: {params['dx']} m
    総面積: {total_area_km2:.1f} km²
    陸地面積: {land_area_km2:.1f} km² ({land_area_km2/total_area_km2*100:.0f}%)
    
    海面変動シナリオ: {params['sea_level_scenario']}
    海浜帯: {params['beach_rows']} 行
    
    隆起速度: {params['uplift_rate']} m/yr
            = {params['uplift_rate']*1000:.1f} mm/yr
    
    河川侵食係数 K_sp: {params['k_sp']:.1e}
    斜面拡散係数 K_hs: {params['k_hs']} m²/yr
    
    シミュレーション時間: {params['tmax']/1000:.0f} 千年
    
    === 結果 ===
    最終海面: {sea_level:.1f} m
    陸地最高点: {z_final[~sea_mask].max() if np.any(~sea_mask) else 0:.1f} m
    最大侵食量: {abs(z_change.min()):.1f} m
    """
    ax.text(0.05, 0.95, param_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = resolve_output_path(output_file)
    plt.savefig(save_path, dpi=150)
    print(f"\n結果を {save_path} に保存しました")
    
    return fig


def visualize_sea_level_evolution(results, output_file="sea_level_evolution.png"):
    """
    海面変動と地形変化の時系列表示
    """
    snapshots = results['snapshots']
    sea_level_history = results.get('sea_level_history', [])
    
    if len(snapshots) < 2:
        print("スナップショットが少なすぎます")
        return
    
    n_snapshots = min(6, len(snapshots))
    indices = np.linspace(0, len(snapshots)-1, n_snapshots, dtype=int)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 上段: 地形の時間発展
    ls = LightSource(azdeg=315, altdeg=45)
    
    for idx, snap_idx in enumerate(indices):
        ax = fig.add_subplot(2, 3, idx + 1)
        snap = snapshots[snap_idx]
        z = snap['elevation']
        sl = snap['sea_level']
        
        # 地形表示
        rgb = ls.shade(z, cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
        ax.imshow(rgb, origin='lower')
        
        # 海をマスク
        sea_mask = z < sl
        ax.imshow(sea_mask, cmap='Blues', alpha=0.4, origin='lower')
        
        # 海岸線
        ax.contour(z, levels=[sl], colors='yellow', linewidths=1.5, origin='lower')
        
        ax.set_title(f"{snap['time']/1000:.0f} 千年\n(海面: {sl:.1f} m)")
        ax.axis('off')
    
    plt.suptitle('沿岸地形の時間発展', fontsize=14)
    plt.tight_layout()
    save_path = resolve_output_path(output_file)
    plt.savefig(save_path, dpi=150)
    print(f"時間発展を {save_path} に保存しました")
    
    return fig


# ========================================
# メイン実行
# ========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("沿岸地形進化シミュレーション - デモ実行")
    print("="*60 + "\n")
    
    # ---- シナリオ1: 海面固定（基本形）----
    print("【シナリオ1】海面固定（基本形）")
    results_static = run_coastal_evolution(
        nrows=150,
        ncols=150,
        dx=50.0,
        uplift_rate=0.001,
        k_sp=1.0e-5,
        k_hs=0.05,
        dt=1000,
        tmax=200000,  # 20万年
        seed=42,
        sea_level_scenario='static',
        land_fraction=0.75,  # 25%が海
        beach_rows=5,
        beach_k_hs_mult=5.0,
        beach_k_sp_mult=3.0,
        save_interval=40000,
    )
    visualize_coastal_results(results_static, "coastal_static.png")
    
    # ---- シナリオ2: 海面上昇 ----
    print("\n【シナリオ2】海面上昇")
    results_rising = run_coastal_evolution(
        nrows=150,
        ncols=150,
        dx=50.0,
        uplift_rate=0.001,
        k_sp=1.0e-5,
        k_hs=0.05,
        dt=1000,
        tmax=200000,
        seed=42,
        sea_level_scenario='rising',
        sea_level_params={'rate': 0.0003, 'base_level': 0.0},  # 0.3 mm/yr 上昇
        land_fraction=0.75,
        beach_rows=5,
        save_interval=40000,
    )
    visualize_coastal_results(results_rising, "coastal_rising.png")
    
    # ---- シナリオ3: 周期的海面変動（氷期サイクル）----
    print("\n【シナリオ3】周期的海面変動（氷期-間氷期サイクル）")
    results_cycle = run_coastal_evolution(
        nrows=150,
        ncols=150,
        dx=50.0,
        uplift_rate=0.001,
        k_sp=1.0e-5,
        k_hs=0.05,
        dt=1000,
        tmax=400000,  # 40万年（複数サイクル）
        seed=42,
        sea_level_scenario='cycle',
        sea_level_params={'amplitude': 30.0, 'period': 100000, 'base_level': 0.0},
        land_fraction=0.75,
        beach_rows=5,
        save_interval=50000,
    )
    visualize_coastal_results(results_cycle, "coastal_cycle.png")
    visualize_sea_level_evolution(results_cycle, "coastal_cycle_evolution.png")
    
    plt.show()
    
    print("\n" + "="*60)
    print("完了！生成された画像:")
    print("  - coastal_static.png      : 海面固定シナリオ")
    print("  - coastal_rising.png      : 海面上昇シナリオ")
    print("  - coastal_cycle.png       : 周期的海面変動シナリオ")
    print("  - coastal_cycle_evolution.png : 時間発展")
    print("="*60)
