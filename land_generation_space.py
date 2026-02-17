"""
SPACE モジュールを使った沿岸地形進化シミュレーション

SPACE (Stream Power with Alluvium Conservation and Entrainment) の特徴:
- 侵食だけでなく「堆積」も扱える
- 基盤岩の侵食と堆積物（沖積層）の侵食を分離
- 河口三角州、扇状地、砂州などの堆積地形が表現可能

StreamPowerEroder vs SPACE:
------------------------------------------
| 特徴              | StreamPower | SPACE |
|-------------------|-------------|-------|
| 侵食              | ○           | ○     |
| 堆積              | ×           | ○     |
| 堆積物層の追跡    | ×           | ○     |
| 河口三角州        | △（弱い）   | ○     |
| 計算コスト        | 低          | 中    |
------------------------------------------

SPACE パラメータ:
- K_sed: 堆積物の侵食係数 (通常 K_br より大きい)
- K_br:  基盤岩の侵食係数
- F_f:   細粒分の割合 (流れ去る割合、0-1)
- phi:   堆積物の間隙率 (通常 0.3-0.4)
- H_star: 土壌生産のスケール [m]
- v_s:   沈降速度 [m/yr]
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
    SpaceLargeScaleEroder,
    DepressionFinderAndRouter,
)


# ========================================
# パラメータ設定
# ========================================
NROWS = 150
NCOLS = 150
DX = 50.0

# 地形進化パラメータ
UPLIFT_RATE = 0.001   # 隆起速度 [m/yr]
K_BR = 1.0e-5         # 基盤岩の侵食係数
K_SED = 3.0e-5        # 堆積物の侵食係数（基盤岩より侵食されやすい）
M_SP = 0.5
N_SP = 1.0
K_HS = 0.05           # 斜面拡散係数

# SPACE 固有パラメータ
PHI = 0.3             # 堆積物の間隙率
H_STAR = 0.5          # 土壌生産スケール [m]
V_S = 1.0             # 沈降速度 [m/yr]
F_F = 0.0             # 細粒分割合（0 = すべて堆積、1 = すべて流出）

# 時間設定
DT = 1000
TMAX = 300000         # 30万年

OUTPUT_DIR = Path(__file__).resolve().parent / "archive" / "trial_images"


def resolve_output_path(filename):
    """出力先を archive/trial_images に固定"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / Path(filename).name


def generate_coastal_terrain(nrows, ncols, seed=42, amplitude=300.0, land_fraction=0.75):
    """海岸を含む初期地形を生成"""
    terrain = np.zeros((nrows, ncols))
    
    for octave in range(5):
        freq = 2 ** octave
        amp = 0.5 ** octave
        for i in range(nrows):
            for j in range(ncols):
                val = pnoise2(i * freq / 80.0, j * freq / 80.0, base=seed + octave)
                terrain[i, j] += val * amp
    
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    terrain *= amplitude
    
    # 海岸に向かう傾斜
    sea_boundary_row = int(nrows * (1 - land_fraction))
    for i in range(nrows):
        if i < sea_boundary_row:
            depth_factor = (sea_boundary_row - i) / sea_boundary_row
            terrain[i, :] -= amplitude * 0.3 * depth_factor
            terrain[i, :] -= 50.0
        else:
            distance_from_coast = i - sea_boundary_row
            terrain[i, :] += distance_from_coast * DX * 0.015
    
    return terrain, sea_boundary_row


def run_space_simulation(
    nrows=NROWS,
    ncols=NCOLS,
    dx=DX,
    uplift_rate=UPLIFT_RATE,
    k_br=K_BR,
    k_sed=K_SED,
    m_sp=M_SP,
    n_sp=N_SP,
    k_hs=K_HS,
    phi=PHI,
    h_star=H_STAR,
    v_s=V_S,
    f_f=F_F,
    dt=DT,
    tmax=TMAX,
    seed=42,
    land_fraction=0.75,
    beach_rows=5,
    save_interval=50000,
):
    """
    SPACE モジュールを使った沿岸地形進化シミュレーション
    """
    print("=" * 60)
    print("SPACE 沿岸地形進化シミュレーション")
    print("=" * 60)
    print(f"Grid: {nrows} x {ncols}, dx = {dx} m")
    print(f"Domain: {nrows * dx / 1000:.1f} km x {ncols * dx / 1000:.1f} km")
    print(f"Uplift rate: {uplift_rate * 1000:.1f} mm/yr")
    print(f"K_br (bedrock): {k_br:.1e}")
    print(f"K_sed (sediment): {k_sed:.1e} ({k_sed/k_br:.1f}x bedrock)")
    print(f"Sediment porosity (phi): {phi}")
    print(f"Fine fraction (F_f): {f_f}")
    print(f"Simulation time: {tmax/1000:.0f} kyr")
    print("=" * 60)
    
    # ----- グリッド作成 -----
    mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
    
    # ----- 初期地形 -----
    terrain_2d, sea_boundary_row = generate_coastal_terrain(
        nrows, ncols, seed=seed, amplitude=300.0, land_fraction=land_fraction
    )
    
    z = mg.add_zeros("topographic__elevation", at="node")
    z += terrain_2d.ravel()
    z_initial = z.copy()
    
    # ----- 堆積物層（初期は薄い表土）-----
    soil = mg.add_zeros("soil__depth", at="node")
    soil += 0.5  # 初期50cm の表土
    
    # ----- 境界条件 -----
    mg.set_closed_boundaries_at_grid_edges(
        right_is_closed=True,
        top_is_closed=True,
        left_is_closed=True,
        bottom_is_closed=True,
    )
    
    # 海側を固定値境界に
    bottom_nodes = mg.nodes_at_bottom_edge
    mg.status_at_node[bottom_nodes] = mg.BC_NODE_IS_FIXED_VALUE
    sea_level = 0.0
    z[bottom_nodes] = sea_level
    
    # ----- 海浜帯の設定（拡散係数を空間分布化）-----
    beach_nodes = []
    for row in range(beach_rows):
        start_node = row * ncols
        end_node = start_node + ncols
        beach_nodes.extend(range(start_node, end_node))
    beach_nodes = np.array(beach_nodes)
    
    # 拡散係数の空間分布
    k_hs_field = mg.add_ones("linear_diffusivity", at="link", clobber=True) * k_hs
    for node in beach_nodes:
        links = mg.links_at_node[node]
        valid_links = links[links != -1]
        k_hs_field[valid_links] = 5.0 * k_hs
    
    # ----- コンポーネント初期化 -----
    fa = FlowAccumulator(mg, flow_director='D8')
    ld = LinearDiffuser(mg, linear_diffusivity=k_hs_field, deposit=False)
    
    # SPACE コンポーネント
    space = SpaceLargeScaleEroder(
        mg,
        K_sed=k_sed,
        K_br=k_br,
        F_f=f_f,
        phi=phi,
        H_star=h_star,
        v_s=v_s,
        m_sp=m_sp,
        n_sp=n_sp,
        sp_crit_sed=0,
        sp_crit_br=0,
    )
    
    # 窪地処理
    df = DepressionFinderAndRouter(mg)
    
    # ----- 記録用 -----
    snapshots = []
    
    # ----- シミュレーションループ -----
    total_time = 0
    t_values = np.arange(0, tmax, dt)
    
    print(f"\nSimulation starting...")
    
    for ti in t_values:
        # 1. 地殻隆起（内陸のみ）
        inland_core = np.setdiff1d(mg.core_nodes, beach_nodes)
        z[inland_core] += uplift_rate * dt
        
        # 2. 斜面拡散
        ld.run_one_step(dt)
        
        # 3. 流路計算
        fa.run_one_step()
        
        # 4. 窪地処理
        df.map_depressions()
        
        # 5. SPACE（侵食＋堆積）
        space.run_one_step(dt)
        
        # 6. 海面以下を制限
        z[z < sea_level - 100] = sea_level - 100
        
        total_time += dt
        
        # 進捗表示
        if total_time % (tmax // 10) == 0:
            soil_mean = soil[mg.core_nodes].mean()
            print(f"  {total_time/1000:.0f} kyr... (mean soil depth: {soil_mean:.2f} m)")
        
        # スナップショット
        if total_time % save_interval == 0:
            snapshots.append({
                'time': total_time,
                'elevation': z.copy().reshape((nrows, ncols)),
                'soil_depth': soil.copy().reshape((nrows, ncols)),
                'drainage': mg.at_node["drainage_area"].copy().reshape((nrows, ncols)),
            })
    
    print(f"\nSimulation complete: {total_time/1000:.0f} kyr")
    
    # ----- 最終結果 -----
    z_final = z.reshape((nrows, ncols))
    soil_final = soil.reshape((nrows, ncols))
    drainage = mg.at_node["drainage_area"].reshape((nrows, ncols))
    z_change = z_final - z_initial.reshape((nrows, ncols))
    
    print(f"\n=== Results Summary ===")
    print(f"Initial elevation: {z_initial.min():.1f} - {z_initial.max():.1f} m")
    print(f"Final elevation: {z_final.min():.1f} - {z_final.max():.1f} m")
    print(f"Max soil depth: {soil_final.max():.2f} m")
    print(f"Mean soil depth (land): {soil_final[z_final > sea_level].mean():.2f} m")
    
    return {
        'grid': mg,
        'z_initial': z_initial.reshape((nrows, ncols)),
        'z_final': z_final,
        'soil_depth': soil_final,
        'drainage': drainage,
        'z_change': z_change,
        'snapshots': snapshots,
        'sea_level': sea_level,
        'params': {
            'nrows': nrows, 'ncols': ncols, 'dx': dx,
            'uplift_rate': uplift_rate, 'k_br': k_br, 'k_sed': k_sed,
            'k_hs': k_hs, 'phi': phi, 'f_f': f_f,
            'tmax': tmax, 'dt': dt,
        }
    }


def visualize_space_results(results, output_file="space_result.png"):
    """SPACE シミュレーション結果の可視化"""
    z_initial = results['z_initial']
    z_final = results['z_final']
    soil_depth = results['soil_depth']
    drainage = results['drainage']
    z_change = results['z_change']
    params = results['params']
    sea_level = results['sea_level']
    
    nrows, ncols = z_final.shape
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ls = LightSource(azdeg=315, altdeg=45)
    
    sea_mask = z_final < sea_level
    
    # ----- 1. Initial terrain -----
    ax = axes[0, 0]
    rgb = ls.shade(z_initial, cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    ax.set_title('Initial Terrain')
    ax.axis('off')
    
    # ----- 2. Final terrain -----
    ax = axes[0, 1]
    rgb = ls.shade(z_final, cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    ax.imshow(sea_mask, cmap='Blues', alpha=0.4, origin='lower')
    ax.contour(z_final, levels=[sea_level], colors='navy', linewidths=2, origin='lower')
    ax.set_title(f'Final Terrain ({params["tmax"]/1000:.0f} kyr)\nSea level: {sea_level:.1f} m')
    ax.axis('off')
    
    # ----- 3. Elevation change -----
    ax = axes[0, 2]
    vmax = max(abs(z_change.min()), abs(z_change.max()))
    im = ax.imshow(z_change, cmap='RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    ax.contour(z_final, levels=[sea_level], colors='black', linewidths=1, linestyles='--', origin='lower')
    ax.set_title('Elevation Change\n(blue=erosion, red=deposition/uplift)')
    plt.colorbar(im, ax=ax, label='Change [m]')
    ax.axis('off')
    
    # ----- 4. Soil/Sediment depth (SPACE specific!) -----
    ax = axes[1, 0]
    soil_masked = np.ma.masked_where(sea_mask, soil_depth)
    im = ax.imshow(soil_masked, cmap='YlOrBr', origin='lower', vmin=0)
    ax.contour(z_final, levels=[sea_level], colors='blue', linewidths=1, origin='lower')
    ax.set_title('Sediment/Soil Depth [m]\n(SPACE feature: tracks deposition)')
    plt.colorbar(im, ax=ax, label='Depth [m]')
    ax.axis('off')
    
    # ----- 5. Terrain + Rivers -----
    ax = axes[1, 1]
    rgb = ls.shade(z_final, cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    ax.imshow(sea_mask, cmap='Blues', alpha=0.4, origin='lower')
    
    # Rivers
    land_drainage = drainage.copy()
    land_drainage[sea_mask] = 0
    if np.any(land_drainage > 0):
        levels = [np.percentile(land_drainage[land_drainage > 0], p) for p in [90, 95, 98]]
        ax.contour(drainage, levels=levels, colors=['cyan', 'blue', 'darkblue'],
                   linewidths=[0.5, 1, 2], origin='lower')
    
    ax.contour(z_final, levels=[sea_level], colors='yellow', linewidths=2, origin='lower')
    ax.set_title('Terrain + Rivers + Coastline')
    ax.axis('off')
    
    # ----- 6. Parameter info -----
    ax = axes[1, 2]
    ax.axis('off')
    
    land_area_km2 = np.sum(~sea_mask) * (params['dx'] / 1000) ** 2
    
    param_text = f"""
    === SPACE Coastal Simulation ===
    
    Grid: {params['nrows']} x {params['ncols']} cells
    Resolution: {params['dx']} m
    Land area: {land_area_km2:.1f} km²
    
    --- Erosion Parameters ---
    K_br (bedrock): {params['k_br']:.1e}
    K_sed (sediment): {params['k_sed']:.1e}
    K_sed / K_br: {params['k_sed']/params['k_br']:.1f}x
    
    --- Deposition Parameters ---
    Porosity (phi): {params['phi']}
    Fine fraction (F_f): {params['f_f']}
    
    --- Other ---
    Uplift rate: {params['uplift_rate']*1000:.1f} mm/yr
    K_hs (diffusion): {params['k_hs']} m²/yr
    Time: {params['tmax']/1000:.0f} kyr
    
    === Results ===
    Max elevation: {z_final.max():.1f} m
    Max sediment depth: {soil_depth.max():.2f} m
    Mean sediment (land): {soil_depth[~sea_mask].mean():.2f} m
    """
    ax.text(0.05, 0.95, param_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = resolve_output_path(output_file)
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    
    return fig


def visualize_sediment_evolution(snapshots, sea_level=0.0, output_file="space_sediment_evolution.png"):
    """堆積物厚の時間発展を表示"""
    if len(snapshots) < 2:
        print("Not enough snapshots")
        return
    
    n_snapshots = min(6, len(snapshots))
    indices = np.linspace(0, len(snapshots)-1, n_snapshots, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, snap_idx in enumerate(indices):
        ax = axes[idx]
        snap = snapshots[snap_idx]
        soil = snap['soil_depth']
        z = snap['elevation']
        
        sea_mask = z < sea_level
        soil_masked = np.ma.masked_where(sea_mask, soil)
        
        im = ax.imshow(soil_masked, cmap='YlOrBr', origin='lower', vmin=0, vmax=5)
        ax.contour(z, levels=[sea_level], colors='blue', linewidths=1, origin='lower')
        ax.set_title(f"{snap['time']/1000:.0f} kyr\nmax: {soil.max():.2f} m")
        ax.axis('off')
    
    plt.suptitle('Sediment Depth Evolution (SPACE)', fontsize=14)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Sediment Depth [m]')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    save_path = resolve_output_path(output_file)
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    
    return fig


def compare_with_without_deposition():
    """
    堆積の有無（F_f パラメータ）による違いを比較
    F_f = 0: すべて堆積
    F_f = 1: すべて流出（堆積なし）
    """
    print("\n" + "="*60)
    print("Comparison: Deposition ON vs OFF")
    print("="*60)
    
    # 共通パラメータ
    common_params = dict(
        nrows=120, ncols=120, dx=50.0,
        tmax=200000,  # 20万年
        seed=42,
        save_interval=50000,
    )
    
    # Case 1: 堆積あり (F_f = 0)
    print("\n[Case 1] F_f = 0 (full deposition)")
    results_dep = run_space_simulation(f_f=0.0, **common_params)
    
    # Case 2: 堆積なし (F_f = 1)
    print("\n[Case 2] F_f = 1 (no deposition, like StreamPower)")
    results_nodep = run_space_simulation(f_f=1.0, **common_params)
    
    # 比較可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ls = LightSource(azdeg=315, altdeg=45)
    
    sea_level = 0.0
    
    # Row 1: With deposition
    ax = axes[0, 0]
    rgb = ls.shade(results_dep['z_final'], cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    ax.imshow(results_dep['z_final'] < sea_level, cmap='Blues', alpha=0.4, origin='lower')
    ax.set_title('F_f=0 (Full Deposition)\nFinal Terrain')
    ax.axis('off')
    
    ax = axes[0, 1]
    im = ax.imshow(results_dep['soil_depth'], cmap='YlOrBr', origin='lower', vmin=0)
    ax.set_title('Sediment Depth')
    plt.colorbar(im, ax=ax, label='[m]')
    ax.axis('off')
    
    ax = axes[0, 2]
    vmax = max(abs(results_dep['z_change'].min()), abs(results_dep['z_change'].max()))
    im = ax.imshow(results_dep['z_change'], cmap='RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    ax.set_title('Elevation Change')
    plt.colorbar(im, ax=ax, label='[m]')
    ax.axis('off')
    
    # Row 2: Without deposition
    ax = axes[1, 0]
    rgb = ls.shade(results_nodep['z_final'], cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    ax.imshow(results_nodep['z_final'] < sea_level, cmap='Blues', alpha=0.4, origin='lower')
    ax.set_title('F_f=1 (No Deposition)\nFinal Terrain')
    ax.axis('off')
    
    ax = axes[1, 1]
    im = ax.imshow(results_nodep['soil_depth'], cmap='YlOrBr', origin='lower', vmin=0)
    ax.set_title('Sediment Depth\n(minimal)')
    plt.colorbar(im, ax=ax, label='[m]')
    ax.axis('off')
    
    ax = axes[1, 2]
    im = ax.imshow(results_nodep['z_change'], cmap='RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    ax.set_title('Elevation Change')
    plt.colorbar(im, ax=ax, label='[m]')
    ax.axis('off')
    
    plt.suptitle('SPACE: Effect of Fine Fraction (F_f)\nTop: Deposition ON | Bottom: Deposition OFF', fontsize=14)
    plt.tight_layout()
    comparison_path = resolve_output_path("space_comparison.png")
    plt.savefig(comparison_path, dpi=150)
    print(f"\nSaved: {comparison_path}")
    
    return results_dep, results_nodep


# ========================================
# メイン実行
# ========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SPACE Coastal Evolution Simulation")
    print("="*60 + "\n")
    
    # ---- 基本シミュレーション ----
    print("【Basic SPACE Simulation】")
    results = run_space_simulation(
        nrows=150,
        ncols=150,
        dx=50.0,
        uplift_rate=0.001,
        k_br=1.0e-5,
        k_sed=3.0e-5,  # 堆積物は3倍侵食されやすい
        k_hs=0.05,
        phi=0.3,
        f_f=0.0,       # 全堆積
        dt=1000,
        tmax=300000,   # 30万年
        seed=42,
        land_fraction=0.75,
        save_interval=50000,
    )
    
    visualize_space_results(results, "space_coastal.png")
    visualize_sediment_evolution(results['snapshots'], output_file="space_sediment_evolution.png")
    
    # ---- 堆積の有無を比較 ----
    print("\n【Comparison: Deposition ON vs OFF】")
    compare_with_without_deposition()
    
    plt.show()
    
    print("\n" + "="*60)
    print("Complete! Generated images:")
    print("  - space_coastal.png            : SPACE basic result")
    print("  - space_sediment_evolution.png : Sediment depth over time")
    print("  - space_comparison.png         : F_f=0 vs F_f=1 comparison")
    print("="*60)
