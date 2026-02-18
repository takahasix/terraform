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
from datetime import datetime
from matplotlib.colors import LightSource, ListedColormap
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
RELIEF_RATIO = 0.30   # 初期地形の高低差比率（ドメイン長に対する割合）

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


def create_history_dir(run_name="space_v2"):
    """途中経過画像の保存先ディレクトリを作成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_dir = OUTPUT_DIR / "history" / f"{timestamp}_{run_name}"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


def save_history_frame(z_2d, sea_level, out_path, title, particle_xy=None, drainage=None):
    """途中経過の標高画像を保存（海域マゼンダ＋海岸線黒＋川描画）"""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(z_2d, cmap=plt.get_cmap('terrain'), vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')

    # --- 海域をマゼンダで塗る ---
    sea_mask = z_2d <= sea_level
    if np.any(sea_mask):
        sea_overlay = np.ma.masked_where(~sea_mask, np.ones_like(z_2d, dtype=float))
        ax.imshow(
            sea_overlay,
            cmap=ListedColormap(['magenta']),
            alpha=0.32,
            origin='lower',
            vmin=0,
            vmax=1,
        )

    # --- 標高0m の境界線（黒） ---
    z_range = z_2d.max() - z_2d.min()
    if z_range > 1e-6 and z_2d.min() <= sea_level <= z_2d.max():
        ax.contour(z_2d, levels=[sea_level], colors='black', linewidths=1.6, origin='lower')

    # --- 川の描画（集水面積ベース） ---
    if drainage is not None:
        land_drainage = drainage.copy()
        land_drainage[sea_mask] = 0
        positive = land_drainage[land_drainage > 0]
        if len(positive) > 0:
            levels = [np.percentile(positive, p) for p in [90, 95, 98]]
            ax.contour(
                drainage, levels=levels,
                colors=['cyan', 'deepskyblue', 'blue'],
                linewidths=[0.5, 1.0, 2.0],
                origin='lower',
            )

    # --- パーティクル ---
    if particle_xy is not None:
        px, py = particle_xy
        sample = min(len(px), 1200)
        if sample > 0:
            idx = np.linspace(0, len(px) - 1, sample, dtype=int)
            ax.scatter(px[idx], py[idx], s=2, c='white', alpha=0.7, linewidths=0)

    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def initialize_particles(nrows, ncols, particle_count, seed=123):
    """粒子をランダム配置で初期化"""
    rng = np.random.default_rng(seed)
    px = rng.uniform(0, ncols - 1, size=particle_count)
    py = rng.uniform(0, nrows - 1, size=particle_count)
    sediment_load = np.zeros(particle_count, dtype=float)
    return px, py, sediment_load, rng


def update_sea_particles(
    z_2d,
    px,
    py,
    sediment_load,
    rng,
    drainage_2d=None,
    dx=50.0,
    dt=1000.0,
    energy_split_quantile=60.0,
):
    """エネルギー（勾配×流量proxy）に応じて粒子移動を切り替える。Fail Fastで不正値検知。"""
    nrows, ncols = z_2d.shape

    gx = np.gradient(z_2d, axis=1) / dx
    gy = np.gradient(z_2d, axis=0) / dx

    ix = np.clip(np.rint(px).astype(int), 0, ncols - 1)
    iy = np.clip(np.rint(py).astype(int), 0, nrows - 1)

    local_gx = gx[iy, ix]
    local_gy = gy[iy, ix]

    # 局所勾配
    slope_mag = np.hypot(local_gx, local_gy)

    # 流量proxy（集水面積）を0-1へ正規化してエネルギーを作る
    if drainage_2d is not None:
        local_da = np.maximum(drainage_2d[iy, ix], 0.0)
        da_max = np.max(drainage_2d)
        if da_max > 0.0:
            da_norm = np.log1p(local_da) / np.log1p(da_max)
        else:
            da_norm = np.zeros_like(local_da)
    else:
        da_norm = np.ones_like(slope_mag)

    energy = slope_mag * (0.35 + 0.65 * da_norm)
    energy_threshold = np.percentile(energy, energy_split_quantile)
    high_energy_mask = energy >= energy_threshold
    low_energy_mask = ~high_energy_mask

    vx = np.zeros_like(px)
    vy = np.zeros_like(py)

    # 陸上: 勾配に沿って緩く移動
    slope_vx = -local_gx
    slope_vy = -local_gy
    slope_norm = np.hypot(slope_vx, slope_vy)
    valid_slope = slope_norm > 1.0e-12
    slope_vx[valid_slope] /= slope_norm[valid_slope]
    slope_vy[valid_slope] /= slope_norm[valid_slope]

    tx = -local_gy
    ty = local_gx
    t_norm = np.hypot(tx, ty)
    valid_t = t_norm > 1.0e-12
    tx[valid_t] /= t_norm[valid_t]
    ty[valid_t] /= t_norm[valid_t]

    energy_norm = energy / max(energy.max(), 1.0e-12)

    # 高エネルギー: 河道寄り（下流輸送・侵食寄り）
    high_count = np.count_nonzero(high_energy_mask)
    if high_count > 0:
        vx[high_energy_mask] = (
            0.55 * slope_vx[high_energy_mask]
            + rng.normal(0.0, 0.05, size=high_count)
        )
        vy[high_energy_mask] = (
            0.55 * slope_vy[high_energy_mask]
            + rng.normal(0.0, 0.05, size=high_count)
        )

    # 低エネルギー: 滞水・拡散寄り（ゆらぎ＋弱い接線方向ドリフト）
    low_count = np.count_nonzero(low_energy_mask)
    if low_count > 0:
        vx[low_energy_mask] = (
            0.10 * slope_vx[low_energy_mask]
            + 0.10 * tx[low_energy_mask]
            + rng.normal(0.0, 0.12, size=low_count)
        )
        vy[low_energy_mask] = (
            0.10 * slope_vy[low_energy_mask]
            + 0.10 * ty[low_energy_mask]
            + rng.normal(0.0, 0.12, size=low_count)
        )

    # 簡易の土砂積載モデル（エネルギー依存）
    if high_count > 0:
        sediment_load[high_energy_mask] += (1.2e-5 * dt) * (0.8 + energy_norm[high_energy_mask])
    if low_count > 0:
        sediment_load[low_energy_mask] -= (0.9e-5 * dt) * (1.2 - energy_norm[low_energy_mask])
    np.maximum(sediment_load, 0.0, out=sediment_load)

    px += vx
    py += vy

    if not np.all(np.isfinite(px)) or not np.all(np.isfinite(py)):
        raise ValueError("Particle coordinates became non-finite")

    np.clip(px, 0.0, ncols - 1.0001, out=px)
    np.clip(py, 0.0, nrows - 1.0001, out=py)

    return {
        'high_energy_ratio': float(np.mean(high_energy_mask)),
        'mean_energy': float(np.mean(energy)),
        'mean_load': float(np.mean(sediment_load)),
        'mean_speed': float(np.mean(np.hypot(vx, vy))),
    }


def generate_initial_terrain(
    nrows,
    ncols,
    dx,
    seed=42,
    relief_ratio=RELIEF_RATIO,
):
    """初期地形を生成（純粋なPerlin地形、高低差はドメイン長×比率）"""
    terrain = np.zeros((nrows, ncols))
    
    for octave in range(5):
        freq = 2 ** octave
        amp = 0.5 ** octave
        for i in range(nrows):
            for j in range(ncols):
                val = pnoise2(i * freq / 80.0, j * freq / 80.0, base=seed + octave)
                terrain[i, j] += val * amp
    
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())

    # 高低差（max-min）をドメイン長に対する比率で決定
    # 例: 100x100, dx=50m, relief_ratio=0.30 -> 5000m * 0.30 = 1500m
    domain_length_m = min(nrows, ncols) * dx
    target_relief_m = domain_length_m * relief_ratio
    terrain *= target_relief_m

    return terrain


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
    relief_ratio=RELIEF_RATIO,
    dt=DT,
    tmax=TMAX,
    seed=42,
    save_interval=50000,
    sea_level=0.0,
    enable_sea_particles=True,
    particle_count=3500,
    particle_seed=123,
    history_interval=50000,
    save_history_images=True,
    history_run_name="space_v2",
):
    """SPACE モジュールを使った地形進化シミュレーション"""
    print("=" * 60)
    print("SPACE 地形進化シミュレーション")
    print("=" * 60)
    print(f"Grid: {nrows} x {ncols}, dx = {dx} m")
    print(f"Domain: {nrows * dx / 1000:.1f} km x {ncols * dx / 1000:.1f} km")
    print(f"Uplift rate: {uplift_rate * 1000:.1f} mm/yr")
    print(f"K_br (bedrock): {k_br:.1e}")
    print(f"K_sed (sediment): {k_sed:.1e} ({k_sed/k_br:.1f}x bedrock)")
    print(f"Initial relief ratio: {relief_ratio*100:.1f}% of domain length")
    print(f"Sediment porosity (phi): {phi}")
    print(f"Fine fraction (F_f): {f_f}")
    print(f"Simulation time: {tmax/1000:.0f} kyr")
    print(f"Sea level: {sea_level:.1f} m")
    print(f"Sea particles: {'ON' if enable_sea_particles else 'OFF'}")
    print("=" * 60)
    
    # ----- グリッド作成 -----
    mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
    
    # ----- 初期地形 -----
    terrain_2d = generate_initial_terrain(
        nrows,
        ncols,
        dx=dx,
        seed=seed,
        relief_ratio=relief_ratio,
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
        bottom_is_closed=False,
    )

    # ----- コンポーネント初期化 -----
    fa = FlowAccumulator(mg, flow_director='D8')
    ld = LinearDiffuser(mg, linear_diffusivity=k_hs, deposit=False)
    
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

    # ----- history 出力 -----
    history_dir = None
    if save_history_images:
        history_dir = create_history_dir(run_name=history_run_name)
        print(f"History dir: {history_dir}")

    # ----- 海用パーティクル初期化 -----
    particles = None
    particle_rng = None
    if enable_sea_particles:
        px, py, sediment_load, particle_rng = initialize_particles(
            nrows=nrows,
            ncols=ncols,
            particle_count=particle_count,
            seed=particle_seed,
        )
        particles = {
            'x': px,
            'y': py,
            'sediment_load': sediment_load,
            'stats_last': None,
        }

        if save_history_images and history_dir is not None:
            save_history_frame(
                z.reshape((nrows, ncols)),
                sea_level,
                history_dir / "terrain_0000kyr.png",
                "Terrain @ 0 kyr",
                particle_xy=(px, py),
                drainage=None,  # 初期時点では流路未計算
            )
    
    # ----- シミュレーションループ -----
    total_time = 0
    t_values = np.arange(0, tmax, dt)
    
    print(f"\nSimulation starting...")
    
    for ti in t_values:
        # 1. 地殻隆起
        z[mg.core_nodes] += uplift_rate * dt
        
        # 2. 斜面拡散
        ld.run_one_step(dt)
        
        # 3. 流路計算
        fa.run_one_step()
        
        # 4. 窪地処理
        df.map_depressions()
        
        # 5. SPACE（侵食＋堆積）
        space.run_one_step(dt)

        # 6. 海用パーティクル（標高0以下=海中）
        if enable_sea_particles:
            if particles is None or particle_rng is None:
                raise RuntimeError("Sea particle mode is enabled but particles are not initialized")
            drainage_now = mg.at_node["drainage_area"].reshape((nrows, ncols))
            particle_stats = update_sea_particles(
                z.reshape((nrows, ncols)),
                particles['x'],
                particles['y'],
                particles['sediment_load'],
                particle_rng,
                drainage_2d=drainage_now,
                dx=dx,
                dt=dt,
            )
            particles['stats_last'] = particle_stats
        
        total_time += dt
        
        # 進捗表示
        if total_time % (tmax // 10) == 0:
            soil_mean = soil[mg.core_nodes].mean()
            if enable_sea_particles:
                if particles is None:
                    raise RuntimeError("Sea particle mode is enabled but particle container is missing")
                stats_last = particles['stats_last']
            else:
                stats_last = None

            if stats_last is not None:
                print(
                    f"  {total_time/1000:.0f} kyr... "
                    f"(mean soil: {soil_mean:.2f} m, "
                    f"high-E ratio: {stats_last['high_energy_ratio']:.2f}, "
                    f"mean E: {stats_last['mean_energy']:.4f}, "
                    f"particle speed: {stats_last['mean_speed']:.2f})"
                )
            else:
                print(f"  {total_time/1000:.0f} kyr... (mean soil depth: {soil_mean:.2f} m)")
        
        # スナップショット
        if total_time % save_interval == 0:
            z_now = z.copy().reshape((nrows, ncols))
            sea_ratio_now = float(np.mean(z_now <= sea_level))
            snapshots.append({
                'time': total_time,
                'elevation': z_now,
                'soil_depth': soil.copy().reshape((nrows, ncols)),
                'drainage': mg.at_node["drainage_area"].copy().reshape((nrows, ncols)),
                'sea_ratio': sea_ratio_now,
            })

        if save_history_images and history_dir is not None and (total_time % history_interval == 0):
            z_now = z.reshape((nrows, ncols))
            sea_ratio_now = float(np.mean(z_now <= sea_level))
            drainage_now = mg.at_node["drainage_area"].copy().reshape((nrows, ncols))
            particle_xy = None
            if enable_sea_particles:
                if particles is None:
                    raise RuntimeError("Sea particle mode is enabled but particle container is missing")
                particle_xy = (particles['x'], particles['y'])
            save_history_frame(
                z_now,
                sea_level,
                history_dir / f"terrain_{int(total_time/1000):04d}kyr.png",
                f"Terrain @ {total_time/1000:.0f} kyr | sea<=0m: {sea_ratio_now*100:.1f}%",
                particle_xy=particle_xy,
                drainage=drainage_now,
            )
            print(
                f"    [history] {total_time/1000:.0f} kyr: "
                f"sea<=0m {sea_ratio_now*100:.1f}% "
                f"-> terrain_{int(total_time/1000):04d}kyr.png"
            )
    
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
        'particles': particles,
        'history_dir': history_dir,
        'sea_level': sea_level,
        'params': {
            'nrows': nrows, 'ncols': ncols, 'dx': dx,
            'uplift_rate': uplift_rate, 'k_br': k_br, 'k_sed': k_sed,
            'k_hs': k_hs, 'phi': phi, 'f_f': f_f, 'relief_ratio': relief_ratio,
            'tmax': tmax, 'dt': dt,
            'enable_sea_particles': enable_sea_particles,
            'particle_count': particle_count,
            'history_interval': history_interval,
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
    rgb = ls.shade(z_initial, cmap=plt.get_cmap('terrain'), vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    ax.set_title('Initial Terrain')
    ax.axis('off')
    
    # ----- 2. Final terrain -----
    ax = axes[0, 1]
    rgb = ls.shade(z_final, cmap=plt.get_cmap('terrain'), vert_exag=2, blend_mode='overlay')
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
    rgb = ls.shade(z_final, cmap=plt.get_cmap('terrain'), vert_exag=2, blend_mode='overlay')
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
    im = None
    
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
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    if im is not None:
        fig.colorbar(im, cax=cbar_ax, label='Sediment Depth [m]')
    
    plt.tight_layout(rect=(0, 0, 0.9, 1))
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
    
    # Case 1: 堆積あり (F_f = 0)
    print("\n[Case 1] F_f = 0 (full deposition)")
    results_dep = run_space_simulation(
        nrows=120,
        ncols=120,
        dx=50.0,
        tmax=200000,  # 20万年
        seed=42,
        save_interval=50000,
        f_f=0.0,
        enable_sea_particles=False,
        save_history_images=False,
    )
    
    # Case 2: 堆積なし (F_f = 1)
    print("\n[Case 2] F_f = 1 (no deposition, like StreamPower)")
    results_nodep = run_space_simulation(
        nrows=120,
        ncols=120,
        dx=50.0,
        tmax=200000,  # 20万年
        seed=42,
        save_interval=50000,
        f_f=1.0,
        enable_sea_particles=False,
        save_history_images=False,
    )
    
    # 比較可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ls = LightSource(azdeg=315, altdeg=45)
    
    sea_level = 0.0
    
    # Row 1: With deposition
    ax = axes[0, 0]
    rgb = ls.shade(results_dep['z_final'], cmap=plt.get_cmap('terrain'), vert_exag=2, blend_mode='overlay')
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
    rgb = ls.shade(results_nodep['z_final'], cmap=plt.get_cmap('terrain'), vert_exag=2, blend_mode='overlay')
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
    print("SPACE Terrain Evolution Simulation")
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
        relief_ratio=0.30,
        dt=1000,
        tmax=300000,   # 30万年
        seed=42,
        save_interval=50000,
        sea_level=0.0,
        enable_sea_particles=True,
        particle_count=3500,
        particle_seed=123,
        history_interval=10000,
        save_history_images=True,
        history_run_name="space_v2_main",
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
    if results.get('history_dir') is not None:
        print(f"  - {results['history_dir']} : 途中経過フレーム")
    print("="*60)
