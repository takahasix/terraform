"""
初期地形に一切手を加えず、純粋なパーリンノイズだけでSPACEを走らせる
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from noise import pnoise2
from landlab import RasterModelGrid
from landlab.components import (
    FlowAccumulator,
    LinearDiffuser,
    SpaceLargeScaleEroder,
    DepressionFinderAndRouter,
)

# パラメータ
NROWS, NCOLS, DX = 150, 150, 50.0
TMAX = 5000000  # 500万年
DT = 1000
SAVE_INTERVAL = 100000  # 10万年ごとに保存

def generate_pure_perlin(nrows, ncols, seed=42, amplitude=300.0):
    """純粋なパーリンノイズのみ（一切の加工なし）"""
    terrain = np.zeros((nrows, ncols))
    
    for octave in range(5):
        freq = 2 ** octave
        amp = 0.5 ** octave
        for i in range(nrows):
            for j in range(ncols):
                val = pnoise2(i * freq / 80.0, j * freq / 80.0, base=seed + octave)
                terrain[i, j] += val * amp
    
    # 正規化して 0 ~ amplitude に
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    terrain *= amplitude
    
    return terrain

if __name__ == "__main__":
    print("=" * 60)
    print("Pure Perlin Noise + SPACE (no terrain modification)")
    print("=" * 60)

    # グリッド作成
    mg = RasterModelGrid(shape=(NROWS, NCOLS), xy_spacing=DX)

    # 純粋なパーリンノイズ
    terrain = generate_pure_perlin(NROWS, NCOLS, seed=42, amplitude=300.0)
    z = mg.add_zeros("topographic__elevation", at="node")
    z += terrain.ravel()
    z_initial = z.copy()

    # 堆積物層
    soil = mg.add_zeros("soil__depth", at="node")
    soil += 0.5

    # 境界条件：南側のみ開放（固定値境界）
    mg.set_closed_boundaries_at_grid_edges(True, True, True, True)
    bottom_nodes = mg.nodes_at_bottom_edge
    mg.status_at_node[bottom_nodes] = mg.BC_NODE_IS_FIXED_VALUE
    # 境界の標高は最低点に合わせる
    z[bottom_nodes] = terrain.min()

    print(f"Initial elevation range: {terrain.min():.1f} - {terrain.max():.1f} m")
    print(f"Boundary (south) fixed at: {terrain.min():.1f} m")

    # コンポーネント
    fa = FlowAccumulator(mg, flow_director='D8')
    ld = LinearDiffuser(mg, linear_diffusivity=0.05, deposit=False)
    space = SpaceLargeScaleEroder(
        mg, K_sed=3.0e-5, K_br=1.0e-5, F_f=0.0, 
        phi=0.3, H_star=0.5, v_s=1.0, m_sp=0.5, n_sp=1.0
    )
    # df = DepressionFinderAndRouter(mg)  # 遅いので無効化

    # シミュレーション
    print("\nRunning simulation...")
    total_time = 0
    snapshots = []  # スナップショット保存用
    
    for ti in np.arange(0, TMAX, DT):
        z[mg.core_nodes] += 0.001 * DT  # 隆起
        ld.run_one_step(DT)
        fa.run_one_step()
        # df.map_depressions()  # 遅いので無効化
        space.run_one_step(DT)
        total_time += DT
        
        # 10万年ごとにスナップショット保存
        if total_time % SAVE_INTERVAL == 0:
            print(f"  {total_time/1000:.0f} kyr... saving snapshot")
            snapshots.append({
                'time': total_time,
                'elevation': z.copy().reshape((NROWS, NCOLS)),
                'soil_depth': soil.copy().reshape((NROWS, NCOLS)),
                'drainage': mg.at_node["drainage_area"].copy().reshape((NROWS, NCOLS)),
            })

    print(f"\nComplete: {total_time/1000:.0f} kyr")

    # 結果
    z_final = z.reshape((NROWS, NCOLS))
    soil_final = soil.reshape((NROWS, NCOLS))
    drainage = mg.at_node["drainage_area"].reshape((NROWS, NCOLS))

    print(f"Final elevation: {z_final.min():.1f} - {z_final.max():.1f} m")
    print(f"Max soil depth: {soil_final.max():.2f} m")

    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ls = LightSource(azdeg=315, altdeg=45)

    # Initial
    ax = axes[0, 0]
    rgb = ls.shade(z_initial.reshape((NROWS, NCOLS)), cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    ax.set_title('Initial (Pure Perlin Noise)')
    ax.axis('off')

    # Final
    ax = axes[0, 1]
    rgb = ls.shade(z_final, cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    ax.set_title(f'Final ({TMAX/1000:.0f} kyr)')
    ax.axis('off')

    # Change
    ax = axes[0, 2]
    z_change = z_final - z_initial.reshape((NROWS, NCOLS))
    vmax = max(abs(z_change.min()), abs(z_change.max()))
    im = ax.imshow(z_change, cmap='RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    ax.set_title('Elevation Change\n(blue=erosion, red=uplift)')
    plt.colorbar(im, ax=ax, label='[m]')
    ax.axis('off')

    # Soil depth
    ax = axes[1, 0]
    im = ax.imshow(soil_final, cmap='YlOrBr', origin='lower')
    ax.set_title('Sediment Depth [m]')
    plt.colorbar(im, ax=ax, label='[m]')
    ax.axis('off')

    # Drainage
    ax = axes[1, 1]
    im = ax.imshow(np.log10(drainage + 1), cmap='Blues', origin='lower')
    ax.set_title('Drainage Area (log10)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    # Terrain + Rivers
    ax = axes[1, 2]
    rgb = ls.shade(z_final, cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    levels = [np.percentile(drainage, p) for p in [90, 95, 98]]
    ax.contour(drainage, levels=levels, colors=['cyan', 'blue', 'darkblue'], linewidths=[0.5, 1, 1], origin='lower')
    ax.set_title('Terrain + Rivers')
    ax.axis('off')

    plt.suptitle('SPACE with Pure Perlin Noise (No Terrain Modification)', fontsize=14)
    plt.tight_layout()
    plt.savefig('space_pure_perlin.png', dpi=150)
    print(f"\nSaved: space_pure_perlin.png")
    
    # 各スナップショットを個別に保存
    print("\nSaving snapshots...")
    for snap in snapshots:
        fig_snap, ax_snap = plt.subplots(1, 2, figsize=(12, 5))
        
        # 地形
        ax = ax_snap[0]
        rgb = ls.shade(snap['elevation'], cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
        ax.imshow(rgb, origin='lower')
        levels = [np.percentile(snap['drainage'], p) for p in [90, 95, 98]]
        ax.contour(snap['drainage'], levels=levels, colors=['cyan', 'blue', 'darkblue'], linewidths=[0.5, 1, 1], origin='lower')
        ax.set_title(f'Terrain + Rivers')
        ax.axis('off')
        
        # 堆積物
        ax = ax_snap[1]
        im = ax.imshow(snap['soil_depth'], cmap='YlOrBr', origin='lower', vmin=0, vmax=50)
        ax.set_title(f'Sediment Depth (max: {snap["soil_depth"].max():.1f} m)')
        plt.colorbar(im, ax=ax, label='[m]')
        ax.axis('off')
        
        time_kyr = snap['time'] / 1000
        time_myr = snap['time'] / 1000000
        plt.suptitle(f'Time: {time_kyr:.0f} kyr ({time_myr:.1f} Myr)', fontsize=14)
        plt.tight_layout()
        
        filename = f'snapshot_{int(time_kyr):05d}kyr.png'
        plt.savefig(filename, dpi=100)
        plt.close(fig_snap)
        print(f"  Saved: {filename}")
    
    print(f"\nAll {len(snapshots)} snapshots saved!")
    plt.show()
