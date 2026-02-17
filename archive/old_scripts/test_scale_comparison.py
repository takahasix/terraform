"""
セルサイズの違いによる結果の比較

DX=50m (7.5km x 7.5km) vs DX=5m (750m x 750m)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from noise import pnoise2
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator, LinearDiffuser, SpaceLargeScaleEroder


def generate_pure_perlin(nrows, ncols, seed=42, amplitude=300.0, scale=80.0):
    """純粋なパーリンノイズ"""
    terrain = np.zeros((nrows, ncols))
    
    for octave in range(5):
        freq = 2 ** octave
        amp = 0.5 ** octave
        for i in range(nrows):
            for j in range(ncols):
                val = pnoise2(i * freq / scale, j * freq / scale, base=seed + octave)
                terrain[i, j] += val * amp
    
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    terrain *= amplitude
    return terrain


def run_simulation(nrows, ncols, dx, amplitude, tmax, dt, label):
    """シミュレーション実行"""
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"Grid: {nrows}x{ncols}, DX={dx}m")
    print(f"Area: {nrows*dx/1000:.1f}km x {ncols*dx/1000:.1f}km")
    print(f"{'='*60}")
    
    mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
    
    # パーリンノイズのスケールも調整
    # 地形の「波長」を物理的に同じにするため
    perlin_scale = 80.0 * (dx / 50.0)  # DX=50mで80、DX=5mで8
    terrain = generate_pure_perlin(nrows, ncols, seed=42, amplitude=amplitude, scale=perlin_scale)
    
    z = mg.add_zeros("topographic__elevation", at="node")
    z += terrain.ravel()
    
    soil = mg.add_zeros("soil__depth", at="node")
    soil += 0.5
    
    mg.set_closed_boundaries_at_grid_edges(True, True, True, True)
    bottom_nodes = mg.nodes_at_bottom_edge
    mg.status_at_node[bottom_nodes] = mg.BC_NODE_IS_FIXED_VALUE
    z[bottom_nodes] = terrain.min()
    
    print(f"Initial elevation: {terrain.min():.1f} - {terrain.max():.1f} m")
    
    fa = FlowAccumulator(mg, flow_director='D8')
    ld = LinearDiffuser(mg, linear_diffusivity=0.05, deposit=False)
    
    # K値はスケールに依存する可能性があるが、まずは同じ値で試す
    space = SpaceLargeScaleEroder(
        mg, K_sed=3.0e-5, K_br=1.0e-5, F_f=0.0,
        phi=0.3, H_star=0.5, v_s=1.0, m_sp=0.5, n_sp=1.0
    )
    
    print("Running simulation...")
    for ti in range(0, tmax, dt):
        z[mg.core_nodes] += 0.001 * dt  # 1mm/yr uplift
        ld.run_one_step(dt)
        fa.run_one_step()
        space.run_one_step(dt)
        if (ti + dt) % 100000 == 0:
            print(f"  {(ti+dt)/1000:.0f} kyr...")
    
    elevation = z.reshape((nrows, ncols))
    drainage = mg.at_node["drainage_area"].reshape((nrows, ncols))
    soil_depth = soil.reshape((nrows, ncols))
    
    print(f"Final elevation: {elevation.min():.1f} - {elevation.max():.1f} m")
    
    return {
        'elevation': elevation,
        'drainage': drainage,
        'soil_depth': soil_depth,
        'dx': dx,
        'label': label,
        'area_km': (nrows * dx / 1000, ncols * dx / 1000),
    }


def visualize_comparison(results):
    """比較可視化"""
    fig, axes = plt.subplots(2, len(results), figsize=(6*len(results), 10))
    ls = LightSource(azdeg=315, altdeg=45)
    
    for i, res in enumerate(results):
        # 地形
        ax = axes[0, i]
        rgb = ls.shade(res['elevation'], cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
        ax.imshow(rgb, origin='lower')
        ax.set_title(f"{res['label']}\n{res['area_km'][0]:.1f}km x {res['area_km'][1]:.1f}km\n(DX={res['dx']}m)")
        ax.axis('off')
        
        # 排水パターン
        ax = axes[1, i]
        elev = res['elevation']
        drain = res['drainage']
        
        ax.imshow(elev, cmap='terrain', origin='lower', alpha=0.5)
        
        # 河川を集水面積の閾値で表示
        river_threshold = np.percentile(drain, 95)
        rivers = np.where(drain > river_threshold, drain, np.nan)
        ax.imshow(rivers, cmap='Blues', origin='lower', alpha=0.8)
        
        ax.set_title(f"Drainage Pattern\n(rivers = top 5% drainage)")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('scale_comparison.png', dpi=150)
    print(f"\nSaved: scale_comparison.png")
    return fig


if __name__ == "__main__":
    # シミュレーション設定
    TMAX = 500000  # 50万年
    DT = 1000
    
    results = []
    
    # ケース1: DX=50m (標準)
    # 150x150 = 7.5km x 7.5km
    res1 = run_simulation(
        nrows=150, ncols=150, dx=50.0,
        amplitude=300.0,  # 300m の起伏
        tmax=TMAX, dt=DT,
        label="Case 1: DX=50m"
    )
    results.append(res1)
    
    # ケース2: DX=5m (10倍細かい)
    # 150x150 = 750m x 750m (小さいエリア)
    res2 = run_simulation(
        nrows=150, ncols=150, dx=5.0,
        amplitude=30.0,   # 30m の起伏（スケールに合わせて縮小）
        tmax=TMAX, dt=DT,
        label="Case 2: DX=5m"
    )
    results.append(res2)
    
    # ケース3: DX=5m で同じ7.5kmエリア
    # 1500x1500 は重いので 300x300 (1.5km x 1.5km) で試す
    print("\n※ ケース3は計算が重いので 300x300 (1.5km) で実行")
    res3 = run_simulation(
        nrows=300, ncols=300, dx=5.0,
        amplitude=60.0,   # 60m の起伏
        tmax=TMAX, dt=DT,
        label="Case 3: DX=5m (larger)"
    )
    results.append(res3)
    
    # 可視化
    visualize_comparison(results)
    
    # 統計比較
    print("\n" + "=" * 60)
    print("Statistics Comparison")
    print("=" * 60)
    print(f"{'Case':<25} {'Area (km²)':<12} {'Elev Range':<15} {'Drainage 95%':<15}")
    print("-" * 70)
    for res in results:
        area = res['area_km'][0] * res['area_km'][1]
        elev_range = f"{res['elevation'].min():.0f}-{res['elevation'].max():.0f}m"
        drain_95 = np.percentile(res['drainage'], 95)
        print(f"{res['label']:<25} {area:<12.2f} {elev_range:<15} {drain_95:<15.0f}")
    
    plt.show()
