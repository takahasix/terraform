"""
シミュレーション結果をゲーム用タイルマップに変換

4種類のタイル:
- 水 (water): 河川・湖
- 砂 (sand): 河岸・堆積地・海岸
- 草 (grass): 緩斜面・平地
- 土 (dirt): 急斜面・岩場

分類の物理的根拠:
1. 水: 集水面積が大きい場所 = 水が集まる = 河川
2. 砂: 水の近く + 堆積物が厚い = 河岸の砂地、氾濫原
3. 草: 傾斜が緩やか + 土壌がある = 植生が育つ
4. 土: 傾斜が急 or 堆積物が薄い = 岩が露出、植生が乏しい
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import ListedColormap
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline


OUTPUT_DIR = Path(__file__).resolve().parent / "archive" / "trial_images"


def resolve_output_path(filename):
    """出力先を archive/trial_images に固定"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / Path(filename).name


def calculate_slope(elevation, dx=1.0):
    """傾斜を計算（度）"""
    gy, gx = np.gradient(elevation, dx)
    slope = np.sqrt(gx**2 + gy**2)
    slope_degrees = np.degrees(np.arctan(slope))
    return slope_degrees


def distance_from_water(water_mask):
    """水からの距離を計算（セル単位）"""
    # 水でないところからの距離
    distance = ndimage.distance_transform_edt(~water_mask)
    return distance


def classify_terrain(elevation, drainage, soil_depth, dx=50.0,
                     river_percentile=95,      # 河川判定の閾値（集水面積の上位%）
                     sand_distance=3,          # 水からの距離（セル）
                     sand_soil_threshold=1.0,  # 砂と判定する堆積物の厚さ
                     grass_slope_threshold=15, # 草と判定する傾斜（度）
                     dirt_slope_threshold=25): # 土（急斜面）の傾斜（度）
    """
    地形を4種類に分類
    
    Returns:
        tile_map: 0=water, 1=sand, 2=grass, 3=dirt
    """
    nrows, ncols = elevation.shape
    tile_map = np.zeros((nrows, ncols), dtype=int)
    
    # 1. 傾斜を計算
    slope = calculate_slope(elevation, dx)
    
    # 2. 水（河川）の判定
    river_threshold = np.percentile(drainage, river_percentile)
    water_mask = drainage > river_threshold
    tile_map[water_mask] = 0  # water
    
    # 3. 水からの距離
    dist_from_water = distance_from_water(water_mask)
    
    # 4. 砂の判定（水の近く OR 堆積物が厚い低地）
    near_water = (dist_from_water > 0) & (dist_from_water <= sand_distance)
    thick_sediment = soil_depth > sand_soil_threshold
    low_slope = slope < grass_slope_threshold
    sand_mask = (near_water | (thick_sediment & low_slope)) & ~water_mask
    tile_map[sand_mask] = 1  # sand
    
    # 5. 残りを傾斜で分類
    remaining = ~water_mask & ~sand_mask
    
    # 草: 緩斜面
    grass_mask = remaining & (slope < grass_slope_threshold)
    tile_map[grass_mask] = 2  # grass
    
    # 土: 急斜面
    dirt_mask = remaining & (slope >= grass_slope_threshold)
    tile_map[dirt_mask] = 3  # dirt
    
    return tile_map, {
        'slope': slope,
        'water_mask': water_mask,
        'dist_from_water': dist_from_water,
    }


def upscale_terrain(elevation, scale=4, method='cubic'):
    """
    地形を拡大（アップスケール）
    
    method:
        'nearest': 最近傍（ピクセル感を残す）
        'linear': 線形補間
        'cubic': 3次スプライン（滑らか）
    """
    nrows, ncols = elevation.shape
    
    # 元の座標
    x = np.arange(ncols)
    y = np.arange(nrows)
    
    # 新しい座標
    new_x = np.linspace(0, ncols - 1, ncols * scale)
    new_y = np.linspace(0, nrows - 1, nrows * scale)
    
    if method == 'nearest':
        # 最近傍補間
        from scipy.ndimage import zoom
        return zoom(elevation, scale, order=0)
    elif method == 'linear':
        from scipy.ndimage import zoom
        return zoom(elevation, scale, order=1)
    else:  # cubic
        # 3次スプライン補間
        spline = RectBivariateSpline(y, x, elevation, kx=3, ky=3)
        return spline(new_y, new_x)


def extract_and_process_region(elevation, drainage, soil_depth, 
                                region=(50, 100, 50, 100),  # (row_start, row_end, col_start, col_end)
                                scale=4,
                                dx=50.0):
    """
    領域を切り出して拡大・分類
    """
    r0, r1, c0, c1 = region
    
    # 切り出し
    elev_crop = elevation[r0:r1, c0:c1]
    drain_crop = drainage[r0:r1, c0:c1]
    soil_crop = soil_depth[r0:r1, c0:c1]
    
    # 拡大
    elev_up = upscale_terrain(elev_crop, scale, method='cubic')
    drain_up = upscale_terrain(drain_crop, scale, method='cubic')
    soil_up = upscale_terrain(soil_crop, scale, method='cubic')
    
    # 分類
    tile_map, debug_info = classify_terrain(
        elev_up, drain_up, soil_up, 
        dx=dx/scale  # 拡大後のセルサイズ
    )
    
    return {
        'elevation': elev_up,
        'drainage': drain_up,
        'soil_depth': soil_up,
        'tile_map': tile_map,
        'debug': debug_info,
        'original': {
            'elevation': elev_crop,
            'drainage': drain_crop,
            'soil_depth': soil_crop,
        }
    }


def visualize_tile_map(result, output_file='tile_map.png'):
    """タイルマップを可視化"""
    tile_map = result['tile_map']
    elevation = result['elevation']
    
    # カラーマップ: 水=青, 砂=黄, 草=緑, 土=茶
    colors = ['#3498db', '#f4d03f', '#27ae60', '#8b4513']
    cmap = ListedColormap(colors)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 元の地形（切り出し前のスケール）
    ax = axes[0, 0]
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(result['original']['elevation'], cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    ax.set_title('Original Region')
    ax.axis('off')
    
    # 2. 拡大後の地形
    ax = axes[0, 1]
    rgb = ls.shade(elevation, cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    ax.set_title('Upscaled Terrain')
    ax.axis('off')
    
    # 3. 傾斜
    ax = axes[0, 2]
    im = ax.imshow(result['debug']['slope'], cmap='Reds', origin='lower', vmin=0, vmax=45)
    ax.set_title('Slope (degrees)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    # 4. 堆積物
    ax = axes[1, 0]
    im = ax.imshow(result['soil_depth'], cmap='YlOrBr', origin='lower')
    ax.set_title('Sediment Depth')
    plt.colorbar(im, ax=ax, label='[m]')
    ax.axis('off')
    
    # 5. 水からの距離
    ax = axes[1, 1]
    im = ax.imshow(result['debug']['dist_from_water'], cmap='Blues_r', origin='lower')
    ax.set_title('Distance from Water')
    plt.colorbar(im, ax=ax, label='[cells]')
    ax.axis('off')
    
    # 6. タイルマップ
    ax = axes[1, 2]
    im = ax.imshow(tile_map, cmap=cmap, origin='lower', vmin=0, vmax=3)
    ax.set_title('Tile Map\n(blue=water, yellow=sand, green=grass, brown=dirt)')
    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], label='Water'),
        Patch(facecolor=colors[1], label='Sand'),
        Patch(facecolor=colors[2], label='Grass'),
        Patch(facecolor=colors[3], label='Dirt'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax.axis('off')
    
    plt.tight_layout()
    save_path = resolve_output_path(output_file)
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    
    # タイル統計
    unique, counts = np.unique(tile_map, return_counts=True)
    total = tile_map.size
    print("\nTile Statistics:")
    names = ['Water', 'Sand', 'Grass', 'Dirt']
    for u, c in zip(unique, counts):
        print(f"  {names[u]}: {c} cells ({100*c/total:.1f}%)")
    
    return fig


def export_tile_map(tile_map, filename='tile_map.csv'):
    """タイルマップをCSVでエクスポート（ゲームエンジン用）"""
    save_path = resolve_output_path(filename)
    np.savetxt(save_path, tile_map, fmt='%d', delimiter=',')
    print(f"Exported: {save_path}")


def export_tile_map_image(tile_map, filename='tile_map_raw.png'):
    """タイルマップを画像でエクスポート（各ピクセル = 1タイル）"""
    # RGBA画像として保存
    colors = np.array([
        [52, 152, 219, 255],   # water: blue
        [244, 208, 63, 255],   # sand: yellow
        [39, 174, 96, 255],    # grass: green
        [139, 69, 19, 255],    # dirt: brown
    ], dtype=np.uint8)
    
    img = colors[tile_map]
    
    from PIL import Image
    im = Image.fromarray(img)
    save_path = resolve_output_path(filename)
    im.save(save_path)
    print(f"Exported: {save_path}")


# ========================================
# メイン実行
# ========================================
if __name__ == "__main__":
    import os
    
    # 最新のスナップショットを読み込む（あれば）
    # なければシミュレーションを実行
    
    print("=" * 60)
    print("Terrain to Tile Map Converter")
    print("=" * 60)
    
    # test_pure_perlin.py の結果を使う
    # まずシミュレーションを実行して結果を取得
    
    try:
        from test_pure_perlin import generate_pure_perlin, NROWS, NCOLS, DX
    except ModuleNotFoundError:
        from archive.old_scripts.test_pure_perlin import generate_pure_perlin, NROWS, NCOLS, DX
    from landlab import RasterModelGrid
    from landlab.components import FlowAccumulator, LinearDiffuser, SpaceLargeScaleEroder
    
    print("\nRunning quick simulation for demo...")
    
    # 短いシミュレーション（デモ用）
    mg = RasterModelGrid(shape=(NROWS, NCOLS), xy_spacing=DX)
    terrain = generate_pure_perlin(NROWS, NCOLS, seed=42, amplitude=300.0)
    z = mg.add_zeros("topographic__elevation", at="node")
    z += terrain.ravel()
    
    soil = mg.add_zeros("soil__depth", at="node")
    soil += 0.5
    
    mg.set_closed_boundaries_at_grid_edges(True, True, True, True)
    bottom_nodes = mg.nodes_at_bottom_edge
    mg.status_at_node[bottom_nodes] = mg.BC_NODE_IS_FIXED_VALUE
    z[bottom_nodes] = terrain.min()
    
    fa = FlowAccumulator(mg, flow_director='D8')
    ld = LinearDiffuser(mg, linear_diffusivity=0.05, deposit=False)
    space = SpaceLargeScaleEroder(mg, K_sed=3.0e-5, K_br=1.0e-5, F_f=0.0, 
                                   phi=0.3, H_star=0.5, v_s=1.0, m_sp=0.5, n_sp=1.0)
    
    # 50万年シミュレーション
    TMAX_DEMO = 500000
    DT = 1000
    for ti in range(0, TMAX_DEMO, DT):
        z[mg.core_nodes] += 0.001 * DT
        ld.run_one_step(DT)
        fa.run_one_step()
        space.run_one_step(DT)
        if (ti + DT) % 100000 == 0:
            print(f"  {(ti+DT)/1000:.0f} kyr...")
    
    elevation = z.reshape((NROWS, NCOLS))
    drainage = mg.at_node["drainage_area"].reshape((NROWS, NCOLS))
    soil_depth = soil.reshape((NROWS, NCOLS))
    
    print(f"\nSimulation complete.")
    print(f"Elevation range: {elevation.min():.1f} - {elevation.max():.1f} m")
    
    # ========================================
    # 領域を切り出して処理
    # ========================================
    print("\n" + "=" * 60)
    print("Extracting and processing region...")
    print("=" * 60)
    
    # 中央付近を切り出し（50x50 セル → 4倍拡大で 200x200）
    region = (50, 100, 50, 100)  # 中央50x50
    scale = 4
    
    result = extract_and_process_region(
        elevation, drainage, soil_depth,
        region=region,
        scale=scale,
        dx=DX
    )
    
    # 分類パラメータを調整して再分類
    tile_map, debug_info = classify_terrain(
        result['elevation'], 
        result['drainage'], 
        result['soil_depth'],
        dx=DX/scale,
        river_percentile=94,       # 97 → 94 に減少（水を約2倍に）
        sand_distance=6,           # 3 → 6 に増加（砂浜を広く）
        sand_soil_threshold=0.3,   # 1.0 → 0.3 に減少（薄い堆積でも砂）
        grass_slope_threshold=15,
    )
    result['tile_map'] = tile_map
    result['debug'] = debug_info
    
    print(f"Original region: {region[1]-region[0]} x {region[3]-region[2]} cells")
    print(f"Upscaled: {result['tile_map'].shape[0]} x {result['tile_map'].shape[1]} cells")
    
    # 可視化
    visualize_tile_map(result, 'tile_map_demo.png')
    
    # エクスポート
    export_tile_map(result['tile_map'], 'tile_map.csv')
    
    try:
        export_tile_map_image(result['tile_map'], 'tile_map_raw.png')
    except ImportError:
        print("PIL not installed, skipping image export")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
