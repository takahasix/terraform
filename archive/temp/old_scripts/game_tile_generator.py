"""
ゲーム用タイルマップ生成（1タイル = 2m）

地形シム: 20x20セル (DX=50m) = 1km x 1km
    ↓ 25倍拡大
ゲーム: 500x500タイル (1タイル=2m) = 1km x 1km
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LightSource
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline


def calculate_slope(elevation, dx=1.0):
    """傾斜を計算（度）"""
    gy, gx = np.gradient(elevation, dx)
    slope = np.sqrt(gx**2 + gy**2)
    slope_degrees = np.degrees(np.arctan(slope))
    return slope_degrees


def distance_from_water(water_mask):
    """水からの距離を計算（セル単位）"""
    distance = ndimage.distance_transform_edt(~water_mask)
    return distance


def classify_terrain(elevation, drainage, soil_depth, dx=2.0,
                     river_percentile=94,
                     sand_distance=6,
                     sand_soil_threshold=0.3,
                     grass_slope_threshold=15):
    """地形を4種類に分類 (0=water, 1=sand, 2=grass, 3=dirt)"""
    nrows, ncols = elevation.shape
    tile_map = np.zeros((nrows, ncols), dtype=int)
    
    slope = calculate_slope(elevation, dx)
    
    # 水（河川）
    river_threshold = np.percentile(drainage, river_percentile)
    water_mask = drainage > river_threshold
    tile_map[water_mask] = 0
    
    # 水からの距離
    dist_from_water = distance_from_water(water_mask)
    
    # 砂
    near_water = (dist_from_water > 0) & (dist_from_water <= sand_distance)
    thick_sediment = soil_depth > sand_soil_threshold
    low_slope = slope < grass_slope_threshold
    sand_mask = (near_water | (thick_sediment & low_slope)) & ~water_mask
    tile_map[sand_mask] = 1
    
    # 残りを傾斜で分類
    remaining = ~water_mask & ~sand_mask
    grass_mask = remaining & (slope < grass_slope_threshold)
    tile_map[grass_mask] = 2
    dirt_mask = remaining & (slope >= grass_slope_threshold)
    tile_map[dirt_mask] = 3
    
    return tile_map, {'slope': slope, 'water_mask': water_mask, 'dist_from_water': dist_from_water}


def upscale_terrain(data, target_size, method='cubic'):
    """地形を指定サイズに拡大"""
    nrows, ncols = data.shape
    x = np.arange(ncols)
    y = np.arange(nrows)
    new_x = np.linspace(0, ncols - 1, target_size)
    new_y = np.linspace(0, nrows - 1, target_size)
    
    if method == 'cubic':
        spline = RectBivariateSpline(y, x, data, kx=3, ky=3)
        return spline(new_y, new_x)
    else:
        from scipy.ndimage import zoom
        scale = target_size / nrows
        return zoom(data, scale, order=1 if method == 'linear' else 0)


def find_interesting_region(elevation, drainage, size=20):
    """川がありそうな興味深い領域を探す"""
    nrows, ncols = elevation.shape
    
    best_score = -1
    best_region = (0, size, 0, size)
    
    # スライディングウィンドウで探索
    for r in range(0, nrows - size, 5):
        for c in range(0, ncols - size, 5):
            region_drain = drainage[r:r+size, c:c+size]
            region_elev = elevation[r:r+size, c:c+size]
            
            # スコア: 高い集水面積 + 標高差（起伏）があると良い
            drain_score = np.percentile(region_drain, 95)
            elev_range = region_elev.max() - region_elev.min()
            
            score = drain_score * elev_range
            
            if score > best_score:
                best_score = score
                best_region = (r, r+size, c, c+size)
    
    return best_region


def visualize_with_region(elevation, drainage, region, output_file='region_selection.png'):
    """領域選択を可視化"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ls = LightSource(azdeg=315, altdeg=45)
    
    # 地形全体
    ax = axes[0]
    rgb = ls.shade(elevation, cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    
    # 選択領域を赤枠で表示
    r0, r1, c0, c1 = region
    rect = plt.Rectangle((c0, r0), c1-c0, r1-r0, 
                          fill=False, edgecolor='red', linewidth=3)
    ax.add_patch(rect)
    ax.set_title(f'Full Terrain (150x150, 7.5km x 7.5km)\nSelected: {r1-r0}x{c1-c0} cells = {(r1-r0)*50}m x {(c1-c0)*50}m')
    ax.axis('off')
    
    # 集水面積
    ax = axes[1]
    ax.imshow(elevation, cmap='terrain', origin='lower', alpha=0.5)
    drain_vis = np.log10(drainage + 1)
    ax.imshow(drain_vis, cmap='Blues', origin='lower', alpha=0.6)
    rect = plt.Rectangle((c0, r0), c1-c0, r1-r0, 
                          fill=False, edgecolor='red', linewidth=3)
    ax.add_patch(rect)
    ax.set_title('Drainage Area (log scale)\nBlue = Rivers')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    return fig


def visualize_game_tiles(result, output_file='game_tiles.png'):
    """ゲーム用タイルマップを可視化"""
    tile_map = result['tile_map']
    elevation = result['elevation']
    
    colors = ['#3498db', '#f4d03f', '#27ae60', '#8b4513']
    cmap = ListedColormap(colors)
    ls = LightSource(azdeg=315, altdeg=45)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 1. 元の領域
    ax = axes[0, 0]
    rgb = ls.shade(result['original']['elevation'], cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    ax.set_title(f"Original Region\n{result['original']['elevation'].shape[0]}x{result['original']['elevation'].shape[1]} cells (DX=50m)")
    ax.axis('off')
    
    # 2. 拡大後の地形
    ax = axes[0, 1]
    rgb = ls.shade(elevation, cmap=plt.cm.terrain, vert_exag=2, blend_mode='overlay')
    ax.imshow(rgb, origin='lower')
    ax.set_title(f"Upscaled Terrain\n{elevation.shape[0]}x{elevation.shape[1]} tiles (1 tile = 2m)")
    ax.axis('off')
    
    # 3. 傾斜 + 集水
    ax = axes[1, 0]
    ax.imshow(result['debug']['slope'], cmap='Reds', origin='lower', vmin=0, vmax=45, alpha=0.7)
    water = np.where(result['debug']['water_mask'], 1, np.nan)
    ax.imshow(water, cmap='Blues', origin='lower', vmin=0, vmax=1)
    ax.set_title('Slope (red) + Water (blue)')
    ax.axis('off')
    
    # 4. タイルマップ
    ax = axes[1, 1]
    ax.imshow(tile_map, cmap=cmap, origin='lower', vmin=0, vmax=3)
    ax.set_title('Game Tile Map\n(blue=water, yellow=sand, green=grass, brown=dirt)')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], label='Water'),
        Patch(facecolor=colors[1], label='Sand'),
        Patch(facecolor=colors[2], label='Grass'),
        Patch(facecolor=colors[3], label='Dirt'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    
    # 統計
    unique, counts = np.unique(tile_map, return_counts=True)
    total = tile_map.size
    print("\nTile Statistics:")
    names = ['Water', 'Sand', 'Grass', 'Dirt']
    for u, c in zip(unique, counts):
        print(f"  {names[u]}: {c} cells ({100*c/total:.1f}%)")
    
    return fig


def export_tile_map(tile_map, filename='game_tiles.csv'):
    """CSVエクスポート"""
    np.savetxt(filename, tile_map, fmt='%d', delimiter=',')
    print(f"Exported: {filename}")


def export_tile_map_image(tile_map, filename='game_tiles_raw.png'):
    """PNG画像エクスポート（1ピクセル=1タイル）"""
    colors = np.array([
        [52, 152, 219, 255],   # water
        [244, 208, 63, 255],   # sand
        [39, 174, 96, 255],    # grass
        [139, 69, 19, 255],    # dirt
    ], dtype=np.uint8)
    img = colors[tile_map]
    from PIL import Image
    im = Image.fromarray(img)
    im.save(filename)
    print(f"Exported: {filename}")


# ========================================
if __name__ == "__main__":
    from test_pure_perlin import generate_pure_perlin
    from landlab import RasterModelGrid
    from landlab.components import FlowAccumulator, LinearDiffuser, SpaceLargeScaleEroder
    
    print("=" * 60)
    print("Game Tile Map Generator")
    print("1 tile = 2m, 500x500 tiles = 1km x 1km")
    print("=" * 60)
    
    # パラメータ
    NROWS, NCOLS, DX = 150, 150, 50.0  # 地形シム: 7.5km x 7.5km
    REGION_SIZE = 20  # 切り出しサイズ: 20セル = 1km
    GAME_SIZE = 500   # ゲームタイル: 500x500
    TILE_SIZE = 2.0   # 1タイル = 2m
    
    print(f"\nSimulation grid: {NROWS}x{NCOLS}, DX={DX}m")
    print(f"Region to extract: {REGION_SIZE}x{REGION_SIZE} cells = {REGION_SIZE*DX}m x {REGION_SIZE*DX}m")
    print(f"Game tiles: {GAME_SIZE}x{GAME_SIZE}, 1 tile = {TILE_SIZE}m")
    
    # シミュレーション実行
    print("\nRunning simulation...")
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
    
    TMAX = 500000
    DT = 1000
    for ti in range(0, TMAX, DT):
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
    # 興味深い領域を探す
    # ========================================
    print("\n" + "=" * 60)
    print("Finding interesting region with rivers...")
    print("=" * 60)
    
    region = find_interesting_region(elevation, drainage, size=REGION_SIZE)
    r0, r1, c0, c1 = region
    print(f"Selected region: rows {r0}-{r1}, cols {c0}-{c1}")
    print(f"Physical size: {(r1-r0)*DX}m x {(c1-c0)*DX}m")
    
    # 領域選択を可視化
    visualize_with_region(elevation, drainage, region, 'region_selection.png')
    
    # ========================================
    # 切り出し & 拡大
    # ========================================
    print("\n" + "=" * 60)
    print(f"Extracting and upscaling to {GAME_SIZE}x{GAME_SIZE}...")
    print("=" * 60)
    
    # 切り出し
    elev_crop = elevation[r0:r1, c0:c1]
    drain_crop = drainage[r0:r1, c0:c1]
    soil_crop = soil_depth[r0:r1, c0:c1]
    
    # 拡大
    elev_up = upscale_terrain(elev_crop, GAME_SIZE)
    drain_up = upscale_terrain(drain_crop, GAME_SIZE)
    soil_up = upscale_terrain(soil_crop, GAME_SIZE)
    
    # 分類
    tile_map, debug_info = classify_terrain(
        elev_up, drain_up, soil_up,
        dx=TILE_SIZE,
        river_percentile=94,
        sand_distance=8,
        sand_soil_threshold=0.3,
        grass_slope_threshold=15,
    )
    
    result = {
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
    
    # 可視化
    visualize_game_tiles(result, 'game_tiles.png')
    
    # エクスポート
    export_tile_map(result['tile_map'], 'game_tiles.csv')
    try:
        export_tile_map_image(result['tile_map'], 'game_tiles_raw.png')
    except ImportError:
        print("PIL not installed, skipping image export")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Done!")
    print(f"Game area: {GAME_SIZE * TILE_SIZE}m x {GAME_SIZE * TILE_SIZE}m = 1km x 1km")
    print("=" * 60)
