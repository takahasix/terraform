# 簡易村プロジェクト - 地形生成トライアル

## プロジェクト概要
簡易的な村のシミュレーションおよび地形生成のトライアルプロジェクト。
Landlab ライブラリを使用した地形進化シミュレーション。

## 🎯 成功したパラメータ設定（推奨）
※ `land_evolution.py` は `archive/temp/` に移動しました。SPACEアルゴリズムを使用した `land_evolution_space.py` を基本とします。
### ベスト設定（500万年シミュレーション）

```python
from land_evolution import run_land_evolution_simulation, visualize_results

results = run_land_evolution_simulation(
    nrows=500,          # グリッド行数
    ncols=500,          # グリッド列数
    dx=50.0,            # セル解像度 (m) → 25km×25kmの領域
    uplift_rate=0.001,  # 隆起速度 1mm/yr（活発な造山帯）
    k_sp=1.0e-5,        # 河川侵食係数（Hatari Labs推奨）
    k_hs=0.05,          # 斜面拡散係数（標準：滑らかな斜面）
    dt=5000,            # タイムステップ（5000年）
    tmax=5000000,       # 総時間（500万年）
    seed=42,            # 乱数シード
    use_depression_finder=False,  # 窪地処理OFF（高速化）
    save_interval=1000000,        # 100万年ごとにスナップショット
)
visualize_results(results, output_file='land_evolution_500x500_5M.png')
```

**結果：**
- 最終標高: 20.4 - 1180.4 m
- 総隆起量: 5,000 m
- 定常状態（隆起≒侵食）に到達
- 明瞭な河川ネットワーク・分水嶺が形成

### パラメータバリエーション

| 設定名 | K_hs | 結果の特徴                  |
| ------ | ---- | --------------------------- |
| 標準   | 0.05 | 滑らかな斜面、自然な見た目  |
| 急峻   | 0.01 | シャープな稜線、V字谷が顕著 |

## システム設計

### 地形進化の物理モデル

地形進化は以下の3つのプロセスの組み合わせ（現在は `archive/temp/land_evolution.py` または `land_evolution_space.py` で実装されています）:

1. **河川侵食 (Stream Power Erosion)**
   - 公式: `E = K_sp × A^m × S^n`
   - A: 集水面積、S: 勾配
   - 河川が谷を削り込む

2. **斜面拡散 (Hillslope Diffusion)**
   - 公式: `∂z/∂t = K_hs × ∇²z`
   - 土壌クリープ、風化による斜面の滑らかな変化

3. **地殻隆起 (Tectonic Uplift)**
   - 公式: `∂z/∂t = U`
   - 一定速度での地盤上昇

### SPACE アルゴリズム (land_evolution_space.py)

Landlab の SPACE (Stream Power with Alluvium Conservation and Entrainment) モデルを使用。
これは `land_evolution.py` で使用されている単純な StreamPowerEroder とは異なり、土砂の「堆積（積もる）」をシミュレートできます。
- **岩盤 (Bedrock)** と **土砂 (Sediment)** の2層構造で地形を管理
- 沿岸地形や河口デルタ、扇状地などの再現に適している

### パラメータガイドライン（Hatari Labs参考）

| パラメータ     | 記号 | 推奨値       | 説明                   |
| -------------- | ---- | ------------ | ---------------------- |
| 河川侵食係数   | K_sp | 1.0e-5       | 大きいと侵食が速い     |
| 集水面積指数   | m_sp | 0.5          | 通常 0.3-0.6           |
| 勾配指数       | n_sp | 1.0          | 通常 0.7-1.0           |
| 斜面拡散係数   | K_hs | 0.05 m²/yr   | 大きいと斜面が滑らか   |
| 隆起速度       | U    | 0.001 m/yr   | 1mm/yr = 活発な隆起域  |
| タイムステップ | dt   | 5000 yr      | 安定性と速度のバランス |
| 総時間         | tmax | 5,000,000 yr | 定常状態に達するまで   |

### パラメータの意味と調整指針

- **K_hs（斜面拡散）**: 斜面の滑らかさを制御。大きいと丸みを帯び、小さいと急峻
- **K_sp（河川侵食）**: 河川の削り込み速度。大きいと深い谷が形成
- **uplift_rate**: 地盤隆起速度。大きいと高い山、小さいと低い丘陵
- **tmax**: 長いほど地形が「成熟」。定常状態（100万年以上）で河川ネットワーク明瞭

**重要：K_hs×時間 と 時間延長 は似た効果だが異なる**
- K_hs大：斜面だけ滑らか、河川や標高への影響小
- 時間延長：全プロセス（隆起・河川・斜面）が進む

## 🌊 沿岸地形シミュレーション（coastal_evolution.py）

### 概要
内陸版（land_evolution.py）を拡張し、海辺・沿岸の地形進化を表現。

### 追加した機能
1. **海のベースレベル設定**: 海側境界を固定値境界として設定
2. **海面変動シナリオ**: static（固定）/ rising（上昇）/ falling（低下）/ cycle（周期的）
3. **海浜帯の物性差**: 
   - 高拡散係数（波による平滑化の近似）
   - 高侵食係数（未固結堆積物の侵食されやすさ）
4. **可視化**: 海陸境界・海岸線の表示、時間発展アニメーション

### 使い方

```python
from coastal_evolution import run_coastal_evolution, visualize_coastal_results

# シナリオ1: 海面固定（基本形）
results = run_coastal_evolution(
    nrows=150, ncols=150, dx=50.0,
    tmax=200000,  # 20万年
    sea_level_scenario='static',
    land_fraction=0.75,  # 25%が海
    beach_rows=5,         # 海浜帯5行
    beach_k_hs_mult=5.0,  # 海浜帯の拡散係数倍率
    beach_k_sp_mult=3.0,  # 海浜帯の侵食係数倍率
)
visualize_coastal_results(results, "coastal_result.png")

# シナリオ2: 海面上昇
results = run_coastal_evolution(
    sea_level_scenario='rising',
    sea_level_params={'rate': 0.0003, 'base_level': 0.0},  # 0.3 mm/yr
)

# シナリオ3: 周期的海面変動（氷期-間氷期サイクル）
results = run_coastal_evolution(
    tmax=400000,  # 40万年（複数サイクル）
    sea_level_scenario='cycle',
    sea_level_params={'amplitude': 30.0, 'period': 100000},  # 振幅30m, 周期10万年
)
```

### パラメータガイド

| パラメータ         | 説明                                     | 推奨値                      |
| ------------------ | ---------------------------------------- | --------------------------- |
| land_fraction      | 陸地の割合（0.75なら25%が初期海域）      | 0.7-0.85                    |
| beach_rows         | 海浜帯の幅（グリッド行数）               | 3-10                        |
| beach_k_hs_mult    | 海浜帯での拡散係数倍率（波の平滑化効果） | 3.0-10.0                    |
| beach_k_sp_mult    | 海浜帯での侵食係数倍率（未固結堆積物）   | 2.0-5.0                     |
| sea_level_scenario | 海面変動シナリオ                         | static/rising/falling/cycle |

### 今後の拡張案
- ~~**SPACE モジュール**: 侵食だけでなく堆積も扱う（河口三角州、砂州）~~ → **実装済み**
- **波食崖（wave-cut cliff）**: 海面付近で侵食を強化
- **沿岸漂砂**: 沿岸流による横方向の砂輸送（ルールベースで近似）

## 🏔️ SPACE モジュール（land_evolution_space.py）

### 概要
StreamPowerEroder の代わりに **SPACE (Stream Power with Alluvium Conservation and Entrainment)** を使用。
侵食だけでなく **堆積** も扱えるため、河口三角州・扇状地・砂州などの堆積地形が表現可能。

### StreamPowerEroder vs SPACE

| 特徴           | StreamPower | SPACE |
| -------------- | ----------- | ----- |
| 侵食           | ○           | ○     |
| 堆積           | ×           | ○     |
| 堆積物層の追跡 | ×           | ○     |
| 河口三角州     | △（弱い）   | ○     |
| 計算コスト     | 低          | 中    |

### SPACE パラメータ

| パラメータ | 説明                             | 推奨値        |
| ---------- | -------------------------------- | ------------- |
| K_sed      | 堆積物の侵食係数                 | K_br の 2-5倍 |
| K_br       | 基盤岩の侵食係数                 | 1.0e-5        |
| F_f        | 細粒分割合（0=全堆積、1=全流出） | 0.0-0.5       |
| phi        | 堆積物の間隙率                   | 0.3           |
| H_star     | 土壌生産スケール                 | 0.5 m         |
| v_s        | 沈降速度                         | 1.0 m/yr      |

### 使い方

```python
from land_evolution_space import run_space_simulation, visualize_space_results

results = run_space_simulation(
    nrows=150, ncols=150, dx=50.0,
    k_br=1.0e-5,      # 基盤岩の侵食係数
    k_sed=3.0e-5,     # 堆積物の侵食係数（3倍）
    f_f=0.0,          # 全堆積（0=堆積、1=流出）
    phi=0.3,          # 間隙率
    tmax=300000,      # 30万年
)
visualize_space_results(results, "space_result.png")
```

### 実験結果

**F_f（細粒分割合）の効果:**
- F_f = 0.0（全堆積）: 最大堆積厚 39m、河口付近に堆積物が蓄積
- F_f = 1.0（全流出）: 最大堆積厚 1m、StreamPowerと同様の挙動

## 修正履歴
- 2026/01/31: プロジェクト初期化 (Git init, .gitignore, .venv)
- 2026/01/31: land_evolution.py 作成（Hatari Labs参考版）
- 2026/02/01: パラメータ実験完了、500万年シミュレーションで定常状態を確認
- 2026/02/01: ファイル整理（過去の試行をarchiveフォルダへ移動)
- 2026/02/01: **coastal_evolution.py 作成**（海辺・沿岸地形の表現）
- 2026/02/01: **space_coastal.py 作成**（SPACE モジュールで侵食＋堆積）
- 2026/02/18: `space_coastal.py` を `land_evolution_space.py` にリネーム
- 2026/02/18: `land_evolution_space.py` の初期地形で「下部25%の強制標高加工」をデフォルトOFF化（Perlin初期地形を優先）
- 2026/02/18: `land_evolution_space.py` を簡素化（海専用の固定境界・海浜補正・海面下クランプを削除し、通常SPACE計算へ統一）
- 2026/02/18: ルート直下の中間生成物を整理（試行スクリプトを archive/old_scripts、確認用画像・CSVを archive/trial_images へ移動）
- 2026/02/18: ルート直下の実行スクリプト出力を archive/trial_images に固定（画像・CSV の保存先統一）
- 2026/02/18: `land_evolution_space_v2.py` に「SPACE→海用パーティクル」の2段階更新を追加（標高0以下を海中判定）
- 2026/02/18: `land_evolution_space_v2.py` に history フレーム出力を追加（`archive/trial_images/history/<timestamp>_...` へ定期間隔保存）
- 2026/02/18: `land_evolution_space_v2.py` の初期地形高低差を固定300mから「ドメイン長×比率（既定30%）」へ変更（`relief_ratio` パラメータ化）

## タスク管理
- [x] プロジェクト初期化
- [x] 地形生成プロトタイプ作成
- [x] Hatari Labs チュートリアル参考版作成
- [x] パラメータ実験・最適化 → **500x500, 500万年が最適**
- [x] **沿岸地形シミュレーション実装** → archive/temp/coastal_evolution.py
- [x] **SPACE モジュールによる堆積表現** → land_evolution_space.py
- [ ] 3D可視化の改善
- [ ] 実データ（DEM）との連携
- [ ] ゲーム用地形データへの変換

## 参考資料
- [Hatari Labs Tutorial](https://hatarilabs.com/ih-en/modeling-land-evolution-at-basin-scale-with-python-and-landlab-tutorial)
- [Landlab Documentation](https://landlab.readthedocs.io/)

## ファイル構成

```
地形生成トライアル/
├── land_evolution.py           # 【メイン】地形進化シミュレーション
├── natural_terrain.py          # パーリンノイズによる初期地形生成
├── land_evolution_space.py     # SPACE（侵食＋堆積）版
├── terrain_to_tiles.py         # タイルマップ変換
├── memo.md                     # このファイル
├── archive/                    # 過去の試行錯誤
│   ├── trial_images/           # 過去に生成した画像
│   ├── old_scripts/            # 過去の試行スクリプト
│   └── temp/                   # 一時退避スクリプト（coastal_evolution.py など）
└── ドキュメント/               # プロジェクトドキュメント
```

