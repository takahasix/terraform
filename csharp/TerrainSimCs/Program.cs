using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

const int Nrows = 150;
const int Ncols = 150;
const double Dx = 50.0;
const double UpliftRate = 0.001;
// 案B: 境界セルを内険標高の下位N%パーセンタイルに追従させる
// 0にすると従来通り (=無効化)、大きくすると境界が内険と同じ高さに近づく
const double BoundaryPercentile = 20.0;
const double ReliefRatio = 0.30;
const double KBr = 1.0e-5;
const double KSed = 3.0e-5;
const double MSp = 0.5;
const double NSp = 1.0;
const double Phi = 0.3;
const double HStar = 0.5;
const double VS = 1.0;
const double Ff = 0.0;          // 細粒分割合 (F_f)
const double KHs = 0.05;        // 斜面拡散係数 (LinearDiffuser)
const double ThicknessLim = 100.0; // 土層厚上限 (SpaceLargeScaleEroder)
const int Dt = 1000;
const int Tmax = 500000;
const int HistoryInterval = 10000;
// SeaLevel は初期値。境界が上昇すると共に動的に同期する (RunSimulation 内で var に変換)
const double SeaLevelInitial = 0.0;

// ===== 実験的オプション: 海面変動 & 海岸侵食 =====
// EnableSeaLevelChange: 海面高度を時間で変化させる (true にすると有効)
// 200kyr以降に正弦波状の変動を加える。幅と周期は下記定数で調整。
const bool EnableSeaLevelChange = false;
// 200kyr 以降に seaLevel を ±この振幅だけ上下させる (m)
const double SeaLevelAmplitude = 50.0;
// 海面変動の周期 (yr)。例: 100000 = 10万年周期
const double SeaLevelPeriod = 100000.0;
// 海面変動を開始する時刻 (yr)
const int SeaLevelChangeStartTime = 200000;

// EnableCoastalErosion: 海岸線付近の侵食を追加する (true にすると有効)
// 海面±CoastalBeltWidth(m) の標高帯にあるセルに追加の侵食を適用する
const bool EnableCoastalErosion = false;
// 海岸帯の標高幅 (m)。海面から上この範囲内を海岸帯とみなす
const double CoastalBeltWidth = 20.0;
// 波浪侵食係数 (m/yr)。このレートで海岸帯の岩盤を削る
const double CoastalErosionRate = 1.0e-4;
// =========================================

const bool EnableSeaParticles = true;
const int ParticleCount = 3500;
const int ParticleSeed = 123;
const double ParticleEnergySplitQuantile = 60.0;
const bool UseSharedInitialTerrain = true;
const string SharedInitialTerrainCsv = "archive/trial_images/shared_initial_terrain_150x150_seed42.csv";
const string CSharpHistoryRoot = "archive/trial_images/history_cs";

RunSimulation();

return;

static void RunSimulation()
{
    Console.WriteLine("============================================================");
    Console.WriteLine("Terrain Evolution (C# Landlab-equivalent SPACE)");
    Console.WriteLine("============================================================");
    Console.WriteLine($"Grid: {Nrows} x {Ncols}, dx={Dx} m");
    Console.WriteLine($"Uplift: {UpliftRate * 1000.0:F1} mm/yr, dt={Dt} yr, tmax={Tmax / 1000.0:F0} kyr");
    Console.WriteLine($"K_br={KBr:E1}, K_sed={KSed:E1}, m={MSp:F1}, n={NSp:F1}, phi={Phi:F1}, H*={HStar:F1}, v_s={VS:F1}, F_f={Ff:F1}, K_hs={KHs:F3}");
    Console.WriteLine($"Sea particles: {(EnableSeaParticles ? "ON" : "OFF")}, count={ParticleCount}, seed={ParticleSeed}");

    var historyDir = CreateHistoryDir("space_cs_main");
    Console.WriteLine($"History dir: {historyDir}");

    var bedrock = LoadOrGenerateInitialTerrain(
        Nrows,
        Ncols,
        Dx,
        seed: 42,
        reliefRatio: ReliefRatio,
        sharedCsvRelativePath: SharedInitialTerrainCsv,
        useShared: UseSharedInitialTerrain
    );
    var soil = new double[Nrows, Ncols];
    for (var r = 0; r < Nrows; r++)
    {
        for (var c = 0; c < Ncols; c++)
        {
            soil[r, c] = 0.5;
        }
    }

    SetBoundaryToZero(bedrock);
    SetBoundaryToZero(soil);
    var z = SumFields(bedrock, soil);

    double[] particleX = Array.Empty<double>();
    double[] particleY = Array.Empty<double>();
    double[] sedimentLoad = Array.Empty<double>();
    var particleRng = new Random(ParticleSeed);
    (double HighEnergyRatio, double MeanEnergy, double MeanLoad, double MeanSpeed)? statsLast = null;
    if (EnableSeaParticles)
    {
        (particleX, particleY, sedimentLoad) = InitializeParticles(Nrows, Ncols, ParticleCount, particleRng);
    }

    var flow0 = ComputeD8Flow(z, Dx);
    var seaLevel = SeaLevelInitial; // 境界と同期して動的に変化する
    SaveTerrainPng(
        z,
        seaLevel,
        Path.Combine(historyDir, "terrain_0000kyr.png"),
        includeOceanLakeOverlay: true,
        drainageAreaFlat: flow0.Area
    );
    SaveTileMapPng(
        z, soil, bedrock, seaLevel,
        Path.Combine(historyDir, "tilemap_0000kyr.png"),
        drainageAreaFlat: flow0.Area,
        receiverFlat: flow0.Receiver
    );
    SaveSideBySidePng(
        Path.Combine(historyDir, "terrain_0000kyr.png"),
        Path.Combine(historyDir, "tilemap_0000kyr.png"),
        Path.Combine(historyDir, "combined_0000kyr.png")
    );

    var (ocean0, lake0) = ClassifyOcean(z, seaLevel);
    Console.WriteLine($"[t=0] Ocean={Percentage(ocean0):F1}% Lake={Percentage(lake0):F1}%");

    var totalTime = 0;
    while (totalTime < Tmax)
    {
        AddUniformUplift(bedrock, UpliftRate * Dt);
        // 境界は内険標高から下位N%パーセンタイルを計算してその値に追従
        // 返り値はそのパーセンタイル値 = 新しい海面高度
        seaLevel = SetBoundaryToInlandPercentile(bedrock, soil, BoundaryPercentile);
        SetBoundaryToZero(soil); // 土層は境界で0固定

        // 1. 隆起後の地表標高を再計算
        var zNow = SumFields(bedrock, soil);

        // 2. 斜面拡散 (LinearDiffuser相当)
        RunLinearDiffusion(zNow, Nrows, Ncols, Dx, KHs, Dt);
        // LinearDiffuser は z を直接更新するので bedrock を同期
        for (var r = 0; r < Nrows; r++)
            for (var c = 0; c < Ncols; c++)
                bedrock[r, c] = zNow[r, c] - Math.Max(soil[r, c], 0.0);
        SetBoundaryToInlandPercentile(bedrock, soil, BoundaryPercentile);
        SetBoundaryToZero(soil);

        // 3. 窪地充填 (DepressionFinderAndRouter 相当: Priority-Flood)
        var zFilled = new double[Nrows, Ncols];
        Array.Copy(zNow, zFilled, zNow.Length); // flatten が効かないので別途
        {   // 2D -> 1D の別実装
            var flat = new double[Nrows * Ncols];
            for (var r = 0; r < Nrows; r++)
                for (var c = 0; c < Ncols; c++)
                    flat[r * Ncols + c] = zNow[r, c];
            FillDepressions(flat, Nrows, Ncols);
            for (var r = 0; r < Nrows; r++)
                for (var c = 0; c < Ncols; c++)
                    zFilled[r, c] = flat[r * Ncols + c];
        }

        // 4. 充填後地形でD8流向・集水面積計算
        var flow = ComputeD8Flow(zFilled, Dx);

        // 5. SPACE large-scale eroder (正確な式)
        RunSpaceLargeScaleStep(
            bedrock,
            soil,
            flow.Area,
            flow.Slope,
            flow.Receiver,
            flow.Order,
            dt: Dt,
            kBr: KBr,
            kSed: KSed,
            mSp: MSp,
            nSp: NSp,
            phi: Phi,
            hStar: HStar,
            vS: VS,
            fF: Ff,
            thicknessLim: ThicknessLim,
            dx: Dx
        );

        seaLevel = SetBoundaryToInlandPercentile(bedrock, soil, BoundaryPercentile);
        SetBoundaryToZero(soil);
        z = SumFields(bedrock, soil);

        // ===== 実験的: 海面変動 =====
        if (EnableSeaLevelChange && totalTime >= SeaLevelChangeStartTime)
        {
            // 基準値 seaLevel (境界パーセンタイル) に正弦波変動を重ねる
            var phase = 2.0 * Math.PI * (totalTime - SeaLevelChangeStartTime) / SeaLevelPeriod;
            seaLevel += SeaLevelAmplitude * Math.Sin(phase);
        }

        // ===== 実験的: 海岸侵食 =====
        if (EnableCoastalErosion)
            ApplyCoastalErosion(bedrock, soil, seaLevel, CoastalBeltWidth, CoastalErosionRate, Dt);

        if (EnableSeaParticles)
        {
            var flowForParticles = ComputeD8Flow(z, Dx);
            statsLast = UpdateSeaParticles(
                z,
                particleX,
                particleY,
                sedimentLoad,
                particleRng,
                flowForParticles.Area,
                Dx,
                Dt,
                ParticleEnergySplitQuantile
            );
        }

        totalTime += Dt;

        if (totalTime % HistoryInterval == 0)
        {
            CheckBoundaryAndAlert(z, totalTime);
            var meanSoil = MeanField(soil);
            if (statsLast.HasValue)
            {
                var s = statsLast.Value;
                Console.WriteLine(
                    $"  [state] mean soil={meanSoil:F3} m, max z={MaxField(z):F1} m, " +
                    $"high-E={s.HighEnergyRatio:F2}, mean E={s.MeanEnergy:F4}, p-speed={s.MeanSpeed:F2}");
            }
            else
            {
                Console.WriteLine($"  [state] mean soil={meanSoil:F3} m, max z={MaxField(z):F1} m");
            }
            var (ocean, lake) = ClassifyOcean(z, seaLevel);
            var outPath = Path.Combine(historyDir, $"terrain_{totalTime / 1000:0000}kyr.png");
            var tileOutPath = Path.Combine(historyDir, $"tilemap_{totalTime / 1000:0000}kyr.png");
            var flowViz = ComputeD8Flow(z, Dx);
            SaveTerrainPng(
                z,
                seaLevel,
                outPath,
                includeOceanLakeOverlay: true,
                drainageAreaFlat: flowViz.Area
            );
            SaveTileMapPng(
                z, soil, bedrock, seaLevel,
                tileOutPath,
                drainageAreaFlat: flowViz.Area,
                receiverFlat: flowViz.Receiver
            );
            var combinedPath = Path.Combine(historyDir, $"combined_{totalTime / 1000:0000}kyr.png");
            SaveSideBySidePng(outPath, tileOutPath, combinedPath);
            // 高解像度タイルマップ (1000×1000, ワンホット+ガウシアン+argmax)
            var hiresTileOutPath = Path.Combine(historyDir, $"tilemap_hires_{totalTime / 1000:0000}kyr.png");
            SaveHighResTileMapPng(z, seaLevel, hiresTileOutPath,
                drainageAreaFlat: flowViz.Area,
                receiverFlat: flowViz.Receiver);
            Console.WriteLine(
                $"  [history] {totalTime / 1000.0:F0} kyr: SeaLevel={seaLevel:F1}m Ocean={Percentage(ocean):F1}% Lake={Percentage(lake):F1}% -> {Path.GetFileName(outPath)}");
        }
    }

    Console.WriteLine("Simulation complete.");
}

// terrain画像とtilemap画像を横に並べて1枚のPNGに結合して保存する
static void SaveSideBySidePng(string leftPath, string rightPath, string outPath)
{
    using var left = Image.Load<Rgb24>(leftPath);
    using var right = Image.Load<Rgb24>(rightPath);
    var w = left.Width + right.Width;
    var h = Math.Max(left.Height, right.Height);
    using var combined = new Image<Rgb24>(w, h, new Rgb24(20, 20, 20));
    combined.Mutate(ctx =>
    {
        ctx.DrawImage(left, new SixLabors.ImageSharp.Point(0, 0), 1f);
        ctx.DrawImage(right, new SixLabors.ImageSharp.Point(left.Width, 0), 1f);
    });
    combined.SaveAsPng(outPath);
}

static string CreateHistoryDir(string runName)
{
    var stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
    var root = ResolveRepoPath(CSharpHistoryRoot);
    var dir = Path.Combine(root, $"{stamp}_{runName}");
    Directory.CreateDirectory(dir);
    return dir;
}

static string ResolveRepoPath(string repoRelativePath)
{
    var current = new DirectoryInfo(Directory.GetCurrentDirectory());
    while (current is not null)
    {
        var marker = Path.Combine(current.FullName, "memo.md");
        if (File.Exists(marker))
        {
            var normalized = repoRelativePath.Replace('/', Path.DirectorySeparatorChar);
            return Path.Combine(current.FullName, normalized);
        }
        current = current.Parent;
    }

    throw new DirectoryNotFoundException("Could not locate repository root (memo.md not found in parent hierarchy)");
}

static double[,] LoadOrGenerateInitialTerrain(
    int nrows,
    int ncols,
    double dx,
    int seed,
    double reliefRatio,
    string sharedCsvRelativePath,
    bool useShared)
{
    var csvPath = ResolveRepoPath(sharedCsvRelativePath);
    if (useShared)
    {
        if (!File.Exists(csvPath))
        {
            throw new FileNotFoundException(
                "Shared initial terrain CSV was not found. Run Python once to export it.",
                csvPath
            );
        }

        var loaded = LoadTerrainCsv(csvPath, nrows, ncols);
        Console.WriteLine($"  [terrain] loaded shared CSV: {csvPath}");
        return loaded;
    }

    var generated = GenerateInitialTerrain(nrows, ncols, dx, seed, reliefRatio);
    SaveTerrainCsv(generated, csvPath);
    Console.WriteLine($"  [terrain] exported shared CSV: {csvPath}");
    return generated;
}

static double[,] LoadTerrainCsv(string csvPath, int nrows, int ncols)
{
    var lines = File.ReadAllLines(csvPath);
    if (lines.Length != nrows)
    {
        throw new InvalidDataException($"Row count mismatch in {csvPath}: expected {nrows}, got {lines.Length}");
    }

    var data = new double[nrows, ncols];
    for (var r = 0; r < nrows; r++)
    {
        var parts = lines[r].Split(',');
        if (parts.Length != ncols)
        {
            throw new InvalidDataException($"Column count mismatch at row {r} in {csvPath}: expected {ncols}, got {parts.Length}");
        }

        for (var c = 0; c < ncols; c++)
        {
            data[r, c] = double.Parse(parts[c], CultureInfo.InvariantCulture);
        }
    }

    return data;
}

static void SaveTerrainCsv(double[,] terrain, string csvPath)
{
    var dir = Path.GetDirectoryName(csvPath);
    if (!string.IsNullOrWhiteSpace(dir))
    {
        Directory.CreateDirectory(dir);
    }

    using var writer = new StreamWriter(csvPath, false);
    var nrows = terrain.GetLength(0);
    var ncols = terrain.GetLength(1);
    for (var r = 0; r < nrows; r++)
    {
        for (var c = 0; c < ncols; c++)
        {
            if (c > 0)
            {
                writer.Write(',');
            }
            writer.Write(terrain[r, c].ToString("G17", CultureInfo.InvariantCulture));
        }
        writer.WriteLine();
    }
}

// ============================================================
// matplotlib の terrain カラーマップ (主要ノード 9 点を手動定義)
// https://matplotlib.org/stable/gallery/color/colormap_reference.html
// ============================================================
static (byte R, byte G, byte B) TerrainColormap(double t)
{
    // matplotlib 'terrain' の代表的な色ノード (t=0..1)
    // t=0.0: 深海 (0, 0, 0.5)
    // t=0.17: 浅海 (0, 0.6, 1.0)
    // t=0.23: 海岸線/砂 (0.741, 0.718, 0.420)
    // t=0.27: 低地 (0.18, 0.55, 0.18)
    // t=0.50: 山麓 (0.42, 0.65, 0.30)
    // t=0.75: 高山 (0.73, 0.70, 0.55)
    // t=0.87: 岩肌 (0.80, 0.77, 0.68)
    // t=1.00: 雪 (1.0, 1.0, 1.0)
    ReadOnlySpan<double> ks = [0.00, 0.17, 0.23, 0.27, 0.50, 0.75, 0.87, 1.00];
    ReadOnlySpan<double> rs = [0.00, 0.00, 0.741, 0.18, 0.42, 0.73, 0.80, 1.00];
    ReadOnlySpan<double> gs = [0.00, 0.60, 0.718, 0.55, 0.65, 0.70, 0.77, 1.00];
    ReadOnlySpan<double> bs = [0.50, 1.00, 0.420, 0.18, 0.30, 0.55, 0.68, 1.00];

    t = Math.Clamp(t, 0.0, 1.0);
    var i = 0;
    while (i < ks.Length - 2 && t > ks[i + 1]) i++;
    var lo = ks[i]; var hi = ks[i + 1];
    var u = (hi > lo) ? (t - lo) / (hi - lo) : 0.0;
    var rv = rs[i] + u * (rs[i + 1] - rs[i]);
    var gv = gs[i] + u * (gs[i + 1] - gs[i]);
    var bv = bs[i] + u * (bs[i + 1] - bs[i]);
    return ((byte)(rv * 255), (byte)(gv * 255), (byte)(bv * 255));
}

// ============================================================
// LightSource ヒルシェーディング
// matplotlib LightSource(azdeg=315, altdeg=45), vert_exag=2
// blend_mode='overlay' を近似: shade * overlay_blend(base, intensity)
// ============================================================
static double[,] ComputeHillshade(double[,] z, double dx,
    double azdegDeg = 315.0, double altdegDeg = 45.0, double vertExag = 2.0)
{
    var nrows = z.GetLength(0);
    var ncols = z.GetLength(1);
    var hs = new double[nrows, ncols];

    // 光源ベクトル (matplotlib と同じ定義)
    var az = (360.0 - azdegDeg + 90.0) * Math.PI / 180.0; // 北→東→南→西
    var alt = altdegDeg * Math.PI / 180.0;
    var lx = Math.Cos(alt) * Math.Cos(az);
    var ly = Math.Cos(alt) * Math.Sin(az);
    var lz = Math.Sin(alt);

    for (var r = 0; r < nrows; r++)
    {
        for (var c = 0; c < ncols; c++)
        {
            // 中央差分で法線ベクトルを計算 (vert_exag を掛けた高さで)
            // 配列インデックス: r=0が南端、r=nrows-1が北端 (origin='lower')
            // dzdx_geo = 東西方向の勾配 (列 c 方向)
            // dzdy_geo = 南北方向の勾配 (行 r 方向、r増加=北向きなので符号そのまま)
            // 光源ベクトルは lx=東西成分, ly=南北成分 なので対応させる
            var dzdx_geo = c > 0 && c < ncols - 1
                ? (z[r, c + 1] - z[r, c - 1]) * vertExag / (2.0 * dx)   // 東向きが正
                : 0.0;
            var dzdy_geo = r > 0 && r < nrows - 1
                ? (z[r + 1, c] - z[r - 1, c]) * vertExag / (2.0 * dx)   // 北向きが正
                : 0.0;
            // 法線 n = (-dz/dx, -dz/dy, 1) を正規化
            var nx = -dzdx_geo; var ny = -dzdy_geo; var nz = 1.0;
            var nlen = Math.Sqrt(nx * nx + ny * ny + nz * nz);
            nx /= nlen; ny /= nlen; nz /= nlen;
            // 輝度 = 内積, 0..1 にクランプ
            hs[r, c] = Math.Max(0.0, lx * nx + ly * ny + lz * nz);
        }
    }
    return hs;
}

// overlay blend: base=カラーマップ輝度値 (0..1), intensity=hillshade (0..1)
// matplotlib blend_mode='overlay' の近似
static double OverlayBlend(double base_, double intensity)
{
    // overlay: if base < 0.5 => 2*base*intensity, else 1-2*(1-base)*(1-intensity)
    if (base_ < 0.5)
        return 2.0 * base_ * intensity;
    else
        return 1.0 - 2.0 * (1.0 - base_) * (1.0 - intensity);
}

// ============================================================
// タイルマップ描画
// 分類ルール:
//   Water : z <= seaLevel
//   Sand  : seaLevel < z <= seaLevel+SandBand  AND  水辺に隣接(1セル以内)
//   Soil  : soil_depth >= SoilThreshold
//   Rock  : それ以外(薄い土+露頭岩盤)
//   川    : SaveTerrainPng と同じ集水面積比例の円形ブラシで Water 色を上書き
// ============================================================
// タイル分類の山寄計算を共通化: seaLevel と z[,] から sandThreshold/rockThreshold を返す
static (double sand, double rock) CalcTileThresholds(double[,] z, double seaLevel,
    double sandPercentile = 20.0, double rockPercentile = 80.0)
{
    var nrows = z.GetLength(0); var ncols = z.GetLength(1);
    var landElevs = new List<double>();
    for (var r = 0; r < nrows; r++)
        for (var c = 0; c < ncols; c++)
            if (z[r, c] > seaLevel) landElevs.Add(z[r, c]);
    var sand = landElevs.Count > 0 ? Percentile(landElevs, sandPercentile) : seaLevel + 1.0;
    var rock = landElevs.Count > 0 ? Percentile(landElevs, rockPercentile) : seaLevel + 2.0;
    return (sand, rock);
}

// 0=Water,1=Sand,2=Soil,3=Rock のタイル選择
// 山寄基準: seaLevel以下=Water, sand以下=Sand, rock以下=Soil, それ以上=Rock
static int ClassifyTile(double elevation, double seaLevel, double sandThreshold, double rockThreshold)
    => elevation <= seaLevel ? 0
     : elevation <= sandThreshold ? 1
     : elevation <= rockThreshold ? 2
     : 3;

// マジョリティフィルター: 3×3ネイバーの多数決でタイルを滑らかにする (passes回実施)
static int[,] MajorityFilter(int[,] src, int passes = 3)
{
    var nrows = src.GetLength(0); var ncols = src.GetLength(1);
    var cur = (int[,])src.Clone();
    var nxt = new int[nrows, ncols];
    for (var p = 0; p < passes; p++)
    {
        for (var r = 0; r < nrows; r++)
            for (var c = 0; c < ncols; c++)
            {
                var counts = new int[4];
                for (var dr = -1; dr <= 1; dr++)
                    for (var dc = -1; dc <= 1; dc++)
                    {
                        var rr = Math.Clamp(r + dr, 0, nrows - 1);
                        var cc = Math.Clamp(c + dc, 0, ncols - 1);
                        counts[cur[rr, cc]]++;
                    }
                // 最高数のタイルを選択。同数の場合は元の値を維持
                var best = cur[r, c];
                for (var t = 0; t < 4; t++) if (counts[t] > counts[best]) best = t;
                nxt[r, c] = best;
            }
        (cur, nxt) = (nxt, cur);
    }
    return cur;
}

// 分離可能ガウシアンカーネルで float[,] を畳み込む
static float[,] GaussianBlur2D(float[,] src, double sigma)
{
    var nrows = src.GetLength(0); var ncols = src.GetLength(1);
    // カーネル半径 = ceil(3σ)
    var radius = (int)Math.Ceiling(3.0 * sigma);
    var ksize = 2 * radius + 1;
    var kernel = new float[ksize];
    double sum = 0.0;
    for (var i = 0; i < ksize; i++)
    {
        var x = i - radius;
        kernel[i] = (float)Math.Exp(-0.5 * x * x / (sigma * sigma));
        sum += kernel[i];
    }
    for (var i = 0; i < ksize; i++) kernel[i] /= (float)sum;

    // 水平方向
    var tmp = new float[nrows, ncols];
    for (var r = 0; r < nrows; r++)
        for (var c = 0; c < ncols; c++)
        {
            float v = 0f;
            for (var k = 0; k < ksize; k++)
                v += kernel[k] * src[r, Math.Clamp(c + k - radius, 0, ncols - 1)];
            tmp[r, c] = v;
        }
    // 垂直方向
    var dst = new float[nrows, ncols];
    for (var r = 0; r < nrows; r++)
        for (var c = 0; c < ncols; c++)
        {
            float v = 0f;
            for (var k = 0; k < ksize; k++)
                v += kernel[k] * tmp[Math.Clamp(r + k - radius, 0, nrows - 1), c];
            dst[r, c] = v;
        }
    return dst;
}

// 高解像度タイルマップ: 1000×1000 px
// ワンホットベクトル化 → チャンネルごとガウシアンブラー → argmax で4色くっきり
// sigma: ブラー強度 (セル単位)。大きいほど境界が丸くなる
static void SaveHighResTileMapPng(
    double[,] z,
    double seaLevel,
    string filePath,
    int targetSize = 1000,
    double smoothSigma = 1.5,
    double sandPercentile = 20.0,
    double rockPercentile = 80.0,
    double[]? drainageAreaFlat = null,
    int[]? receiverFlat = null)
{
    var nrows = z.GetLength(0);
    var ncols = z.GetLength(1);

    var dir = Path.GetDirectoryName(filePath);
    if (!string.IsNullOrWhiteSpace(dir))
        Directory.CreateDirectory(dir);

    // --- タイル分類 (ネイティブ解像度 150×150) ---
    var (sandThreshold, rockThreshold) = CalcTileThresholds(z, seaLevel, sandPercentile, rockPercentile);
    var tile = new int[nrows, ncols];
    for (var r = 0; r < nrows; r++)
        for (var c = 0; c < ncols; c++)
            tile[r, c] = ClassifyTile(z[r, c], seaLevel, sandThreshold, rockThreshold);

    // --- ワンホットベクトル化: 4チャンネルの float[,] を作成 ---
    // ch[t][r,c] = そのセルがタイル t なら 1.0f, それ以外 0.0f
    const int NumTiles = 4;
    var ch = new float[NumTiles][,];
    for (var t = 0; t < NumTiles; t++)
    {
        ch[t] = new float[nrows, ncols];
        for (var r = 0; r < nrows; r++)
            for (var c = 0; c < ncols; c++)
                ch[t][r, c] = tile[r, c] == t ? 1.0f : 0.0f;
    }

    // --- 各チャンネルにガウシアンブラーをかける ---
    for (var t = 0; t < NumTiles; t++)
        ch[t] = GaussianBlur2D(ch[t], smoothSigma);

    // --- 出力画像を生成: 各出力ピクセルで argmax → 4色くっきり ---
    // タイル色定義
    // 0=Water (30,100,200)  1=Sand (210,190,130)  2=Soil (80,140,55)  3=Rock (150,145,140)
    static (byte r, byte g, byte b) TileColorB(int t) => t switch
    {
        0 => (30, 100, 200),
        1 => (210, 190, 130),
        2 => (80, 140, 55),
        _ => (150, 145, 140),
    };

    using var image = new Image<Rgb24>(targetSize, targetSize);
    for (var py = 0; py < targetSize; py++)
        for (var px = 0; px < targetSize; px++)
        {
            // 出力ピクセル → ネイティブグリッド上の浮動小数点位置
            var tx = (px + 0.5) * ncols / targetSize - 0.5;   // 列方向 (0 = 西端)
            var tyImg = (py + 0.5) * nrows / targetSize - 0.5; // 画像行 (0 = 上端)
            var ty = (nrows - 1) - tyImg;                       // グリッド行 (0 = 南端)

            // 双線形補間で各チャンネルの値を取得
            var x0 = (int)Math.Floor(tx); var y0 = (int)Math.Floor(ty);
            var fx = (float)(tx - x0); var fy = (float)(ty - y0);
            int X(int v) => Math.Clamp(v, 0, ncols - 1);
            int Y(int v) => Math.Clamp(v, 0, nrows - 1);

            // argmax: 4チャンネルの補間値が最大のタイルを選ぶ
            var best = 0; var bestVal = float.MinValue;
            for (var t = 0; t < NumTiles; t++)
            {
                var v00 = ch[t][Y(y0), X(x0)];
                var v10 = ch[t][Y(y0), X(x0 + 1)];
                var v01 = ch[t][Y(y0 + 1), X(x0)];
                var v11 = ch[t][Y(y0 + 1), X(x0 + 1)];
                var val = v00 * (1 - fx) * (1 - fy)
                        + v10 * fx * (1 - fy)
                        + v01 * (1 - fx) * fy
                        + v11 * fx * fy;
                if (val > bestVal) { bestVal = val; best = t; }
            }

            var (cr, cg, cb) = TileColorB(best);
            image[px, py] = new Rgb24(cr, cg, cb);
        }

    // --- 川レイヤー後処理: Gaussian+argmax後にピクセル直接描画 ---
    // SaveTileMapPng と同じ最近傍線分アルゴリズムで川を上書き
    // (川をGaussian前に混ぜると境界がにじんでチリジリになるため後処理が正解)
    if (drainageAreaFlat is not null && receiverFlat is not null)
    {
        // --- 川の riverRadius 計算 (px単位: 1セル = targetSize/ncols px) ---
        var cellPx = (double)targetSize / ncols; // 1セルあたりのピクセル数
        var riverRadius = new double[nrows, ncols];
        var posDrain = new List<double>();
        for (var rr = 0; rr < nrows; rr++)
            for (var cc = 0; cc < ncols; cc++)
            {
                if (z[rr, cc] <= seaLevel) continue;
                var a = drainageAreaFlat[rr * ncols + cc];
                if (a > 0.0) posDrain.Add(a);
            }
        if (posDrain.Count > 0)
        {
            var drThreshold = Percentile(posDrain, 80.0);
            var logMin = Math.Log(1.0 + drThreshold);
            var logMax = Math.Log(1.0 + posDrain.Max());
            var logRange = Math.Max(logMax - logMin, 1e-12);
            for (var rr = 0; rr < nrows; rr++)
                for (var cc = 0; cc < ncols; cc++)
                {
                    if (z[rr, cc] <= seaLevel) continue;
                    var a = drainageAreaFlat[rr * ncols + cc];
                    if (a < drThreshold) continue;
                    var norm = Math.Clamp((Math.Log(1.0 + a) - logMin) / logRange, 0.0, 1.0);
                    riverRadius[rr, cc] = 1.2 * norm * cellPx; // px単位
                }
        }

        // --- Catmull-Romスプライン化のための前処理: 各riverノードの「最良上流」を求める ---
        // 上流ノードが複数ある場合は最大riverRadiusを持つものを採用（本流優先）
        var bestUpstream = new int[nrows * ncols];
        Array.Fill(bestUpstream, -1);
        for (var rr = 0; rr < nrows; rr++)
            for (var cc = 0; cc < ncols; cc++)
            {
                if (z[rr, cc] <= seaLevel || riverRadius[rr, cc] <= 0.0) continue;
                var node = rr * ncols + cc;
                var rec = receiverFlat[node];
                if (rec == node) continue;
                // このnodeはrecの上流候補: より大きなriverRadiusなら更新
                var curBest = bestUpstream[rec];
                if (curBest < 0 || riverRadius[rr, cc] > riverRadius[curBest / ncols, curBest % ncols])
                    bestUpstream[rec] = node;
            }

        // --- Catmull-Romスプラインをサンプリングして微小線分列に変換 ---
        // 各辺(node→rec)に対してP0(上流),P1(node),P2(rec),P3(下流)の4制御点を設定し
        // t∈[0,1]を CatmullRomSubdiv 等分でサンプリングして短い直線セグメントに分解する
        // → 既存の最近傍線分アルゴリズムでそのまま描画できる
        const int CatmullRomSubdiv = 8; // 1辺あたりのサブ分割数（増やすほど滑らか）

        // 画像座標変換ヘルパー（グリッド行インデックス→画像Y）
        double GridToImgX(int c) => (c + 0.5) * cellPx;
        double GridToImgY(int r) => (nrows - 1 - r + 0.5) * cellPx;

        // Centripetal Catmull-Rom (alpha=0.5): Barry-Goldman法
        // コード長^alpha で再パラメータ化するため、不均一間隔でもオーバーシュートしない
        // t_in は P1→P2 区間内の正規化パラメータ (0..1)
        static (double x, double y) CatmullRomCentripetal(
            double p0x, double p0y, double p1x, double p1y,
            double p2x, double p2y, double p3x, double p3y, double tNorm)
        {
            static double KnotDist(double ax, double ay, double bx, double by)
            {
                var dx = bx - ax; var dy = by - ay;
                // alpha=0.5: sqrt(|d|) = |d|^0.5
                return Math.Pow(dx * dx + dy * dy, 0.25); // sqrt(sqrt(dx²+dy²))
            }
            var t0 = 0.0;
            var t1 = t0 + KnotDist(p0x, p0y, p1x, p1y);
            var t2 = t1 + KnotDist(p1x, p1y, p2x, p2y);
            var t3 = t2 + KnotDist(p2x, p2y, p3x, p3y);

            // P1..P2 区間で t_in ∈ [0,1] → t ∈ [t1, t2]
            var t = t1 + tNorm * (t2 - t1);

            // 縮退ガード: 区間幅が極小なら線形補間
            if (t2 - t1 < 1e-9) return (p1x + tNorm * (p2x - p1x), p1y + tNorm * (p2y - p1y));

            static double Lerp(double a, double b, double ta, double tb, double tv)
                => (Math.Abs(tb - ta) < 1e-12) ? a : a + (b - a) * (tv - ta) / (tb - ta);

            var a1x = Lerp(p0x, p1x, t0, t1, t); var a1y = Lerp(p0y, p1y, t0, t1, t);
            var a2x = Lerp(p1x, p2x, t1, t2, t); var a2y = Lerp(p1y, p2y, t1, t2, t);
            var a3x = Lerp(p2x, p3x, t2, t3, t); var a3y = Lerp(p2y, p3y, t2, t3, t);
            var b1x = Lerp(a1x, a2x, t0, t2, t); var b1y = Lerp(a1y, a2y, t0, t2, t);
            var b2x = Lerp(a2x, a3x, t1, t3, t); var b2y = Lerp(a2y, a3y, t1, t3, t);
            return (Lerp(b1x, b2x, t1, t2, t), Lerp(b1y, b2y, t1, t2, t));
        }

        var segments = new List<(double ax, double ay, double bx, double by, double radA, double radB)>();
        for (var rr = 0; rr < nrows; rr++)
            for (var cc = 0; cc < ncols; cc++)
            {
                var node = rr * ncols + cc;
                var rec = receiverFlat[node];
                if (rec == node) continue;
                if (z[rr, cc] <= seaLevel) continue;
                var recR = rec / ncols; var recC = rec % ncols;
                var receiverIsOcean = z[recR, recC] <= seaLevel;
                var radA = riverRadius[rr, cc];
                var radB = receiverIsOcean ? radA : riverRadius[recR, recC];
                if (Math.Max(radA, radB) < 1.0) continue;

                // 短い支流フィルタ: bestUpstream を遡って MinRiverChainCells セル未満はスキップ
                // 1セル ≈ cellPx px なので 4セル ≈ 26px 未満の支流を除去
                const int MinRiverChainCells = 4;
                {
                    var walkNode = node;
                    var chainLen = 0;
                    while (chainLen < MinRiverChainCells)
                    {
                        chainLen++;
                        var upCheck = bestUpstream[walkNode];
                        if (upCheck < 0 || riverRadius[upCheck / ncols, upCheck % ncols] <= 0.0) break;
                        walkNode = upCheck;
                    }
                    if (chainLen < MinRiverChainCells) continue;
                }

                // P1 = node, P2 = receiver
                var p1x = GridToImgX(cc); var p1y = GridToImgY(rr);
                var p2x = GridToImgX(recC); var p2y = GridToImgY(recR);

                // P0 = P1の上流ノード（なければ P0=2*P1-P2 で外挿）
                double p0x, p0y;
                var up = bestUpstream[node];
                if (up >= 0 && riverRadius[up / ncols, up % ncols] > 0.0)
                {
                    p0x = GridToImgX(up % ncols); p0y = GridToImgY(up / ncols);
                }
                else { p0x = 2 * p1x - p2x; p0y = 2 * p1y - p2y; }

                // P3 = recの下流ノード（なければ P3=2*P2-P1 で外挿）
                double p3x, p3y;
                var rec2 = receiverFlat[rec];
                var rec2R = rec2 / ncols; var rec2C = rec2 % ncols;
                if (rec2 != rec && z[rec2R, rec2C] > seaLevel && riverRadius[rec2R, rec2C] > 0.0)
                {
                    p3x = GridToImgX(rec2C); p3y = GridToImgY(rec2R);
                }
                else { p3x = 2 * p2x - p1x; p3y = 2 * p2y - p1y; }

                // Centripetal Catmull-Romをサブ分割して短い線分列に変換
                var (prevX, prevY) = CatmullRomCentripetal(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, 0.0);
                for (var si = 1; si <= CatmullRomSubdiv; si++)
                {
                    var t = si / (double)CatmullRomSubdiv;
                    var (curX, curY) = CatmullRomCentripetal(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, t);
                    var tMid = (si - 0.5) / CatmullRomSubdiv;
                    var interpRad = radA + tMid * (radB - radA);
                    if (interpRad >= 1.0)
                        segments.Add((prevX, prevY, curX, curY, interpRad, interpRad));
                    (prevX, prevY) = (curX, curY);
                }
            }

        // --- 空間バケット ---
        var bucketPx = (int)Math.Max(1, cellPx * 2);
        var bucketCols = (targetSize + bucketPx - 1) / bucketPx;
        var bucketRows = (targetSize + bucketPx - 1) / bucketPx;
        var buckets = new List<int>[bucketRows, bucketCols];
        for (var br = 0; br < bucketRows; br++)
            for (var bc = 0; bc < bucketCols; bc++)
                buckets[br, bc] = [];
        for (var si = 0; si < segments.Count; si++)
        {
            var (ax, ay, bx, by, radA, radB) = segments[si];
            var maxRad = Math.Max(radA, radB);
            var bx0 = (int)Math.Max(0, Math.Floor((Math.Min(ax, bx) - maxRad) / bucketPx));
            var bx1 = (int)Math.Min(bucketCols - 1, Math.Floor((Math.Max(ax, bx) + maxRad) / bucketPx));
            var by0 = (int)Math.Max(0, Math.Floor((Math.Min(ay, by) - maxRad) / bucketPx));
            var by1 = (int)Math.Min(bucketRows - 1, Math.Floor((Math.Max(ay, by) + maxRad) / bucketPx));
            for (var br = by0; br <= by1; br++)
                for (var bc = bx0; bc <= bx1; bc++)
                    buckets[br, bc].Add(si);
        }

        // --- ピクセルごとに最近傍線分距離判定 ---
        var (wr, wg, wb) = TileColorB(0);
        for (var py = 0; py < targetSize; py++)
            for (var px = 0; px < targetSize; px++)
            {
                var bRow = py / bucketPx; var bCol = px / bucketPx;
                var insideRiver = false;
                for (var dbr = -1; dbr <= 1 && !insideRiver; dbr++)
                    for (var dbc = -1; dbc <= 1 && !insideRiver; dbc++)
                    {
                        var nbr = bRow + dbr; var nbc = bCol + dbc;
                        if (nbr < 0 || nbr >= bucketRows || nbc < 0 || nbc >= bucketCols) continue;
                        foreach (var si in buckets[nbr, nbc])
                        {
                            var (ax, ay, bx, by, radA, radB) = segments[si];
                            var abx = bx - ax; var aby = by - ay;
                            var ab2 = abx * abx + aby * aby;
                            double tParam; double dist2;
                            if (ab2 < 1e-12)
                            {
                                var ddx = px - ax; var ddy = py - ay;
                                dist2 = ddx * ddx + ddy * ddy; tParam = 0.0;
                            }
                            else
                            {
                                tParam = Math.Clamp(((px - ax) * abx + (py - ay) * aby) / ab2, 0.0, 1.0);
                                var qx = ax + tParam * abx - px;
                                var qy = ay + tParam * aby - py;
                                dist2 = qx * qx + qy * qy;
                            }
                            var interpRad = radA + tParam * (radB - radA);
                            if (interpRad < 1.0) continue;
                            if (dist2 <= interpRad * interpRad) { insideRiver = true; break; }
                        }
                    }
                if (insideRiver)
                    image[px, py] = new Rgb24(wr, wg, wb);
            }
    }

    image.SaveAsPng(filePath);
}

static void SaveTileMapPng(
    double[,] z,
    double[,] soil,
    double[,] bedrock,
    double seaLevel,
    string filePath,
    double[]? drainageAreaFlat = null,
    int[]? receiverFlat = null)
{
    const int Scale = 4;
    // タイル分類は陸上セルの標高パーセンタイルで決める
    const double SandPercentile = 20.0;
    const double RockPercentile = 80.0;

    var nrows = z.GetLength(0);
    var ncols = z.GetLength(1);

    var dir = Path.GetDirectoryName(filePath);
    if (!string.IsNullOrWhiteSpace(dir))
        Directory.CreateDirectory(dir);

    // --- 陸上セルの標高パーセンタイルを計算 ---
    var (sandThreshold, rockThreshold) = CalcTileThresholds(z, seaLevel, SandPercentile, RockPercentile);

    // --- 川: 集水面積を対数正規化して各セルの riverRadius を計算 ---
    // （SaveTerrainPng と同じロジック）
    var riverRadius = new double[nrows, ncols]; // 0 = 川なし
    if (drainageAreaFlat is not null)
    {
        var posDrain = new List<double>();
        for (var r = 0; r < nrows; r++)
            for (var c = 0; c < ncols; c++)
            {
                if (z[r, c] <= seaLevel) continue;
                var a = drainageAreaFlat[r * ncols + c];
                if (a > 0.0) posDrain.Add(a);
            }
        if (posDrain.Count > 0)
        {
            var threshold = Percentile(posDrain, 80.0);
            var logMin = Math.Log(1.0 + threshold);
            var logMax = Math.Log(1.0 + posDrain.Max());
            var logRange = Math.Max(logMax - logMin, 1e-12);
            for (var r = 0; r < nrows; r++)
                for (var c = 0; c < ncols; c++)
                {
                    if (z[r, c] <= seaLevel) continue;
                    var a = drainageAreaFlat[r * ncols + c];
                    if (a < threshold) continue;
                    var norm = Math.Clamp((Math.Log(1.0 + a) - logMin) / logRange, 0.0, 1.0);
                    riverRadius[r, c] = 1.2 * norm * Scale;
                }
        }
    }

    // --- 各セルのタイル種別を決定 (川はベース描画後に上書き) ---
    // 0=Water, 1=Sand, 2=Soil, 3=Rock
    var tile = new int[nrows, ncols];
    for (var r = 0; r < nrows; r++)
        for (var c = 0; c < ncols; c++)
            tile[r, c] = ClassifyTile(z[r, c], seaLevel, sandThreshold, rockThreshold);

    // --- タイル色定義 ---
    // Water: 深めの青  ( 30, 100, 200)
    // Sand : 砂色      (210, 190, 130)
    // Soil : 草緑      ( 80, 140,  55)
    // Rock : グレー    (150, 145, 140)
    static (byte R, byte G, byte B) TileColor(int t) => t switch
    {
        0 => (30, 100, 200),
        1 => (210, 190, 130),
        2 => (80, 140, 55),
        _ => (150, 145, 140),
    };

    // --- ベースレイヤー描画 ---
    using var image = new Image<Rgb24>(ncols * Scale, nrows * Scale);
    for (var r = 0; r < nrows; r++)
    {
        var imgRow = nrows - 1 - r; // origin='lower'
        for (var c = 0; c < ncols; c++)
        {
            var (cr, cg, cb) = TileColor(tile[r, c]);
            for (var dy = 0; dy < Scale; dy++)
                for (var dx2 = 0; dx2 < Scale; dx2++)
                    image[c * Scale + dx2, imgRow * Scale + dy] = new Rgb24(cr, cg, cb);
        }
    }

    // --- 川レイヤー: 最近傍線分距離によるラスタライズ ---
    // アルゴリズム:
    //   1. 有効な川セグメント (node→receiver) を全てリストアップし、
    //      空間バケット（セルサイズ = BucketPx）に登録する
    //   2. 各ピクセルの周辺バケットだけを検索して最近傍線分を求める
    //   3. そのピクセルから最近傍線分への距離 ≤ 線分上の補間半径 なら川と判定
    //   → D8の8方向パターンが繰り返し見える問題を解消する
    var (wr, wg, wb) = TileColor(0); // Water 色
    if (receiverFlat is not null)
    {
        var imgW = ncols * Scale;
        var imgH = nrows * Scale;

        // --- ステップ1: 川セグメントをリストアップ ---
        // 各要素: (ax, ay, bx, by, radA, radB)
        //   A = 上流ノードの画像座標, B = 下流ノードの画像座標
        //   radA/radB = それぞれのノードの川幅半径（ピクセル）
        var segments = new List<(double ax, double ay, double bx, double by, double radA, double radB)>();
        for (var r = 0; r < nrows; r++)
            for (var c = 0; c < ncols; c++)
            {
                var node = r * ncols + c;
                var rec = receiverFlat[node];
                if (rec == node) continue;
                if (z[r, c] <= seaLevel) continue;

                var recR = rec / ncols;
                var recC = rec % ncols;
                var receiverIsOcean = z[recR, recC] <= seaLevel;

                var radA = riverRadius[r, c];
                // receiver が海の場合は上流端と同じ幅のまま接続（先細りにしない）
                var radB = receiverIsOcean ? radA : riverRadius[recR, recC];
                if (radA <= 0.0 && radB <= 0.0) continue;

                // 少なくとも一端が1px以上の場合のみ登録
                if (Math.Max(radA, radB) < 1.0) continue;

                double ax = c * Scale + Scale * 0.5;
                double ay = (nrows - 1 - r) * Scale + Scale * 0.5;
                double bx = recC * Scale + Scale * 0.5;
                double by = (nrows - 1 - recR) * Scale + Scale * 0.5;
                segments.Add((ax, ay, bx, by, radA, radB));
            }

        // --- ステップ2: 空間バケットに登録 ---
        // バケットサイズ = Scale*2 px（格子間距離の半分程度）
        const int BucketPx = Scale * 2;
        var bucketCols = (imgW + BucketPx - 1) / BucketPx;
        var bucketRows = (imgH + BucketPx - 1) / BucketPx;
        // 各バケットが持つセグメントの index リスト
        var buckets = new List<int>[bucketRows, bucketCols];
        for (var br = 0; br < bucketRows; br++)
            for (var bc = 0; bc < bucketCols; bc++)
                buckets[br, bc] = [];

        for (var si = 0; si < segments.Count; si++)
        {
            var (ax, ay, bx, by, radA, radB) = segments[si];
            var maxRad = Math.Max(radA, radB);
            // このセグメントが影響しうるバケット範囲を登録
            var bx0 = (int)Math.Max(0, Math.Floor((Math.Min(ax, bx) - maxRad) / BucketPx));
            var bx1 = (int)Math.Min(bucketCols - 1, Math.Floor((Math.Max(ax, bx) + maxRad) / BucketPx));
            var by0 = (int)Math.Max(0, Math.Floor((Math.Min(ay, by) - maxRad) / BucketPx));
            var by1 = (int)Math.Min(bucketRows - 1, Math.Floor((Math.Max(ay, by) + maxRad) / BucketPx));
            for (var br = by0; br <= by1; br++)
                for (var bc = bx0; bc <= bx1; bc++)
                    buckets[br, bc].Add(si);
        }

        // --- ステップ3: ピクセルごとに最近傍線分を検索して塗る ---
        // 斜め方向セグメントがバケット境界をまたぐ場合に見落とさないよう
        // ピクセルのバケットとその3×3近傍 (最大9バケット) を検索する
        for (var py = 0; py < imgH; py++)
            for (var px = 0; px < imgW; px++)
            {
                var bRow = py / BucketPx;
                var bCol = px / BucketPx;

                // このピクセルのバケット内の全線分に対して距離判定
                var insideRiver = false;
                for (var dbr = -1; dbr <= 1 && !insideRiver; dbr++)
                    for (var dbc = -1; dbc <= 1 && !insideRiver; dbc++)
                    {
                        var nbr = bRow + dbr; var nbc = bCol + dbc;
                        if (nbr < 0 || nbr >= bucketRows || nbc < 0 || nbc >= bucketCols) continue;
                        foreach (var si in buckets[nbr, nbc])
                        {
                            var (ax, ay, bx, by, radA, radB) = segments[si];
                            var abx = bx - ax;
                            var aby = by - ay;
                            var ab2 = abx * abx + aby * aby;

                            double tParam;
                            double dist2;
                            if (ab2 < 1e-12)
                            {
                                // 点セグメント
                                var ddx = px - ax; var ddy = py - ay;
                                dist2 = ddx * ddx + ddy * ddy;
                                tParam = 0.0;
                            }
                            else
                            {
                                tParam = Math.Clamp(((px - ax) * abx + (py - ay) * aby) / ab2, 0.0, 1.0);
                                var qx = ax + tParam * abx - px;
                                var qy = ay + tParam * aby - py;
                                dist2 = qx * qx + qy * qy;
                            }

                            // 最近傍点での幅を線形補間 (A→B で radA→radB)
                            var interpRad = radA + tParam * (radB - radA);
                            if (interpRad < 1.0) continue; // 補間後も細すぎる部分はスキップ

                            if (dist2 <= interpRad * interpRad)
                            {
                                insideRiver = true;
                                break;
                            }
                        }
                    } // end 3x3 bucket loop

                if (insideRiver)
                    image[px, py] = new Rgb24(wr, wg, wb);
            }
    }
    else
    {
        // フォールバック: 旧来の円形ブラシ
        for (var r = 0; r < nrows; r++)
            for (var c = 0; c < ncols; c++)
            {
                var rad = riverRadius[r, c];
                if (rad <= 0.0) continue;
                var imgRow = nrows - 1 - r;
                var cx = c * Scale + Scale / 2;
                var cy = imgRow * Scale + Scale / 2;
                var iRad = (int)Math.Ceiling(rad);
                for (var dy = -iRad; dy <= iRad; dy++)
                    for (var dx2 = -iRad; dx2 <= iRad; dx2++)
                    {
                        if (dy * dy + dx2 * dx2 > rad * rad) continue;
                        var px2 = cx + dx2;
                        var py2 = cy + dy;
                        if (px2 < 0 || px2 >= ncols * Scale) continue;
                        if (py2 < 0 || py2 >= nrows * Scale) continue;
                        image[px2, py2] = new Rgb24(wr, wg, wb);
                    }
            }
    }

    image.SaveAsPng(filePath);
}

static void SaveTerrainPng(
    double[,] z,
    double seaLevel,
    string filePath,
    bool includeOceanLakeOverlay,
    double[]? drainageAreaFlat = null)
{
    const int Scale = 4; // 1セル → Scale×Scale ピクセル

    var nrows = z.GetLength(0);
    var ncols = z.GetLength(1);

    var dir = Path.GetDirectoryName(filePath);
    if (!string.IsNullOrWhiteSpace(dir))
        Directory.CreateDirectory(dir);

    // --- 標高の全体範囲 ---
    var zMin = double.PositiveInfinity;
    var zMax = double.NegativeInfinity;
    for (var r = 0; r < nrows; r++)
        for (var c = 0; c < ncols; c++)
        {
            if (z[r, c] < zMin) zMin = z[r, c];
            if (z[r, c] > zMax) zMax = z[r, c];
        }
    var zSpan = Math.Max(zMax - zMin, 1e-12);

    // --- ヒルシェーディング ---
    var hillshade = ComputeHillshade(z, Dx, azdegDeg: 315, altdegDeg: 45, vertExag: 2.0);

    // --- ocean / lake マスク ---
    var oceanMask = new bool[nrows, ncols];
    var lakeMask = new bool[nrows, ncols];
    if (includeOceanLakeOverlay)
        (oceanMask, lakeMask) = ClassifyOcean(z, seaLevel);

    // --- 川: 集水面積を対数正規化し各セルの「川半径」を算出 ---
    // 陸上の集水面積上位20%のセルのみ川として描画
    // 川の半径 (ピクセル単位) = 0.5 + 2.0 * norm  (norm: 0-1)
    var riverRadius = new double[nrows, ncols]; // 0 = 川なし
    if (drainageAreaFlat is not null && drainageAreaFlat.Length == nrows * ncols)
    {
        var posDrain = new List<double>();
        for (var rr = 0; rr < nrows; rr++)
            for (var cc = 0; cc < ncols; cc++)
            {
                if (z[rr, cc] <= seaLevel) continue;
                var a = drainageAreaFlat[rr * ncols + cc];
                if (a > 0.0) posDrain.Add(a);
            }
        if (posDrain.Count > 0)
        {
            var threshold = Percentile(posDrain, 80.0);
            var logMin = Math.Log(1.0 + threshold);
            var logMax = Math.Log(1.0 + posDrain.Max());
            var logRange = Math.Max(logMax - logMin, 1e-12);

            for (var rr = 0; rr < nrows; rr++)
                for (var cc = 0; cc < ncols; cc++)
                {
                    if (z[rr, cc] <= seaLevel) continue;
                    var a = drainageAreaFlat[rr * ncols + cc];
                    if (a < threshold) continue;
                    var norm = Math.Clamp((Math.Log(1.0 + a) - logMin) / logRange, 0.0, 1.0);
                    // Scale=4 のとき 0〜2.0 ピクセル半径 → 本流は直径 8px 相当
                    riverRadius[rr, cc] = 2.0 * norm * Scale;
                }
        }
    }

    // --- ベースレイヤー (1セル=1ピクセルで描画してからスケールアップ) ---
    using var image = new Image<Rgb24>(ncols * Scale, nrows * Scale);

    for (var rr = 0; rr < nrows; rr++)
    {
        for (var c = 0; c < ncols; c++)
        {
            var imgRow = nrows - 1 - rr; // origin='lower' で上下反転
            var hs = hillshade[rr, c];

            byte pr, pg, pb;

            if (includeOceanLakeOverlay && oceanMask[rr, c])
            {
                var d = 0.3 + 0.7 * hs;
                pr = (byte)(30 * d);
                pg = (byte)(144 * d);
                pb = (byte)(255 * d);
            }
            else if (includeOceanLakeOverlay && lakeMask[rr, c])
            {
                var d = 0.3 + 0.7 * hs;
                pr = (byte)(255 * d);
                pg = (byte)(99 * d);
                pb = (byte)(71 * d);
            }
            else
            {
                var t = (z[rr, c] - zMin) / zSpan;
                var (tr, tg, tb) = TerrainColormap(t);
                var br2 = tr / 255.0;
                var bg2 = tg / 255.0;
                var bb2 = tb / 255.0;
                pr = (byte)(Math.Clamp(OverlayBlend(br2, hs), 0.0, 1.0) * 255);
                pg = (byte)(Math.Clamp(OverlayBlend(bg2, hs), 0.0, 1.0) * 255);
                pb = (byte)(Math.Clamp(OverlayBlend(bb2, hs), 0.0, 1.0) * 255);
            }

            // Scale×Scale ブロックに書き込む
            for (var dy = 0; dy < Scale; dy++)
                for (var dx2 = 0; dx2 < Scale; dx2++)
                    image[c * Scale + dx2, imgRow * Scale + dy] = new Rgb24(pr, pg, pb);
        }
    }

    // --- 川レイヤー: 集水面積に応じた太さで円形ブラシ描画 ---
    // 色: Blues カラーマップ相当 (norm 0→薄水色, 1→濃紺)
    for (var rr = 0; rr < nrows; rr++)
    {
        for (var cc = 0; cc < ncols; cc++)
        {
            var rad = riverRadius[rr, cc];
            if (rad <= 0.0) continue;

            var norm = Math.Clamp((rad / Scale) / 2.0, 0.0, 1.0);

            // Blues カラーマップ: (0.35+0.65*norm) をR/G/Bに手動近似
            // 薄: (198, 219, 239) → 濃: (8, 48, 107)
            var cr = (byte)(198 + (8 - 198) * norm);
            var cg = (byte)(219 + (48 - 219) * norm);
            var cb = (byte)(239 + (107 - 239) * norm);
            const double alpha = 0.75;

            var imgRow = nrows - 1 - rr;
            var cx = cc * Scale + Scale / 2;
            var cy = imgRow * Scale + Scale / 2;
            var iRad = (int)Math.Ceiling(rad);

            for (var dy = -iRad; dy <= iRad; dy++)
                for (var dx2 = -iRad; dx2 <= iRad; dx2++)
                {
                    if (dy * dy + dx2 * dx2 > rad * rad) continue;
                    var px2 = cx + dx2;
                    var py2 = cy + dy;
                    if (px2 < 0 || px2 >= ncols * Scale) continue;
                    if (py2 < 0 || py2 >= nrows * Scale) continue;
                    var existing = image[px2, py2];
                    image[px2, py2] = new Rgb24(
                        (byte)(existing.R * (1 - alpha) + cr * alpha),
                        (byte)(existing.G * (1 - alpha) + cg * alpha),
                        (byte)(existing.B * (1 - alpha) + cb * alpha)
                    );
                }
        }
    }

    image.SaveAsPng(filePath);
}

static (double[] Px, double[] Py, double[] SedimentLoad) InitializeParticles(
    int nrows,
    int ncols,
    int particleCount,
    Random rng)
{
    var px = new double[particleCount];
    var py = new double[particleCount];
    var sedimentLoad = new double[particleCount];

    for (var i = 0; i < particleCount; i++)
    {
        px[i] = rng.NextDouble() * (ncols - 1);
        py[i] = rng.NextDouble() * (nrows - 1);
        sedimentLoad[i] = 0.0;
    }

    return (px, py, sedimentLoad);
}

static (double HighEnergyRatio, double MeanEnergy, double MeanLoad, double MeanSpeed) UpdateSeaParticles(
    double[,] z,
    double[] px,
    double[] py,
    double[] sedimentLoad,
    Random rng,
    double[]? drainageAreaFlat,
    double dx,
    double dt,
    double energySplitQuantile)
{
    var nrows = z.GetLength(0);
    var ncols = z.GetLength(1);
    var n = px.Length;

    var gx = GradientAxis1(z, dx);
    var gy = GradientAxis0(z, dx);

    var ix = new int[n];
    var iy = new int[n];
    var localGx = new double[n];
    var localGy = new double[n];
    var slopeMag = new double[n];
    var daNorm = new double[n];
    var energy = new double[n];
    var highMask = new bool[n];

    var daMax = 0.0;
    if (drainageAreaFlat is not null)
    {
        for (var i = 0; i < drainageAreaFlat.Length; i++)
        {
            if (drainageAreaFlat[i] > daMax)
            {
                daMax = drainageAreaFlat[i];
            }
        }
    }

    for (var i = 0; i < n; i++)
    {
        ix[i] = (int)Math.Clamp(Math.Round(px[i]), 0, ncols - 1);
        iy[i] = (int)Math.Clamp(Math.Round(py[i]), 0, nrows - 1);
        localGx[i] = gx[iy[i], ix[i]];
        localGy[i] = gy[iy[i], ix[i]];
        slopeMag[i] = Math.Sqrt(localGx[i] * localGx[i] + localGy[i] * localGy[i]);

        if (drainageAreaFlat is not null && daMax > 0.0)
        {
            var idx = iy[i] * ncols + ix[i];
            var localDa = Math.Max(drainageAreaFlat[idx], 0.0);
            daNorm[i] = Math.Log(1.0 + localDa) / Math.Log(1.0 + daMax);
        }
        else if (drainageAreaFlat is null)
        {
            daNorm[i] = 1.0;
        }
        else
        {
            daNorm[i] = 0.0;
        }

        energy[i] = slopeMag[i] * (0.35 + 0.65 * daNorm[i]);
    }

    var energyList = new List<double>(energy);
    var threshold = Percentile(energyList, energySplitQuantile);
    var highCount = 0;
    for (var i = 0; i < n; i++)
    {
        highMask[i] = energy[i] >= threshold;
        if (highMask[i])
        {
            highCount++;
        }
    }

    var lowCount = n - highCount;

    var slopeVx = new double[n];
    var slopeVy = new double[n];
    var tx = new double[n];
    var ty = new double[n];
    var vx = new double[n];
    var vy = new double[n];

    for (var i = 0; i < n; i++)
    {
        slopeVx[i] = -localGx[i];
        slopeVy[i] = -localGy[i];
        var sn = Math.Sqrt(slopeVx[i] * slopeVx[i] + slopeVy[i] * slopeVy[i]);
        if (sn > 1.0e-12)
        {
            slopeVx[i] /= sn;
            slopeVy[i] /= sn;
        }

        tx[i] = -localGy[i];
        ty[i] = localGx[i];
        var tn = Math.Sqrt(tx[i] * tx[i] + ty[i] * ty[i]);
        if (tn > 1.0e-12)
        {
            tx[i] /= tn;
            ty[i] /= tn;
        }
    }

    var energyMax = 1.0e-12;
    for (var i = 0; i < n; i++)
    {
        if (energy[i] > energyMax)
        {
            energyMax = energy[i];
        }
    }

    var energyNorm = new double[n];
    for (var i = 0; i < n; i++)
    {
        energyNorm[i] = energy[i] / energyMax;
    }

    for (var i = 0; i < n; i++)
    {
        if (highMask[i])
        {
            vx[i] = 0.55 * slopeVx[i] + NextGaussian(rng, 0.0, 0.05);
            vy[i] = 0.55 * slopeVy[i] + NextGaussian(rng, 0.0, 0.05);
            sedimentLoad[i] += (1.2e-5 * dt) * (0.8 + energyNorm[i]);
        }
        else
        {
            vx[i] = 0.10 * slopeVx[i] + 0.10 * tx[i] + NextGaussian(rng, 0.0, 0.12);
            vy[i] = 0.10 * slopeVy[i] + 0.10 * ty[i] + NextGaussian(rng, 0.0, 0.12);
            sedimentLoad[i] -= (0.9e-5 * dt) * (1.2 - energyNorm[i]);
        }

        sedimentLoad[i] = Math.Max(sedimentLoad[i], 0.0);
        px[i] += vx[i];
        py[i] += vy[i];

        if (!double.IsFinite(px[i]) || !double.IsFinite(py[i]))
        {
            throw new InvalidOperationException("Particle coordinates became non-finite");
        }

        px[i] = Math.Clamp(px[i], 0.0, ncols - 1.0001);
        py[i] = Math.Clamp(py[i], 0.0, nrows - 1.0001);
    }

    var meanEnergy = 0.0;
    var meanLoad = 0.0;
    var meanSpeed = 0.0;
    for (var i = 0; i < n; i++)
    {
        meanEnergy += energy[i];
        meanLoad += sedimentLoad[i];
        meanSpeed += Math.Sqrt(vx[i] * vx[i] + vy[i] * vy[i]);
    }

    return (
        HighEnergyRatio: (double)highCount / Math.Max(n, 1),
        MeanEnergy: meanEnergy / Math.Max(n, 1),
        MeanLoad: meanLoad / Math.Max(n, 1),
        MeanSpeed: meanSpeed / Math.Max(n, 1)
    );
}

static double[,] GradientAxis1(double[,] z, double dx)
{
    var nrows = z.GetLength(0);
    var ncols = z.GetLength(1);
    var g = new double[nrows, ncols];

    for (var r = 0; r < nrows; r++)
    {
        if (ncols > 1)
        {
            g[r, 0] = (z[r, 1] - z[r, 0]) / dx;
            g[r, ncols - 1] = (z[r, ncols - 1] - z[r, ncols - 2]) / dx;
        }
        for (var c = 1; c < ncols - 1; c++)
        {
            g[r, c] = (z[r, c + 1] - z[r, c - 1]) / (2.0 * dx);
        }
    }

    return g;
}

static double[,] GradientAxis0(double[,] z, double dx)
{
    var nrows = z.GetLength(0);
    var ncols = z.GetLength(1);
    var g = new double[nrows, ncols];

    for (var c = 0; c < ncols; c++)
    {
        if (nrows > 1)
        {
            g[0, c] = (z[1, c] - z[0, c]) / dx;
            g[nrows - 1, c] = (z[nrows - 1, c] - z[nrows - 2, c]) / dx;
        }
        for (var r = 1; r < nrows - 1; r++)
        {
            g[r, c] = (z[r + 1, c] - z[r - 1, c]) / (2.0 * dx);
        }
    }

    return g;
}

static double NextGaussian(Random rng, double mean, double stddev)
{
    var u1 = 1.0 - rng.NextDouble();
    var u2 = 1.0 - rng.NextDouble();
    var randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    return mean + stddev * randStdNormal;
}

static double[,] GenerateInitialTerrain(int nrows, int ncols, double dx, int seed, double reliefRatio)
{
    var terrain = new double[nrows, ncols];

    for (var octave = 0; octave < 7; octave++)
    {
        var freq = Math.Pow(2.0, octave);
        var amp = Math.Pow(0.5, octave);
        for (var i = 0; i < nrows; i++)
        {
            for (var j = 0; j < ncols; j++)
            {
                var x = i * freq / 80.0;
                var y = j * freq / 80.0;
                terrain[i, j] += amp * ValueNoise2D(x, y, seed + octave);
            }
        }
    }

    NormalizeInPlace(terrain);
    var domainLengthM = Math.Min(nrows, ncols) * dx;
    var targetReliefM = domainLengthM * reliefRatio;
    ScaleInPlace(terrain, targetReliefM);

    var borderVals = new List<double>((nrows + ncols) * 2);
    for (var c = 0; c < ncols; c++)
    {
        borderVals.Add(terrain[0, c]);
        borderVals.Add(terrain[nrows - 1, c]);
    }
    for (var r = 1; r < nrows - 1; r++)
    {
        borderVals.Add(terrain[r, 0]);
        borderVals.Add(terrain[r, ncols - 1]);
    }

    var shift = Percentile(borderVals, 20.0);
    for (var i = 0; i < nrows; i++)
    {
        for (var j = 0; j < ncols; j++)
        {
            terrain[i, j] -= shift;
        }
    }

    Console.WriteLine($"  [terrain] all-border p20 shift: -{shift:F1} m");
    return terrain;
}

static void CheckBoundaryAndAlert(double[,] z, int totalTime)
{
    var nrows = z.GetLength(0);
    var ncols = z.GetLength(1);

    var nonZeroCount = 0;
    var min = double.PositiveInfinity;
    var max = double.NegativeInfinity;
    var totalBoundary = 2 * nrows + 2 * ncols - 4;

    void Check(double value)
    {
        if (value != 0.0)
        {
            nonZeroCount++;
            if (value < min) min = value;
            if (value > max) max = value;
        }
    }

    for (var c = 0; c < ncols; c++)
    {
        Check(z[0, c]);
        Check(z[nrows - 1, c]);
    }
    for (var r = 1; r < nrows - 1; r++)
    {
        Check(z[r, 0]);
        Check(z[r, ncols - 1]);
    }

    if (nonZeroCount > 0)
    {
        Console.WriteLine(
            $"  [ALERT] t={totalTime / 1000.0:F0} kyr: 境界ノード {nonZeroCount}/{totalBoundary} が z!=0 " +
            $"(range: {min:F4} ~ {max:F4} m)");
    }
    else
    {
        Console.WriteLine($"  [OK] t={totalTime / 1000.0:F0} kyr: 全境界ノード z=0 ({totalBoundary})");
    }
}

static (bool[,] ocean, bool[,] lake) ClassifyOcean(double[,] z, double seaLevel)
{
    var nrows = z.GetLength(0);
    var ncols = z.GetLength(1);
    var below = new bool[nrows, ncols];
    var ocean = new bool[nrows, ncols];

    var hasWater = false;
    for (var r = 0; r < nrows; r++)
    {
        for (var c = 0; c < ncols; c++)
        {
            below[r, c] = z[r, c] <= seaLevel;
            hasWater |= below[r, c];
        }
    }

    if (!hasWater)
    {
        return (ocean, new bool[nrows, ncols]);
    }

    var queue = new Queue<(int r, int c)>();

    void EnqueueIfWater(int r, int c)
    {
        if (!below[r, c] || ocean[r, c])
        {
            return;
        }
        ocean[r, c] = true;
        queue.Enqueue((r, c));
    }

    for (var c = 0; c < ncols; c++)
    {
        EnqueueIfWater(0, c);
        EnqueueIfWater(nrows - 1, c);
    }
    for (var r = 1; r < nrows - 1; r++)
    {
        EnqueueIfWater(r, 0);
        EnqueueIfWater(r, ncols - 1);
    }

    var dr = new[] { -1, 1, 0, 0 };
    var dc = new[] { 0, 0, -1, 1 };
    while (queue.Count > 0)
    {
        var (r, c) = queue.Dequeue();
        for (var k = 0; k < 4; k++)
        {
            var nr = r + dr[k];
            var nc = c + dc[k];
            if (nr < 0 || nr >= nrows || nc < 0 || nc >= ncols)
            {
                continue;
            }
            EnqueueIfWater(nr, nc);
        }
    }

    var lake = new bool[nrows, ncols];
    for (var r = 0; r < nrows; r++)
    {
        for (var c = 0; c < ncols; c++)
        {
            lake[r, c] = below[r, c] && !ocean[r, c];
        }
    }

    return (ocean, lake);
}

static void SetBoundaryToZero(double[,] z)
{
    var nrows = z.GetLength(0);
    var ncols = z.GetLength(1);
    for (var c = 0; c < ncols; c++)
    {
        z[0, c] = 0.0;
        z[nrows - 1, c] = 0.0;
    }
    for (var r = 1; r < nrows - 1; r++)
    {
        z[r, 0] = 0.0;
        z[r, ncols - 1] = 0.0;
    }
}

// ===== 実験的: 海岸侵食 =====
// seaLevel ± CoastalBeltWidth の標高帯にある陸セルの岩盤を侵食する
// 波の打ち上げ・打ち下げで海岸線付近が削られるシンプルなモデル
// erosionRate [m/yr] × dt [yr] だけ bedrock を下げる
static void ApplyCoastalErosion(
    double[,] bedrock, double[,] soil,
    double seaLevel, double beltWidth, double erosionRate, int dt)
{
    var nrows = bedrock.GetLength(0);
    var ncols = bedrock.GetLength(1);
    var erosionAmount = erosionRate * dt;
    for (var r = 1; r < nrows - 1; r++)
    {
        for (var c = 1; c < ncols - 1; c++)
        {
            var elev = bedrock[r, c] + soil[r, c];
            // 海面より上、かつ海面から beltWidth 以内のセルを海岸帯とみなす
            if (elev <= seaLevel) continue;
            if (elev > seaLevel + beltWidth) continue;

            // 海岸帯内での相対位置: 0=sea level, 1=belt top → 海面に近いほど強く侵食
            var relPos = (elev - seaLevel) / beltWidth; // 0..1
            var factor = 1.0 - relPos; // 海面に近い=1.0、帯上端=0.0
            var cut = erosionAmount * factor;

            // 土層から先に削り、足りなければ岩盤を削る
            var soilCut = Math.Min(soil[r, c], cut);
            soil[r, c] -= soilCut;
            var remaining = cut - soilCut;
            if (remaining > 0.0)
                bedrock[r, c] -= remaining;
        }
    }
}

// 境界セルを内険セルの地表標高 (bedrock+soil) の下位percentile%に設定する
// seaLevel=0 以下には追従させない（海の底に潜らない）
static double SetBoundaryToInlandPercentile(double[,] bedrock, double[,] soil, double percentile)
{
    var nrows = bedrock.GetLength(0);
    var ncols = bedrock.GetLength(1);

    // 内険セル (=境界以外) の地表標高を収集
    var inlandZ = new List<double>((nrows - 2) * (ncols - 2));
    for (var r = 1; r < nrows - 1; r++)
        for (var c = 1; c < ncols - 1; c++)
            inlandZ.Add(bedrock[r, c] + Math.Max(soil[r, c], 0.0));

    if (inlandZ.Count == 0) return 0.0;

    var target = Percentile(inlandZ, percentile);
    // 少なくとも0以上（海水面以下には追従しない）
    target = Math.Max(target, 0.0);

    for (var c = 0; c < ncols; c++)
    {
        bedrock[0, c] = target;
        bedrock[nrows - 1, c] = target;
    }
    for (var r = 1; r < nrows - 1; r++)
    {
        bedrock[r, 0] = target;
        bedrock[r, ncols - 1] = target;
    }
    return target;
}

static void AddUniformUplift(double[,] z, double delta)
{
    var nrows = z.GetLength(0);
    var ncols = z.GetLength(1);
    for (var r = 0; r < nrows; r++)
    {
        for (var c = 0; c < ncols; c++)
        {
            z[r, c] += delta;
        }
    }
}

// ============================================================
// Priority-Flood 窪地充填 (Barnes 2014)
// ============================================================
static void FillDepressions(double[] z, int nrows, int ncols)
{
    // 境界ノードはすでに流出口として機能するため、境界値を起点に
    // Priority Queue で内側へ伝播させ、すべてのノードを
    // 「流出できる最小水位」まで引き上げる。
    const double EPS = 1e-7; // 僅かな勾配を保証するイプシロン
    var n = nrows * ncols;
    var processed = new bool[n];

    // MinHeap: (elevation, index)
    var pq = new SortedSet<(double elev, int idx)>(
        Comparer<(double, int)>.Create((a, b) =>
            a.Item1 != b.Item1 ? a.Item1.CompareTo(b.Item1) : a.Item2.CompareTo(b.Item2)
        )
    );

    // 境界ノードをキューに積む
    for (var r = 0; r < nrows; r++)
    {
        for (var c = 0; c < ncols; c++)
        {
            if (r == 0 || r == nrows - 1 || c == 0 || c == ncols - 1)
            {
                var idx = r * ncols + c;
                pq.Add((z[idx], idx));
                processed[idx] = true;
            }
        }
    }

    var dr8 = new[] { -1, -1, -1, 0, 0, 1, 1, 1 };
    var dc8 = new[] { -1, 0, 1, -1, 1, -1, 0, 1 };

    while (pq.Count > 0)
    {
        var min = pq.Min;
        pq.Remove(min);
        var (elev, node) = min;
        var r = node / ncols;
        var c = node % ncols;

        for (var k = 0; k < 8; k++)
        {
            var nr = r + dr8[k];
            var nc = c + dc8[k];
            if (nr < 0 || nr >= nrows || nc < 0 || nc >= ncols) continue;
            var nb = nr * ncols + nc;
            if (processed[nb]) continue;
            processed[nb] = true;
            // 隣接ノードが現在のウォーターレベルより低ければ引き上げ
            if (z[nb] <= elev)
                z[nb] = elev + EPS;
            pq.Add((z[nb], nb));
        }
    }
}

// ============================================================
// LinearDiffuser (斜面拡散)
// dz/dt = K_hs * nabla^2 z  を前進差分で解く
// Landlab のLinearDiffuserはflux-divergence法だが、
// ラスタ格子では laplacian と等価
// ============================================================
static void RunLinearDiffusion(double[,] z, int nrows, int ncols, double dx, double kHs, double dt)
{
    var dz = new double[nrows, ncols];
    var invDx2 = 1.0 / (dx * dx);

    for (var r = 1; r < nrows - 1; r++)
    {
        for (var c = 1; c < ncols - 1; c++)
        {
            // 4近傍ラプラシアン（Landlab の flux-div と同等）
            var lap = (z[r - 1, c] + z[r + 1, c] + z[r, c - 1] + z[r, c + 1]
                       - 4.0 * z[r, c]) * invDx2;
            dz[r, c] = kHs * lap * dt;
        }
    }

    for (var r = 1; r < nrows - 1; r++)
        for (var c = 1; c < ncols - 1; c++)
            z[r, c] += dz[r, c];
    // 境界は変えない（SetBoundaryToZero は外側で呼ぶ）
}

// ============================================================
// SpaceLargeScaleEroder
// Landlab の calc_sequential_ero_depo.pyx を C# 移植
// ============================================================
static void RunSpaceLargeScaleStep(
    double[,] bedrock,
    double[,] soil,
    double[] area,
    double[] slope,
    int[] receiver,
    int[] order,
    double dt,
    double kBr,
    double kSed,
    double mSp,
    double nSp,
    double phi,
    double hStar,
    double vS,
    double fF,
    double thicknessLim,
    double dx)
{
    var nrows = bedrock.GetLength(0);
    var ncols = bedrock.GetLength(1);
    var n = nrows * ncols;
    var cellArea = dx * dx;

    // qs[i] = 下流へ出て行く堆積物フラックス (m^3/yr)
    // qsIn[i] = 上流から入ってくる堆積物フラックス (m^3/yr)
    var qs = new double[n];
    var qsIn = new double[n];

    // 水の単位流量 q_w[i] = drainage_area (m^2)  (rainfall rate = 1 m/yr と仮定)
    // Landlab の q = surface_water__discharge = drainage_area (デフォルト)
    // ※ FlowAccumulator デフォルトは rainfall=1 なので discharge=area
    var qw = area; // alias

    // --- 1. 侵食率の事前計算 (Es, Er) ---
    var Es = new double[n];
    var Er = new double[n];
    var sedErosionTerm = new double[n]; // omega_sed (K_sed * Q^m * S^n / (1-phi) など)
    var brErosionTerm = new double[n];

    for (var node = 0; node < n; node++)
    {
        var r = node / ncols;
        var c = node % ncols;
        // 境界ノードはスキップ
        if (r == 0 || r == nrows - 1 || c == 0 || c == ncols - 1) continue;

        var s = Math.Max(slope[node], 0.0);
        if (s <= 0.0) continue;
        var q = Math.Max(qw[node], 1.0); // 最小値ガード

        var qToM = Math.Pow(q, mSp);
        var sToN = nSp == 1.0 ? s : Math.Pow(s, nSp);

        var omegaSed = kSed * qToM * sToN;
        var omegaBr = kBr * qToM * sToN;

        var H = Math.Max(soil[r, c], 0.0);
        // 土層による基盤岩被覆率: cover = exp(-H/H*)
        var cover = H > thicknessLim ? 0.0 : Math.Exp(-H / Math.Max(hStar, 1e-12));

        // spCrit = 0 なので簡略化
        sedErosionTerm[node] = omegaSed; // /(1-phi) はあとで適用
        brErosionTerm[node] = omegaBr;
        Es[node] = omegaSed * (1.0 - cover);
        Er[node] = omegaBr * cover;
    }

    // --- 2. 上流→下流順に堆積物フラックスを計算 (pyx の逐次計算を再現) ---
    // order[] はKahn法の結果: order[0] = 最上流、order[n-1] = 最下流
    for (var p = 0; p < n; p++)
    {
        var node = order[p];
        var r = node / ncols;
        var c = node % ncols;
        // 境界ノードはスキップ（qs_in は受け取るが更新しない）
        if (r == 0 || r == nrows - 1 || c == 0 || c == ncols - 1) continue;

        var s = Math.Max(slope[node], 0.0);
        var q = Math.Max(qw[node], 1.0);
        var sedEroLoc = sedErosionTerm[node];
        var brEroLoc = brErosionTerm[node];
        var EsLoc = Es[node];
        var ErLoc = Er[node];
        var H = Math.Max(soil[r, c], 0.0);
        var HBefore = H;

        // qs_out の計算 (pyx L43-45):
        // qs_out = (qs_in + Es * cellArea + (1-F_f) * Er * cellArea)
        //          / (1 + v_s * cellArea / q_w)
        var qsOut = (qsIn[node]
                     + EsLoc * cellArea
                     + (1.0 - fF) * ErLoc * cellArea)
                    / (1.0 + vS * cellArea / q);

        var depoRate = vS * qsOut / q; // [m/yr] 堆積率

        // --- 土層厚の解析解 (pyx L56-88) ---
        double HNew;
        if (H > thicknessLim || s <= 0.0 || sedEroLoc == 0.0)
        {
            // 線形近似 (厚すぎる or 勾配ゼロ or 侵食ゼロ)
            HNew = H + (depoRate / (1.0 - phi) - sedEroLoc / (1.0 - phi)) * dt;
        }
        else
        {
            // Blowup 判定: depo_rate == K_sed * Q^m * S^n
            var kSedOmega = sedEroLoc; // = omega_sed = K_sed * Q^m * S^n
            if (depoRate == kSedOmega)
            {
                // Eq 34 (blowup case): H_new = H* * ln(D/H* * dt + exp(H/H*))
                HNew = hStar * Math.Log(
                    (sedEroLoc / (1.0 - phi)) / hStar * dt
                    + Math.Exp(H / hStar)
                );
            }
            else
            {
                // Eq 32 (general case, pyx L68-86)
                var A_ = depoRate / (1.0 - phi);
                var B_ = sedEroLoc / (1.0 - phi);
                var inner = (1.0 / (A_ / B_ - 1.0))
                             * (Math.Exp((A_ - B_) * (dt / hStar))
                                * ((A_ / B_ - 1.0) * Math.Exp(H / hStar) + 1.0)
                                - 1.0);
                if (inner <= 0.0)
                {
                    // フォールバック: 線形近似
                    HNew = H + (A_ - B_) * dt;
                }
                else
                {
                    HNew = hStar * Math.Log(inner);
                }
            }
            if (double.IsInfinity(HNew))
                HNew = H + (depoRate / (1.0 - phi) - sedEroLoc / (1.0 - phi)) * dt;
        }

        HNew = Math.Max(HNew, 0.0);

        // 基盤岩侵食 (pyx L93-95): ero_bed = br_erosion_term * exp(-H_new / H*)
        var coverNew = HNew > thicknessLim ? 0.0 : Math.Exp(-HNew / Math.Max(hStar, 1e-12));
        var eroBed = brEroLoc * coverNew;

        // 出力フラックス調整 (pyx L97-100): 質量保存
        var qsOutAdj = qsIn[node]
                       - ((HNew - HBefore) * (1.0 - phi) * cellArea / dt)
                       + (1.0 - fF) * eroBed * cellArea;
        qsOutAdj = Math.Max(qsOutAdj, 0.0);

        qs[node] = qsOutAdj;

        // 下流ノードへ流量を渡す
        var rec = receiver[node];
        if (rec != node)
            qsIn[rec] += qsOutAdj;

        // 地形更新
        soil[r, c] = HNew;
        bedrock[r, c] -= eroBed * dt;

        // 有限値チェック
        if (!double.IsFinite(bedrock[r, c]) || !double.IsFinite(soil[r, c]))
            throw new InvalidOperationException(
                $"Non-finite value in SPACE step at ({r},{c}): br={bedrock[r, c]}, H={soil[r, c]}"
            );
    }
}

static (int[] Receiver, double[] Slope, double[] Area, int[] Order) ComputeD8Flow(double[,] z, double dx)
{
    var nrows = z.GetLength(0);
    var ncols = z.GetLength(1);
    var n = nrows * ncols;

    var receiver = new int[n];
    var slope = new double[n];
    var donorCount = new int[n];
    var area = new double[n];
    var order = new int[n];

    for (var i = 0; i < n; i++)
    {
        receiver[i] = i;
        slope[i] = 0.0;
        area[i] = dx * dx;
    }

    var dr = new[] { -1, -1, -1, 0, 0, 1, 1, 1 };
    var dc = new[] { -1, 0, 1, -1, 1, -1, 0, 1 };

    for (var r = 1; r < nrows - 1; r++)
    {
        for (var c = 1; c < ncols - 1; c++)
        {
            var idx = r * ncols + c;
            var z0 = z[r, c];
            var bestSlope = 0.0;
            var bestReceiver = idx;

            for (var k = 0; k < 8; k++)
            {
                var nr = r + dr[k];
                var nc = c + dc[k];
                var dist = (dr[k] == 0 || dc[k] == 0) ? dx : dx * Math.Sqrt(2.0);
                var s = (z0 - z[nr, nc]) / dist;
                if (s > bestSlope)
                {
                    bestSlope = s;
                    bestReceiver = nr * ncols + nc;
                }
            }

            receiver[idx] = bestReceiver;
            slope[idx] = Math.Max(bestSlope, 0.0);
        }
    }

    for (var i = 0; i < n; i++)
    {
        if (receiver[i] != i)
        {
            donorCount[receiver[i]]++;
        }
    }

    var q = new Queue<int>();
    for (var i = 0; i < n; i++)
    {
        if (donorCount[i] == 0)
        {
            q.Enqueue(i);
        }
    }

    var ptr = 0;
    while (q.Count > 0)
    {
        var node = q.Dequeue();
        order[ptr++] = node;
        var rec = receiver[node];
        if (rec != node)
        {
            area[rec] += area[node];
            donorCount[rec]--;
            if (donorCount[rec] == 0)
            {
                q.Enqueue(rec);
            }
        }
    }

    if (ptr != n)
    {
        throw new InvalidOperationException("Flow topology ordering failed");
    }

    return (receiver, slope, area, order);
}

static double ValueNoise2D(double x, double y, int seed)
{
    var x0 = (int)Math.Floor(x);
    var y0 = (int)Math.Floor(y);
    var x1 = x0 + 1;
    var y1 = y0 + 1;

    var sx = SmoothStep(x - x0);
    var sy = SmoothStep(y - y0);

    var n00 = HashToUnit(x0, y0, seed);
    var n10 = HashToUnit(x1, y0, seed);
    var n01 = HashToUnit(x0, y1, seed);
    var n11 = HashToUnit(x1, y1, seed);

    var ix0 = Lerp(n00, n10, sx);
    var ix1 = Lerp(n01, n11, sx);
    var v = Lerp(ix0, ix1, sy);
    return v * 2.0 - 1.0;
}

static double HashToUnit(int x, int y, int seed)
{
    unchecked
    {
        uint h = (uint)seed;
        h = (h * 374761393u) ^ ((uint)x * 668265263u);
        h = (h * 1274126177u) ^ ((uint)y * 2246822519u);
        h ^= h >> 13;
        h *= 1274126177u;
        h ^= h >> 16;
        var positive = h & 0x7fffffffu;
        return positive / (double)int.MaxValue;
    }
}

static double SmoothStep(double t)
{
    return t * t * (3.0 - 2.0 * t);
}

static double Lerp(double a, double b, double t)
{
    return a + (b - a) * t;
}

static void NormalizeInPlace(double[,] data)
{
    var nrows = data.GetLength(0);
    var ncols = data.GetLength(1);
    var min = double.PositiveInfinity;
    var max = double.NegativeInfinity;

    for (var r = 0; r < nrows; r++)
    {
        for (var c = 0; c < ncols; c++)
        {
            var v = data[r, c];
            if (v < min) min = v;
            if (v > max) max = v;
        }
    }

    var span = max - min;
    if (span <= 1e-12)
    {
        for (var r = 0; r < nrows; r++)
        {
            for (var c = 0; c < ncols; c++)
            {
                data[r, c] = 0.0;
            }
        }
        return;
    }

    for (var r = 0; r < nrows; r++)
    {
        for (var c = 0; c < ncols; c++)
        {
            data[r, c] = (data[r, c] - min) / span;
        }
    }
}

static void ScaleInPlace(double[,] data, double scale)
{
    var nrows = data.GetLength(0);
    var ncols = data.GetLength(1);
    for (var r = 0; r < nrows; r++)
    {
        for (var c = 0; c < ncols; c++)
        {
            data[r, c] *= scale;
        }
    }
}

static double Percentile(List<double> values, double p)
{
    if (values.Count == 0)
    {
        throw new InvalidOperationException("Percentile requires non-empty values");
    }
    values.Sort();
    var rank = (p / 100.0) * (values.Count - 1);
    var lo = (int)Math.Floor(rank);
    var hi = (int)Math.Ceiling(rank);
    if (lo == hi)
    {
        return values[lo];
    }
    var t = rank - lo;
    return Lerp(values[lo], values[hi], t);
}

static double Percentage(bool[,] mask)
{
    var nrows = mask.GetLength(0);
    var ncols = mask.GetLength(1);
    var count = 0;
    for (var r = 0; r < nrows; r++)
    {
        for (var c = 0; c < ncols; c++)
        {
            if (mask[r, c])
            {
                count++;
            }
        }
    }
    return 100.0 * count / (nrows * ncols);
}

static double[,] SumFields(double[,] a, double[,] b)
{
    var nrows = a.GetLength(0);
    var ncols = a.GetLength(1);
    var outField = new double[nrows, ncols];
    for (var r = 0; r < nrows; r++)
    {
        for (var c = 0; c < ncols; c++)
        {
            outField[r, c] = a[r, c] + b[r, c];
        }
    }
    return outField;
}

static double MeanField(double[,] field)
{
    var nrows = field.GetLength(0);
    var ncols = field.GetLength(1);
    var sum = 0.0;
    for (var r = 0; r < nrows; r++)
    {
        for (var c = 0; c < ncols; c++)
        {
            sum += field[r, c];
        }
    }
    return sum / (nrows * ncols);
}

static double MaxField(double[,] field)
{
    var nrows = field.GetLength(0);
    var ncols = field.GetLength(1);
    var max = double.NegativeInfinity;
    for (var r = 0; r < nrows; r++)
    {
        for (var c = 0; c < ncols; c++)
        {
            if (field[r, c] > max)
            {
                max = field[r, c];
            }
        }
    }
    return max;
}
