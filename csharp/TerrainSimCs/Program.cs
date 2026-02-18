using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

const int Nrows = 150;
const int Ncols = 150;
const double Dx = 50.0;
const double UpliftRate = 0.001;
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
const int Tmax = 50000;
const int HistoryInterval = 10000;
const double SeaLevel = 0.0;
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
    SaveTerrainPng(
        z,
        SeaLevel,
        Path.Combine(historyDir, "terrain_0000kyr.png"),
        includeOceanLakeOverlay: true,
        drainageAreaFlat: flow0.Area
    );

    var (ocean0, lake0) = ClassifyOcean(z, SeaLevel);
    Console.WriteLine($"[t=0] Ocean={Percentage(ocean0):F1}% Lake={Percentage(lake0):F1}%");

    var totalTime = 0;
    while (totalTime < Tmax)
    {
        AddUniformUplift(bedrock, UpliftRate * Dt);
        SetBoundaryToZero(bedrock);

        // 1. 隆起後の地表標高を再計算
        var zNow = SumFields(bedrock, soil);

        // 2. 斜面拡散 (LinearDiffuser相当)
        RunLinearDiffusion(zNow, Nrows, Ncols, Dx, KHs, Dt);
        // LinearDiffuser は z を直接更新するので bedrock を同期
        for (var r = 0; r < Nrows; r++)
            for (var c = 0; c < Ncols; c++)
                bedrock[r, c] = zNow[r, c] - Math.Max(soil[r, c], 0.0);
        SetBoundaryToZero(bedrock);
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

        SetBoundaryToZero(bedrock);
        SetBoundaryToZero(soil);
        z = SumFields(bedrock, soil);

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
            var (ocean, lake) = ClassifyOcean(z, SeaLevel);
            var outPath = Path.Combine(historyDir, $"terrain_{totalTime / 1000:0000}kyr.png");
            var flowViz = ComputeD8Flow(z, Dx);
            SaveTerrainPng(
                z,
                SeaLevel,
                outPath,
                includeOceanLakeOverlay: true,
                drainageAreaFlat: flowViz.Area
            );
            Console.WriteLine(
                $"  [history] {totalTime / 1000.0:F0} kyr: Ocean={Percentage(ocean):F1}% Lake={Percentage(lake):F1}% -> {Path.GetFileName(outPath)}");
        }
    }

    Console.WriteLine("Simulation complete.");
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
    var az  = (360.0 - azdegDeg + 90.0) * Math.PI / 180.0; // 北→東→南→西
    var alt = altdegDeg * Math.PI / 180.0;
    var lx = Math.Cos(alt) * Math.Cos(az);
    var ly = Math.Cos(alt) * Math.Sin(az);
    var lz = Math.Sin(alt);

    for (var r = 0; r < nrows; r++)
    {
        for (var c = 0; c < ncols; c++)
        {
            // 中央差分で法線ベクトルを計算 (vert_exag を掛けた高さで)
            var dzdx = r > 0 && r < nrows - 1
                ? (z[r + 1, c] - z[r - 1, c]) * vertExag / (2.0 * dx)
                : 0.0;
            var dzdy = c > 0 && c < ncols - 1
                ? (z[r, c + 1] - z[r, c - 1]) * vertExag / (2.0 * dx)
                : 0.0;
            // 法線 n = (-dz/dx, -dz/dy, 1) を正規化
            var nx = -dzdx; var ny = -dzdy; var nz = 1.0;
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

static void SaveTerrainPng(
    double[,] z,
    double seaLevel,
    string filePath,
    bool includeOceanLakeOverlay,
    double[]? drainageAreaFlat = null)
{
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
    var lakeMask  = new bool[nrows, ncols];
    if (includeOceanLakeOverlay)
        (oceanMask, lakeMask) = ClassifyOcean(z, seaLevel);

    // --- 川しきい値 ---
    var riverLevel1 = double.PositiveInfinity;
    var riverLevel2 = double.PositiveInfinity;
    var riverLevel3 = double.PositiveInfinity;
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
            riverLevel1 = Percentile(posDrain, 80.0);
            riverLevel2 = Percentile(posDrain, 90.0);
            riverLevel3 = Percentile(posDrain, 95.0);
        }
    }

    using var image = new Image<Rgb24>(ncols, nrows);

    for (var rr = nrows - 1; rr >= 0; rr--)
    {
        for (var c = 0; c < ncols; c++)
        {
            // 画像の行: origin='lower' に合わせて上下反転
            var imgRow = nrows - 1 - rr;
            var hs = hillshade[rr, c];

            byte pr, pg, pb;

            if (includeOceanLakeOverlay && oceanMask[rr, c])
            {
                // 海: dodgerblue ベースにヒルシェード
                var d = 0.3 + 0.7 * hs; // 暗くしすぎない
                pr = (byte)(30  * d);
                pg = (byte)(144 * d);
                pb = (byte)(255 * d);
            }
            else if (includeOceanLakeOverlay && lakeMask[rr, c])
            {
                // 湖: tomato
                var d = 0.3 + 0.7 * hs;
                pr = (byte)(255 * d);
                pg = (byte)(99  * d);
                pb = (byte)(71  * d);
            }
            else
            {
                // 陸: terrain カラーマップ + overlay hillshade
                var t = (z[rr, c] - zMin) / zSpan;
                var (tr, tg, tb) = TerrainColormap(t);

                // overlay blend per channel
                var br2 = tr / 255.0;
                var bg2 = tg / 255.0;
                var bb2 = tb / 255.0;
                var blendR = OverlayBlend(br2, hs);
                var blendG = OverlayBlend(bg2, hs);
                var blendB = OverlayBlend(bb2, hs);
                pr = (byte)(Math.Clamp(blendR, 0.0, 1.0) * 255);
                pg = (byte)(Math.Clamp(blendG, 0.0, 1.0) * 255);
                pb = (byte)(Math.Clamp(blendB, 0.0, 1.0) * 255);

                // 川オーバーレイ (シアン, 半透明風)
                if (drainageAreaFlat is not null && z[rr, c] > seaLevel)
                {
                    var a = drainageAreaFlat[rr * ncols + c];
                    double alpha = 0.0;
                    (byte cr, byte cg, byte cb) riverCol = (0, 120, 255);
                    if (a >= riverLevel3)      { alpha = 0.75; riverCol = (0, 120, 255); }
                    else if (a >= riverLevel2) { alpha = 0.65; riverCol = (40, 170, 255); }
                    else if (a >= riverLevel1) { alpha = 0.50; riverCol = (90, 210, 255); }
                    if (alpha > 0.0)
                    {
                        pr = (byte)(pr * (1 - alpha) + riverCol.cr * alpha);
                        pg = (byte)(pg * (1 - alpha) + riverCol.cg * alpha);
                        pb = (byte)(pb * (1 - alpha) + riverCol.cb * alpha);
                    }
                }
            }

            image[c, imgRow] = new Rgb24(pr, pg, pb);
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
