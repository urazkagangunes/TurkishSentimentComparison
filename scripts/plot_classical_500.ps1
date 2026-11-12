param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

# Uses the Windows charting library; no Python dependency/model changes.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
$analysisPath = Join-Path $OutputDirectory 'analysis.json'
$analysis = Get-Content -LiteralPath $analysisPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $analysis.complete) { throw 'Plots require a complete four-system evaluation.' }

$displayNames = @{
    hisnet = 'HisNet'
    dp_hisnet = 'DP -> HisNet'
    hisnet_dp = 'HisNet -> DP'
    lw_hisnet_dp = 'LW HisNet -> DP'
}

function Save-ComparisonChart {
    param([string]$Filename, [string]$Title, [string]$AxisTitle, [array]$Fields, [array]$SeriesNames, [double]$Scale, [double]$Maximum)
    $chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    try {
        $chart.Width = 1500
        $chart.Height = 850
        $chart.BackColor = [System.Drawing.Color]::White
        $area = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea
        $area.AxisY.Title = $AxisTitle
        $area.AxisY.Minimum = 0
        if ($Maximum -gt 0) { $area.AxisY.Maximum = $Maximum }
        $area.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
        $area.AxisX.MajorGrid.Enabled = $false
        $area.AxisX.Interval = 1
        $area.AxisX.LabelStyle.Font = New-Object System.Drawing.Font('Segoe UI', 14)
        $area.AxisY.LabelStyle.Font = New-Object System.Drawing.Font('Segoe UI', 12)
        $area.AxisY.TitleFont = New-Object System.Drawing.Font('Segoe UI', 13)
        $chart.ChartAreas.Add($area)
        $legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend
        $legend.Docking = 'Bottom'
        $legend.Font = New-Object System.Drawing.Font('Segoe UI', 13)
        $chart.Legends.Add($legend)
        $chartTitle = $chart.Titles.Add($Title)
        $chartTitle.Font = New-Object System.Drawing.Font('Segoe UI', 21, [System.Drawing.FontStyle]::Bold)
        $colors = @('#167D9A', '#DF7B31', '#7765AA')
        for ($column = 0; $column -lt $Fields.Length; $column++) {
            $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series($SeriesNames[$column])
            $series.ChartType = 'Column'
            $series.Color = [System.Drawing.ColorTranslator]::FromHtml($colors[$column])
            $series.IsValueShownAsLabel = $true
            $series.LabelFormat = '0.00'
            $series.Font = New-Object System.Drawing.Font('Segoe UI', 12)
            foreach ($metric in $analysis.metrics) {
                $value = [double]$metric.($Fields[$column]) * $Scale
                $null = $series.Points.AddXY($displayNames[$metric.system], $value)
            }
            $chart.Series.Add($series)
        }
        $chart.SaveImage((Join-Path $OutputDirectory $Filename), 'Png')
    }
    finally {
        $chart.Dispose()
    }
}

Save-ComparisonChart -Filename 'performance_500.png' -Title 'TRSAv1 | 500 fixed reviews | Four classical systems' -AxisTitle 'Score (%)' -Fields @('accuracy', 'macro_f1') -SeriesNames @('Accuracy', 'Macro F1') -Scale 100 -Maximum 100
Save-ComparisonChart -Filename 'runtime_500.png' -Title 'Independent pipelines | Local CPU classification time' -AxisTitle 'Minutes (initialization and disk writes excluded)' -Fields @('classify_seconds') -SeriesNames @('Classification time') -Scale (1.0 / 60.0) -Maximum 0
Write-Output 'Saved performance_500.png and runtime_500.png'
