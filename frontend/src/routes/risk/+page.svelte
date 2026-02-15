<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import Card from '$lib/components/ui/Card.svelte';
	import MetricCard from '$lib/components/ui/MetricCard.svelte';
	import HeatmapChart from '$lib/components/charts/HeatmapChart.svelte';
	import LineChart from '$lib/components/charts/LineChart.svelte';
	import GaugeChart from '$lib/components/charts/GaugeChart.svelte';
	import TreemapChart from '$lib/components/charts/TreemapChart.svelte';
	import BarChart from '$lib/components/charts/BarChart.svelte';
	import SectorRotationHeatmap from '$lib/components/charts/SectorRotationHeatmap.svelte';
	import RiskStatusBadge from '$lib/components/ui/RiskStatusBadge.svelte';
	import DataTable from '$lib/components/ui/DataTable.svelte';
	import { risk, correlation, sectorRotation, sectorAttribution } from '$lib/stores/dashboard';
	import { api } from '$lib/api/client';
	import { formatPercent, formatDateShort } from '$lib/utils/format';
	import type { RiskReportResponse, SectorContributionDetail } from '$lib/types/api';

	$: riskReport = $risk;
	$: correlationData = $correlation;
	$: sectorRotationData = $sectorRotation;
	$: sectorAttributionData = $sectorAttribution;

	let riskHistory: RiskReportResponse[] = [];
	let loading = true;

	async function loadData() {
		loading = true;
		try {
			await Promise.all([
				risk.fetch(),
				correlation.fetch(),
				sectorRotation.fetch(),
				sectorAttribution.fetch()
			]);
			const historyData = await api.getRiskHistory(30);
			riskHistory = historyData.history;
		} catch (error) {
			console.error('Failed to load risk data:', error);
		} finally {
			loading = false;
		}
	}

	// Fetch data on mount
	onMount(() => {
		loadData();
	});

	// Refetch when route changes (handles tab switching)
	$: if ($page.url.pathname === '/risk') {
		loadData();
	}

	// Sharpe ratio history
	$: sharpeData = riskHistory
		.slice()
		.reverse()
		.filter(r => r.sharpe_ratio != null)
		.map(r => ({
			time: formatDateShort(r.timestamp),
			value: r.sharpe_ratio
		}));

	// Volatility history
	$: volatilityData = riskHistory
		.slice()
		.reverse()
		.filter(r => r.portfolio_volatility != null)
		.map(r => ({
			time: formatDateShort(r.timestamp),
			value: r.portfolio_volatility
		}));

	// Sector attribution data transformations
	$: sectorAllocationData = sectorAttributionData?.contributions
		.filter(c => c.total_value > 0)
		.map(c => ({
			name: c.sector,
			value: c.total_value
		})) || [];

	$: sectorWeightDeltaData = sectorAttributionData?.contributions
		.map(c => ({
			label: c.sector_etf,
			value: c.over_under_weight * 100,
			color: c.over_under_weight > 0 ? '#10b981' : '#ef4444'
		}))
		.sort((a, b) => Math.abs(b.value) - Math.abs(a.value)) || [];

	const sectorContributionColumns: Array<{
		key: keyof SectorContributionDetail;
		label: string;
		format?: (v: any, row: SectorContributionDetail) => string;
		class?: string;
		cellClass?: (v: any, row: SectorContributionDetail) => string;
	}> = [
		{ key: 'sector', label: 'Sector', class: 'font-medium' },
		{ key: 'position_count', label: 'Positions' },
		{
			key: 'portfolio_weight',
			label: 'Portfolio %',
			format: (v: number) => (v * 100).toFixed(2) + '%'
		},
		{
			key: 'benchmark_weight',
			label: 'SPY %',
			format: (v: number) => (v * 100).toFixed(2) + '%'
		},
		{
			key: 'over_under_weight',
			label: 'Delta',
			format: (v: number) => (v > 0 ? '+' : '') + (v * 100).toFixed(2) + '%',
			cellClass: (v: number) => v > 0 ? 'text-green-600' : 'text-red-600'
		},
		{
			key: 'pnl',
			label: 'P&L',
			format: (v: number) => '$' + v.toLocaleString('en-US', { minimumFractionDigits: 2 }),
			cellClass: (v: number) => v >= 0 ? 'text-green-600' : 'text-red-600'
		},
		{
			key: 'return_pct',
			label: 'Return',
			format: (v: number) => (v > 0 ? '+' : '') + v.toFixed(2) + '%',
			cellClass: (v: number) => v >= 0 ? 'text-green-600' : 'text-red-600'
		}
	];
</script>

<svelte:head>
	<title>Risk - AI Casino</title>
</svelte:head>

<div class="space-y-8">
	<!-- Risk Metrics -->
	<div class="grid grid-cols-1 md:grid-cols-4 gap-6">
		<MetricCard
			title="Sharpe Ratio"
			value={riskReport?.sharpe_ratio != null ? riskReport.sharpe_ratio.toFixed(2) : 'N/A'}
			subtitle="Risk-adjusted return"
			icon="📉"
		/>
		<MetricCard
			title="Volatility"
			value={riskReport?.portfolio_volatility != null ? formatPercent(riskReport.portfolio_volatility) : 'N/A'}
			subtitle="Portfolio volatility"
			icon="📊"
		/>
		<MetricCard
			title="Max Drawdown"
			value={riskReport?.max_drawdown != null ? formatPercent(riskReport.max_drawdown) : 'N/A'}
			subtitle="Peak to trough"
			icon="⚠️"
		/>
		<MetricCard
			title="VaR (95%)"
			value={riskReport?.var_95 != null ? formatPercent(riskReport.var_95) : 'N/A'}
			subtitle="Value at Risk"
			icon="🎲"
		/>
	</div>

	<!-- Risk Status -->
	{#if riskReport}
		<Card title="Risk Status">
			<div class="flex items-center justify-center py-4">
				<RiskStatusBadge status={riskReport.risk_status} size="lg" />
			</div>
		</Card>
	{/if}

	<!-- Risk Gauges -->
	{#if riskReport}
		<div class="grid grid-cols-1 md:grid-cols-3 gap-6">
			<Card title="VaR 95%">
				<GaugeChart
					value={riskReport.var_95 * 100}
					min={0}
					max={10}
					thresholds={{ low: 3, medium: 4.5, high: 10 }}
					title="VaR 95%"
					unit="%"
				/>
			</Card>
			<Card title="CVaR 99%">
				<GaugeChart
					value={riskReport.cvar_99 * 100}
					min={0}
					max={15}
					thresholds={{ low: 5, medium: 7.5, high: 15 }}
					title="CVaR 99%"
					unit="%"
				/>
			</Card>
			<Card title="CDaR 95%">
				<GaugeChart
					value={riskReport.cdar_95 * 100}
					min={0}
					max={25}
					thresholds={{ low: 10, medium: 15, high: 25 }}
					title="CDaR 95%"
					unit="%"
				/>
			</Card>
		</div>
	{/if}

	<!-- Sector Rotation Heatmap -->
	<Card title="Sector Rotation">
		{#if sectorRotationData?.sector_strengths && Object.keys(sectorRotationData.sector_strengths).length > 0}
			<SectorRotationHeatmap
				sectorStrengths={sectorRotationData.sector_strengths}
				sectorMomenta={sectorRotationData.sector_momenta}
				leadingSectors={sectorRotationData.leading_sectors}
				laggingSectors={sectorRotationData.lagging_sectors}
				flaggedPositions={sectorRotationData.flagged_positions}
			/>
		{:else}
			<div class="text-center py-12 text-gray-600">
				Sector rotation not enabled or no data available.
			</div>
		{/if}
	</Card>

	<!-- Sector Contribution & Allocation -->
	<Card title="Sector Contribution & Allocation" class="mt-6">
		{#if sectorAttributionData && sectorAttributionData.contributions.length > 0}
			<!-- Charts Row -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
				<!-- Left: Treemap of current allocation -->
				<div>
					<h3 class="text-sm font-medium mb-2">Current Allocation by Sector</h3>
					<TreemapChart
						data={sectorAllocationData}
						height={350}
					/>
				</div>

				<!-- Right: Bar chart of over/underweight -->
				<div>
					<h3 class="text-sm font-medium mb-2">vs SPY Benchmark</h3>
					<BarChart
						data={sectorWeightDeltaData}
						height={350}
						yAxisLabel="Over/Under Weight (%)"
					/>
				</div>
			</div>

			<!-- Data Table -->
			<DataTable
				columns={sectorContributionColumns}
				data={sectorAttributionData.contributions}
				class="mt-4"
			/>
		{:else}
			<div class="text-center py-12 text-gray-600">
				No positions or sector attribution data available.
			</div>
		{/if}
	</Card>

	<!-- Risk Metrics Table -->
	{#if riskReport}
		<Card title="Detailed Risk Metrics">
			<div class="overflow-x-auto">
				<table class="w-full text-sm">
					<thead>
						<tr class="border-b border-gray-300">
							<th class="text-left py-3 px-4 text-gray-700 font-medium">Metric</th>
							<th class="text-right py-3 px-4 text-gray-700 font-medium">Value</th>
						</tr>
					</thead>
					<tbody class="divide-y divide-gray-200">
						<tr class="hover:bg-gray-50">
							<td class="py-3 px-4 text-gray-600">VaR 95%</td>
							<td class="py-3 px-4 text-right text-gray-800 font-mono">{formatPercent(riskReport.var_95)}</td>
						</tr>
						<tr class="hover:bg-gray-50">
							<td class="py-3 px-4 text-gray-600">VaR 99%</td>
							<td class="py-3 px-4 text-right text-gray-800 font-mono">{formatPercent(riskReport.var_99)}</td>
						</tr>
						<tr class="hover:bg-gray-50">
							<td class="py-3 px-4 text-gray-600">CVaR 95%</td>
							<td class="py-3 px-4 text-right text-gray-800 font-mono">{formatPercent(riskReport.cvar_95)}</td>
						</tr>
						<tr class="hover:bg-gray-50">
							<td class="py-3 px-4 text-gray-600">CVaR 99%</td>
							<td class="py-3 px-4 text-right text-gray-800 font-mono">{formatPercent(riskReport.cvar_99)}</td>
						</tr>
						<tr class="hover:bg-gray-50">
							<td class="py-3 px-4 text-gray-600">CDaR 95%</td>
							<td class="py-3 px-4 text-right text-gray-800 font-mono">{formatPercent(riskReport.cdar_95)}</td>
						</tr>
						<tr class="hover:bg-gray-50">
							<td class="py-3 px-4 text-gray-600">Max Drawdown</td>
							<td class="py-3 px-4 text-right text-gray-800 font-mono">{formatPercent(riskReport.max_drawdown)}</td>
						</tr>
						<tr class="hover:bg-gray-50">
							<td class="py-3 px-4 text-gray-600">Risk Status</td>
							<td class="py-3 px-4 text-right">
								<div class="flex justify-end">
									<RiskStatusBadge status={riskReport.risk_status} size="sm" />
								</div>
							</td>
						</tr>
					</tbody>
				</table>
			</div>
		</Card>
	{/if}

	<!-- Risk Charts -->
	<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
		<Card title="Sharpe Ratio Trend">
			{#if !loading && sharpeData.length > 0}
				<LineChart 
					data={sharpeData} 
					height={300}
					color="#3b82f6"
					yAxisLabel="Sharpe Ratio"
				/>
			{:else if loading}
				<div class="text-center py-12 text-gray-600">Loading...</div>
			{:else}
				<div class="text-center py-12 text-gray-600">No risk history</div>
			{/if}
		</Card>

		<Card title="Volatility Trend">
			{#if !loading && volatilityData.length > 0}
				<LineChart 
					data={volatilityData} 
					height={300}
					color="#ef4444"
					yAxisLabel="Volatility"
				/>
			{:else if loading}
				<div class="text-center py-12 text-gray-600">Loading...</div>
			{:else}
				<div class="text-center py-12 text-gray-600">No volatility data</div>
			{/if}
		</Card>
	</div>

	<!-- Correlation Matrix -->
	<Card title="Correlation Matrix">
		{#if correlationData && correlationData.symbols.length > 0}
			<HeatmapChart 
				symbols={correlationData.symbols}
				matrix={correlationData.matrix}
				height={500}
				title=""
			/>
		{:else}
			<div class="text-center py-12 text-gray-600">
				No correlation data available. Requires multiple positions.
			</div>
		{/if}
	</Card>

	<!-- Risk Explanation -->
	<Card title="Risk Metrics Explained">
		<dl class="space-y-4 text-sm">
			<div>
				<dt class="font-medium text-gray-700">Sharpe Ratio</dt>
				<dd class="mt-1 text-gray-600">
					Measures risk-adjusted returns. Higher is better. &gt;1 is good, &gt;2 is excellent.
				</dd>
			</div>
			<div>
				<dt class="font-medium text-gray-700">Volatility</dt>
				<dd class="mt-1 text-gray-600">
					Standard deviation of returns. Lower means more stable portfolio.
				</dd>
			</div>
			<div>
				<dt class="font-medium text-gray-700">Max Drawdown</dt>
				<dd class="mt-1 text-gray-600">
					Largest peak-to-trough decline. Shows worst-case historical loss.
				</dd>
			</div>
			<div>
				<dt class="font-medium text-gray-700">Value at Risk (VaR 95%)</dt>
				<dd class="mt-1 text-gray-600">
					Maximum expected loss over a time period at 95% confidence level.
				</dd>
			</div>
		</dl>
	</Card>
</div>
