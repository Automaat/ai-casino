<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import Card from '$lib/components/ui/Card.svelte';
	import MetricCard from '$lib/components/ui/MetricCard.svelte';
	import HeatmapChart from '$lib/components/charts/HeatmapChart.svelte';
	import LineChart from '$lib/components/charts/LineChart.svelte';
	import GaugeChart from '$lib/components/charts/GaugeChart.svelte';
	import SectorRotationHeatmap from '$lib/components/charts/SectorRotationHeatmap.svelte';
	import RiskStatusBadge from '$lib/components/ui/RiskStatusBadge.svelte';
	import { risk, correlation, sectorRotation } from '$lib/stores/dashboard';
	import { api } from '$lib/api/client';
	import { formatPercent, formatDateShort } from '$lib/utils/format';
	import type { RiskReportResponse } from '$lib/types/api';

	$: riskReport = $risk;
	$: correlationData = $correlation;
	$: sectorRotationData = $sectorRotation;

	let riskHistory: RiskReportResponse[] = [];
	let loading = true;

	async function loadData() {
		loading = true;
		try {
			await risk.fetch();
			await correlation.fetch();
			await sectorRotation.fetch();
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
		.map(r => ({
			time: formatDateShort(r.timestamp),
			value: r.sharpe_ratio
		}));

	// Volatility history
	$: volatilityData = riskHistory
		.slice()
		.reverse()
		.map(r => ({
			time: formatDateShort(r.timestamp),
			value: r.portfolio_volatility
		}));
</script>

<svelte:head>
	<title>Risk - AI Casino</title>
</svelte:head>

<div class="space-y-8">
	<!-- Risk Metrics -->
	<div class="grid grid-cols-1 md:grid-cols-4 gap-6">
		<MetricCard
			title="Sharpe Ratio"
			value={riskReport ? riskReport.sharpe_ratio.toFixed(2) : 'N/A'}
			subtitle="Risk-adjusted return"
			icon="📉"
		/>
		<MetricCard
			title="Volatility"
			value={riskReport ? formatPercent(riskReport.portfolio_volatility) : 'N/A'}
			subtitle="Portfolio volatility"
			icon="📊"
		/>
		<MetricCard
			title="Max Drawdown"
			value={riskReport ? formatPercent(riskReport.max_drawdown) : 'N/A'}
			subtitle="Peak to trough"
			icon="⚠️"
		/>
		<MetricCard
			title="VaR (95%)"
			value={riskReport ? formatPercent(riskReport.var_95) : 'N/A'}
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
