<script lang="ts">
	import { onMount } from 'svelte';
	import Card from '$lib/components/ui/Card.svelte';
	import MetricCard from '$lib/components/ui/MetricCard.svelte';
	import HeatmapChart from '$lib/components/charts/HeatmapChart.svelte';
	import LineChart from '$lib/components/charts/LineChart.svelte';
	import { risk, correlation } from '$lib/stores/dashboard';
	import { api } from '$lib/api/client';
	import { formatPercent, formatDateShort } from '$lib/utils/format';
	import type { RiskReportResponse } from '$lib/types/api';

	$: riskReport = $risk;
	$: correlationData = $correlation;

	let riskHistory: RiskReportResponse[] = [];
	let loading = true;

	onMount(async () => {
		try {
			await risk.fetch();
			await correlation.fetch();
			const historyData = await api.getRiskHistory(30);
			riskHistory = historyData.history;
		} catch (error) {
			console.error('Failed to load risk data:', error);
		} finally {
			loading = false;
		}
	});

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
				<div class="text-center py-12 text-slate-400">Loading...</div>
			{:else}
				<div class="text-center py-12 text-slate-400">No risk history</div>
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
				<div class="text-center py-12 text-slate-400">Loading...</div>
			{:else}
				<div class="text-center py-12 text-slate-400">No volatility data</div>
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
			<div class="text-center py-12 text-slate-400">
				No correlation data available. Requires multiple positions.
			</div>
		{/if}
	</Card>

	<!-- Risk Explanation -->
	<Card title="Risk Metrics Explained">
		<dl class="space-y-4 text-sm">
			<div>
				<dt class="font-medium text-slate-300">Sharpe Ratio</dt>
				<dd class="mt-1 text-slate-400">
					Measures risk-adjusted returns. Higher is better. &gt;1 is good, &gt;2 is excellent.
				</dd>
			</div>
			<div>
				<dt class="font-medium text-slate-300">Volatility</dt>
				<dd class="mt-1 text-slate-400">
					Standard deviation of returns. Lower means more stable portfolio.
				</dd>
			</div>
			<div>
				<dt class="font-medium text-slate-300">Max Drawdown</dt>
				<dd class="mt-1 text-slate-400">
					Largest peak-to-trough decline. Shows worst-case historical loss.
				</dd>
			</div>
			<div>
				<dt class="font-medium text-slate-300">Value at Risk (VaR 95%)</dt>
				<dd class="mt-1 text-slate-400">
					Maximum expected loss over a time period at 95% confidence level.
				</dd>
			</div>
		</dl>
	</Card>
</div>
