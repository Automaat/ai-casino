<script lang="ts">
	import MetricCard from '$lib/components/ui/MetricCard.svelte';
	import Card from '$lib/components/ui/Card.svelte';
	import DataTable from '$lib/components/ui/DataTable.svelte';
	import LineChart from '$lib/components/charts/LineChart.svelte';
	import ServiceHealthBadge from '$lib/components/ui/ServiceHealthBadge.svelte';
	import GamePlanCard from '$lib/components/ui/GamePlanCard.svelte';
	import WatchlistBreakdown from '$lib/components/ui/WatchlistBreakdown.svelte';
	import DegradationAlert from '$lib/components/ui/DegradationAlert.svelte';
	import {
		stateSummary,
		serviceHealth,
		gamePlan,
		watchlist,
		degradation
	} from '$lib/stores/dashboard';
	import { formatPercent, formatDateShort } from '$lib/utils/format';
	import type { AnalysisRecordResponse } from '$lib/types/api';

	$: summary = $stateSummary;
	$: services = $serviceHealth;
	$: plan = $gamePlan;
	$: wlist = $watchlist;
	$: deg = $degradation;
	$: recentAnalyses = summary?.recent_analyses || [];

	// Chart data: confidence over time
	$: confidenceData = recentAnalyses
		.slice()
		.reverse()
		.map(a => ({
			time: formatDateShort(a.timestamp),
			value: a.confidence
		}));

	const analysisColumns = [
		{
			key: 'symbol' as keyof AnalysisRecordResponse,
			label: 'Symbol',
			class: 'font-medium'
		},
		{
			key: 'signal' as keyof AnalysisRecordResponse,
			label: 'Signal',
			format: (value: string) => value
		},
		{
			key: 'confidence' as keyof AnalysisRecordResponse,
			label: 'Confidence',
			format: (value: number) => formatPercent(value)
		},
		{
			key: 'timestamp' as keyof AnalysisRecordResponse,
			label: 'Time',
			format: (value: string) => new Date(value).toLocaleTimeString()
		}
	];
</script>

<svelte:head>
	<title>Overview - AI Casino</title>
</svelte:head>

<div class="space-y-8">
	<!-- Service Health Badges -->
	{#if services && services.service_checks.length > 0}
		<div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
			{#each services.service_checks as check}
				<ServiceHealthBadge {check} />
			{/each}
		</div>
	{/if}

	<!-- Game Plan + Watchlist Row -->
	<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
		<GamePlanCard {plan} />
		<WatchlistBreakdown watchlist={wlist} />
	</div>

	<!-- Degradation Alert (conditional) -->
	<DegradationAlert degradation={deg} />

	<!-- Metrics Grid -->
	<div class="grid grid-cols-1 md:grid-cols-4 gap-6">
		<MetricCard
			title="Total Analyses"
			value={summary?.total_analyses ?? 0}
			icon="📊"
		/>
		<MetricCard
			title="Active Positions"
			value={summary?.positions_count ?? 0}
			icon="💼"
		/>
		<MetricCard
			title="Win Rate"
			value={summary ? formatPercent(summary.win_rate) : '0%'}
			icon="🎯"
		/>
		<MetricCard
			title="Degradation"
			value={summary?.degradation_tier ?? 'UNKNOWN'}
			icon="⚠️"
		/>
	</div>

	<!-- Confidence Trend Chart -->
	<Card title="Confidence Trend">
		{#if confidenceData.length > 0}
			<LineChart
				data={confidenceData}
				height={300}
				color="#3b82f6"
				yAxisLabel="Confidence"
			/>
		{:else}
			<div class="text-center py-12 text-slate-400">
				No analysis data available
			</div>
		{/if}
	</Card>

	<!-- Recent Analyses Table -->
	<Card title="Recent Analyses">
		{#if recentAnalyses.length > 0}
			<DataTable data={recentAnalyses.slice(0, 10)} columns={analysisColumns} />
		{:else}
			<div class="text-center py-12 text-slate-400">
				No analyses yet. Daemon may be stopped or no trading cycles completed.
			</div>
		{/if}
	</Card>

	<!-- System Status -->
	{#if summary}
		<Card title="System Status">
			<dl class="grid grid-cols-2 gap-4 text-sm">
				<div>
					<dt class="text-slate-400">Total Trades</dt>
					<dd class="mt-1 text-lg font-semibold text-slate-100">{summary.total_trades}</dd>
				</div>
				<div>
					<dt class="text-slate-400">Errors</dt>
					<dd class="mt-1 text-lg font-semibold text-slate-100">{summary.error_count}</dd>
				</div>
			</dl>
		</Card>
	{/if}
</div>
