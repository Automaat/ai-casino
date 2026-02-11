<script lang="ts">
	import MetricCard from '$lib/components/ui/MetricCard.svelte';
	import Card from '$lib/components/ui/Card.svelte';
	import Badge from '$lib/components/ui/Badge.svelte';
	import DataTable from '$lib/components/ui/DataTable.svelte';
	import LineChart from '$lib/components/charts/LineChart.svelte';
	import { stateSummary } from '$lib/stores/dashboard';
	import { formatPercent, formatDateShort } from '$lib/utils/format';
	import type { AnalysisRecordResponse } from '$lib/types/api';

	$: summary = $stateSummary;
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

	<!-- Charts Row -->
	<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
		<!-- Confidence Chart -->
		<Card title="Confidence Trend" class="lg:col-span-2">
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
	</div>

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

	<!-- System Info -->
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
