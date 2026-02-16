<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import MetricCard from '$lib/components/ui/MetricCard.svelte';
	import Card from '$lib/components/ui/Card.svelte';
	import BarChart from '$lib/components/charts/BarChart.svelte';
	import PieChart from '$lib/components/charts/PieChart.svelte';
	import DataTable from '$lib/components/ui/DataTable.svelte';
	import { workerPerformance } from '$lib/stores/workerPerformance';

	let activePeriod: '24h' | '7d' | '30d' = $state('24h');
	let refreshInterval: number | null = null;
	let ws: WebSocket | null = null;
	let wsConnected = $state(false);
	let reconnectTimeout: number | null = null;
	let reconnectAttempts = 0;
	const MAX_RECONNECT_ATTEMPTS = 5;
	const RECONNECT_DELAY = 3000;

	let perfState = $derived($workerPerformance);
	let workers24h = $derived(perfState.workers_24h);
	let workers7d = $derived(perfState.workers_7d);
	let workers30d = $derived(perfState.workers_30d);
	let errors24h = $derived(perfState.errors_24h);
	let errors7d = $derived(perfState.errors_7d);
	let errors30d = $derived(perfState.errors_30d);
	let validation = $derived(perfState.validation);

	// Active period data
	let workersCurrent = $derived(
		activePeriod === '24h' ? workers24h : activePeriod === '7d' ? workers7d : workers30d
	);
	let errorsCurrent = $derived(
		activePeriod === '24h' ? errors24h : activePeriod === '7d' ? errors7d : errors30d
	);

	// Worker filtering
	let enabledWorkers: Set<string> = $state(new Set());

	// Initialize enabled workers when data loads
	$effect(() => {
		if (workersCurrent && enabledWorkers.size === 0) {
			enabledWorkers = new Set(Object.keys(workersCurrent.worker_stats));
		}
	});

	let filteredWorkers = $derived.by(() => {
		if (!workersCurrent) return [];
		return Object.keys(workersCurrent.worker_stats).filter((w) => enabledWorkers.has(w));
	});

	// Metric card values
	let totalExecutions = $derived.by(() => {
		if (!workersCurrent) return 0;
		return filteredWorkers.reduce(
			(sum, w) => sum + workersCurrent.worker_stats[w].total_executions,
			0
		);
	});

	let overallSuccessRate = $derived.by(() => {
		if (!workersCurrent || totalExecutions === 0) return 0;
		const totalSuccess = filteredWorkers.reduce(
			(sum, w) => sum + workersCurrent.worker_stats[w].successful_executions,
			0
		);
		return (totalSuccess / totalExecutions) * 100;
	});

	let avgExecutionTime = $derived.by(() => {
		if (!workersCurrent || totalExecutions === 0) return 0;
		const totalDuration = filteredWorkers.reduce(
			(sum, w) =>
				sum +
				workersCurrent.worker_stats[w].avg_duration_ms *
					workersCurrent.worker_stats[w].total_executions,
			0
		);
		return totalDuration / totalExecutions;
	});

	let totalErrors = $derived.by(() => {
		if (!errorsCurrent) return 0;
		return filteredWorkers.reduce((sum, w) => sum + (errorsCurrent.error_counts[w] || 0), 0);
	});

	// Success rate comparison (24h, 7d, 30d for each worker)
	let successRateData24h = $derived.by(() => {
		if (!workers24h) return null;
		return filteredWorkers.map((w) => ({
			label: w,
			value: workers24h.worker_stats[w]?.success_rate || 0,
			color: getSuccessRateColor(workers24h.worker_stats[w]?.success_rate || 0)
		}));
	});

	let successRateData7d = $derived.by(() => {
		if (!workers7d) return null;
		return filteredWorkers.map((w) => ({
			label: w,
			value: workers7d.worker_stats[w]?.success_rate || 0,
			color: getSuccessRateColor(workers7d.worker_stats[w]?.success_rate || 0)
		}));
	});

	let successRateData30d = $derived.by(() => {
		if (!workers30d) return null;
		return filteredWorkers.map((w) => ({
			label: w,
			value: workers30d.worker_stats[w]?.success_rate || 0,
			color: getSuccessRateColor(workers30d.worker_stats[w]?.success_rate || 0)
		}));
	});

	// Execution time (sorted descending)
	let executionTimeData = $derived.by(() => {
		if (!workersCurrent) return null;
		return filteredWorkers
			.map((w) => ({
				label: w,
				value: workersCurrent.worker_stats[w].avg_duration_ms,
				color: workersCurrent.worker_stats[w].avg_duration_ms > 60000 ? '#ef4444' : '#10b981'
			}))
			.sort((a, b) => b.value - a.value);
	});

	// Timeout workers (>60s threshold)
	const TIMEOUT_THRESHOLD_MS = 60000;
	let timeoutWorkers = $derived.by(() => {
		if (!workersCurrent) return [];
		return filteredWorkers
			.filter((w) => workersCurrent.worker_stats[w].avg_duration_ms > TIMEOUT_THRESHOLD_MS)
			.map((w) => ({
				name: w,
				avg_duration: (workersCurrent.worker_stats[w].avg_duration_ms / 1000).toFixed(1) + 's',
				executions: workersCurrent.worker_stats[w].total_executions
			}));
	});

	// Error distribution (PieChart)
	let errorDistribution = $derived.by(() => {
		if (!errorsCurrent) return null;
		const data = filteredWorkers
			.filter((w) => (errorsCurrent.error_counts[w] || 0) > 0)
			.map((w) => ({
				label: w,
				value: errorsCurrent.error_counts[w]
			}));
		return data.length > 0 ? data : null;
	});

	// Execution volume comparison
	let executionVolumeData24h = $derived.by(() => {
		if (!workers24h) return null;
		return filteredWorkers.map((w) => ({
			label: w,
			value: workers24h.worker_stats[w]?.total_executions || 0,
			color: '#3b82f6'
		}));
	});

	let executionVolumeData7d = $derived.by(() => {
		if (!workers7d) return null;
		return filteredWorkers.map((w) => ({
			label: w,
			value: workers7d.worker_stats[w]?.total_executions || 0,
			color: '#8b5cf6'
		}));
	});

	let executionVolumeData30d = $derived.by(() => {
		if (!workers30d) return null;
		return filteredWorkers.map((w) => ({
			label: w,
			value: workers30d.worker_stats[w]?.total_executions || 0,
			color: '#ec4899'
		}));
	});

	// Worker performance table
	let workerTable = $derived.by(() => {
		if (!workersCurrent || !errorsCurrent) return [];
		return filteredWorkers.map((w) => {
			const stats = workersCurrent.worker_stats[w];
			return {
				name: w,
				executions: stats.total_executions,
				success_rate: stats.success_rate.toFixed(1),
				failed: stats.failed_executions,
				avg_duration: (stats.avg_duration_ms / 1000).toFixed(1) + 's',
				errors: errorsCurrent.error_counts[w] || 0,
				status: stats.avg_duration_ms > TIMEOUT_THRESHOLD_MS ? 'Slow' : 'Fast'
			};
		});
	});

	function getSuccessRateColor(rate: number): string {
		if (rate >= 90) return '#10b981'; // green
		if (rate >= 70) return '#eab308'; // yellow
		return '#ef4444'; // red
	}

	function getSuccessRateCellClass(rateStr: string): string {
		const rate = parseFloat(rateStr);
		if (rate >= 90) return 'bg-green-100 text-green-700 font-semibold px-2 py-1 rounded';
		if (rate >= 70) return 'bg-yellow-100 text-yellow-700 font-semibold px-2 py-1 rounded';
		return 'bg-red-100 text-red-700 font-semibold px-2 py-1 rounded';
	}

	function getStatusBadgeClass(status: string): string {
		return status === 'Fast'
			? 'bg-green-100 text-green-700 px-2.5 py-0.5 rounded-full text-xs font-semibold'
			: 'bg-red-100 text-red-700 px-2.5 py-0.5 rounded-full text-xs font-semibold';
	}

	function toggleWorker(worker: string) {
		if (enabledWorkers.has(worker)) {
			enabledWorkers.delete(worker);
		} else {
			enabledWorkers.add(worker);
		}
		enabledWorkers = new Set(enabledWorkers);
	}

	function loadData() {
		workerPerformance.fetchAll();
	}

	const WS_URL =
		import.meta.env.VITE_WS_URL ||
		(typeof window !== 'undefined'
			? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`
			: 'ws://localhost:8484');

	function connectWebSocket() {
		if (typeof window === 'undefined') return;

		const wsUrl = `${WS_URL}/ws/events`;
		ws = new WebSocket(wsUrl);

		ws.onopen = () => {
			wsConnected = true;
			reconnectAttempts = 0;
		};

		ws.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				if (data.event_type === 'CYCLE_COMPLETE') {
					loadData();
				}
			} catch (error) {
				console.error('WebSocket message parse error:', error);
			}
		};

		ws.onclose = () => {
			wsConnected = false;

			if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
				reconnectAttempts++;
				console.log(`Reconnecting (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...`);
				reconnectTimeout = setTimeout(connectWebSocket, RECONNECT_DELAY);
			} else {
				console.error('Failed to connect to WebSocket after multiple attempts');
			}
		};

		ws.onerror = (error) => {
			console.error('WebSocket error:', error);
			wsConnected = false;
		};
	}

	onMount(() => {
		loadData();
		refreshInterval = window.setInterval(loadData, 30000);
		connectWebSocket();
	});

	onDestroy(() => {
		if (reconnectTimeout) clearTimeout(reconnectTimeout);
		if (refreshInterval) clearInterval(refreshInterval);
		if (ws) {
			ws.close();
			ws = null;
		}
	});
</script>

<div class="p-6 space-y-6">
	<!-- Header -->
	<div class="flex justify-between items-center">
		<div>
			<h1 class="text-2xl font-bold">Worker Performance Analytics</h1>
			<p class="text-gray-600 dark:text-gray-400 text-sm mt-1">
				Monitor worker execution metrics and identify bottlenecks
			</p>
		</div>
		<div class="flex items-center gap-2">
			<div
				class="w-2 h-2 rounded-full"
				class:bg-green-500={wsConnected}
				class:bg-red-500={!wsConnected}
			></div>
			<span class="text-sm text-gray-600 dark:text-gray-400">
				{wsConnected ? 'Live' : 'Disconnected'}
			</span>
		</div>
	</div>

	<!-- Period Selector -->
	<Card>
		<div class="flex gap-2">
			<button
				onclick={() => (activePeriod = '24h')}
				class="px-4 py-2 rounded-md text-sm font-medium transition-colors"
				class:bg-blue-500={activePeriod === '24h'}
				class:text-white={activePeriod === '24h'}
				class:bg-gray-200={activePeriod !== '24h'}
				class:dark:bg-gray-700={activePeriod !== '24h'}
			>
				24 Hours
			</button>
			<button
				onclick={() => (activePeriod = '7d')}
				class="px-4 py-2 rounded-md text-sm font-medium transition-colors"
				class:bg-blue-500={activePeriod === '7d'}
				class:text-white={activePeriod === '7d'}
				class:bg-gray-200={activePeriod !== '7d'}
				class:dark:bg-gray-700={activePeriod !== '7d'}
			>
				7 Days
			</button>
			<button
				onclick={() => (activePeriod = '30d')}
				class="px-4 py-2 rounded-md text-sm font-medium transition-colors"
				class:bg-blue-500={activePeriod === '30d'}
				class:text-white={activePeriod === '30d'}
				class:bg-gray-200={activePeriod !== '30d'}
				class:dark:bg-gray-700={activePeriod !== '30d'}
			>
				30 Days
			</button>
		</div>
	</Card>

	<!-- Worker Filter -->
	{#if workersCurrent}
		<Card>
			<h3 class="text-sm font-medium mb-2">Filter Workers</h3>
			<div class="flex flex-wrap gap-2">
				{#each Object.keys(workersCurrent.worker_stats) as worker}
					<button
						onclick={() => toggleWorker(worker)}
						class="px-3 py-1 rounded-full text-sm font-medium transition-colors"
						class:bg-blue-500={enabledWorkers.has(worker)}
						class:text-white={enabledWorkers.has(worker)}
						class:bg-gray-200={!enabledWorkers.has(worker)}
						class:text-gray-600={!enabledWorkers.has(worker)}
						class:dark:bg-gray-700={!enabledWorkers.has(worker)}
						class:dark:text-gray-300={!enabledWorkers.has(worker)}
					>
						{worker}
					</button>
				{/each}
			</div>
		</Card>
	{/if}

	<!-- Metric Cards -->
	{#if workersCurrent && errorsCurrent}
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
			<MetricCard title="Total Executions" value={totalExecutions.toString()} />
			<MetricCard title="Overall Success Rate" value={overallSuccessRate.toFixed(1) + '%'} />
			<MetricCard title="Avg Execution Time" value={(avgExecutionTime / 1000).toFixed(1) + 's'} />
			<MetricCard title="Total Errors" value={totalErrors.toString()} />
		</div>
	{/if}

	<!-- Success Rate Comparison -->
	{#if successRateData24h && successRateData7d && successRateData30d}
		<Card>
			<h3 class="text-lg font-semibold mb-4">Success Rate Evolution</h3>
			<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
				<div>
					<h4 class="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">24 Hours</h4>
					<BarChart data={successRateData24h} yAxisLabel="Success Rate (%)" height={250} />
				</div>
				<div>
					<h4 class="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">7 Days</h4>
					<BarChart data={successRateData7d} yAxisLabel="Success Rate (%)" height={250} />
				</div>
				<div>
					<h4 class="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">30 Days</h4>
					<BarChart data={successRateData30d} yAxisLabel="Success Rate (%)" height={250} />
				</div>
			</div>
		</Card>
	{/if}

	<!-- Execution Time Analysis -->
	{#if executionTimeData}
		<Card>
			<h3 class="text-lg font-semibold mb-4">
				Execution Time Analysis ({activePeriod})
			</h3>
			<p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
				Red bars indicate workers exceeding 60s threshold
			</p>
			<BarChart data={executionTimeData} yAxisLabel="Avg Duration (ms)" height={300} />
		</Card>
	{/if}

	<!-- Timeout Analysis -->
	{#if timeoutWorkers.length > 0}
		<Card>
			<h3 class="text-lg font-semibold mb-4">
				Slow Workers (&gt;60s) - {timeoutWorkers.length} detected
			</h3>
			<DataTable
				columns={[
					{ key: 'name', label: 'Worker' },
					{ key: 'avg_duration', label: 'Avg Duration' },
					{ key: 'executions', label: 'Total Executions' }
				]}
				data={timeoutWorkers}
			/>
		</Card>
	{/if}

	<!-- Error Distribution and Trends -->
	<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
		{#if errorDistribution}
			<Card>
				<h3 class="text-lg font-semibold mb-4">Error Distribution ({activePeriod})</h3>
				<PieChart data={errorDistribution} height={300} />
			</Card>
		{:else}
			<Card>
				<h3 class="text-lg font-semibold mb-4">Error Distribution ({activePeriod})</h3>
				<div class="text-center py-12 text-gray-600 dark:text-gray-400">
					<p>No errors in selected period</p>
				</div>
			</Card>
		{/if}

		{#if errors24h && errors7d && errors30d}
			<Card>
				<h3 class="text-lg font-semibold mb-4">Error Trends</h3>
				<div class="space-y-4">
					<div>
						<div class="flex justify-between text-sm mb-1">
							<span>24 Hours</span>
							<span class="font-medium">{errors24h.total_errors}</span>
						</div>
						<div class="bg-gray-200 dark:bg-gray-700 rounded-full h-2">
							<div
								class="bg-red-500 h-2 rounded-full"
								style="width: {Math.min(100, (errors24h.total_errors / Math.max(errors24h.total_errors, errors7d.total_errors, errors30d.total_errors, 1)) * 100)}%"
							></div>
						</div>
					</div>
					<div>
						<div class="flex justify-between text-sm mb-1">
							<span>7 Days</span>
							<span class="font-medium">{errors7d.total_errors}</span>
						</div>
						<div class="bg-gray-200 dark:bg-gray-700 rounded-full h-2">
							<div
								class="bg-orange-500 h-2 rounded-full"
								style="width: {Math.min(100, (errors7d.total_errors / Math.max(errors24h.total_errors, errors7d.total_errors, errors30d.total_errors, 1)) * 100)}%"
							></div>
						</div>
					</div>
					<div>
						<div class="flex justify-between text-sm mb-1">
							<span>30 Days</span>
							<span class="font-medium">{errors30d.total_errors}</span>
						</div>
						<div class="bg-gray-200 dark:bg-gray-700 rounded-full h-2">
							<div
								class="bg-yellow-500 h-2 rounded-full"
								style="width: {Math.min(100, (errors30d.total_errors / Math.max(errors24h.total_errors, errors7d.total_errors, errors30d.total_errors, 1)) * 100)}%"
							></div>
						</div>
					</div>
				</div>
			</Card>
		{/if}
	</div>

	<!-- Execution Volume Comparison -->
	{#if executionVolumeData24h && executionVolumeData7d && executionVolumeData30d}
		<Card>
			<h3 class="text-lg font-semibold mb-4">Execution Volume Trends</h3>
			<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
				<div>
					<h4 class="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">24 Hours</h4>
					<BarChart data={executionVolumeData24h} yAxisLabel="Executions" height={250} />
				</div>
				<div>
					<h4 class="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">7 Days</h4>
					<BarChart data={executionVolumeData7d} yAxisLabel="Executions" height={250} />
				</div>
				<div>
					<h4 class="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">30 Days</h4>
					<BarChart data={executionVolumeData30d} yAxisLabel="Executions" height={250} />
				</div>
			</div>
		</Card>
	{/if}

	<!-- Validation Status -->
	{#if validation}
		<Card>
			<div class="flex justify-between items-center mb-4">
				<h3 class="text-lg font-semibold">Paper Trading Validation</h3>
				<span
					class="px-3 py-1 rounded-full text-sm font-medium"
					class:bg-green-100={validation.ready_for_live}
					class:text-green-800={validation.ready_for_live}
					class:bg-yellow-100={!validation.ready_for_live}
					class:text-yellow-800={!validation.ready_for_live}
				>
					{validation.ready_for_live ? 'Ready for Live' : 'Not Ready'}
				</span>
			</div>

			<div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
				<div class="text-sm">
					<span class="text-gray-600 dark:text-gray-400">Assessment Date:</span>
					<span class="ml-2 font-medium"
						>{new Date(validation.assessment_date).toLocaleDateString()}</span
					>
				</div>
				<div class="text-sm">
					<span class="text-gray-600 dark:text-gray-400">Paper Trading Duration:</span>
					<span class="ml-2 font-medium">{validation.paper_trading_duration_days} days</span>
				</div>
				<div class="text-sm">
					<span class="text-gray-600 dark:text-gray-400">Total Paper Trades:</span>
					<span class="ml-2 font-medium">{validation.total_paper_trades}</span>
				</div>
			</div>

			<div class="space-y-2">
				<h4 class="font-medium text-sm">Validation Criteria:</h4>
				{#each validation.criteria as criterion}
					<div class="flex items-center justify-between py-2 px-3 bg-gray-50 dark:bg-gray-800 rounded">
						<div class="flex items-center gap-2">
							<span class="text-xl">
								{#if criterion.passed}
									✅
								{:else}
									❌
								{/if}
							</span>
							<span class="text-sm font-medium">{criterion.name}</span>
						</div>
						<div class="text-sm text-gray-600 dark:text-gray-400">
							{criterion.current_value.toFixed(2)} / {criterion.threshold.toFixed(2)}
						</div>
					</div>
					{#if !criterion.passed}
						<p class="text-xs text-gray-600 dark:text-gray-400 ml-9">{criterion.message}</p>
					{/if}
				{/each}
			</div>

			{#if validation.recommendations.length > 0}
				<div class="mt-4">
					<h4 class="font-medium text-sm mb-2">Recommendations:</h4>
					<ul class="list-disc list-inside space-y-1 text-sm text-gray-600 dark:text-gray-400">
						{#each validation.recommendations as recommendation}
							<li>{recommendation}</li>
						{/each}
					</ul>
				</div>
			{/if}
		</Card>
	{/if}

	<!-- Worker Performance Table -->
	{#if workerTable.length > 0}
		<Card>
			<h3 class="text-lg font-semibold mb-4">Worker Performance Details ({activePeriod})</h3>
			<div class="overflow-x-auto">
				<table class="min-w-full divide-y divide-gray-200">
					<thead class="bg-gray-100">
						<tr>
							<th class="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider"
								>Worker Type</th
							>
							<th class="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider"
								>Executions</th
							>
							<th class="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider"
								>Success Rate</th
							>
							<th class="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider"
								>Failed</th
							>
							<th class="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider"
								>Avg Duration</th
							>
							<th class="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider"
								>Errors</th
							>
							<th class="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider"
								>Status</th
							>
						</tr>
					</thead>
					<tbody class="bg-white divide-y divide-gray-200">
						{#each workerTable as row}
							<tr class="hover:bg-gray-50 transition-colors">
								<td class="px-4 py-3 text-sm font-medium text-gray-900">{row.name}</td>
								<td class="px-4 py-3 text-sm text-gray-700">{row.executions}</td>
								<td class="px-4 py-3 text-sm">
									<span class={getSuccessRateCellClass(row.success_rate)}
										>{row.success_rate}%</span
									>
								</td>
								<td class="px-4 py-3 text-sm text-gray-700">{row.failed}</td>
								<td class="px-4 py-3 text-sm text-gray-700">{row.avg_duration}</td>
								<td class="px-4 py-3 text-sm text-gray-700">{row.errors}</td>
								<td class="px-4 py-3 text-sm">
									<span class={getStatusBadgeClass(row.status)}>{row.status}</span>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</Card>
	{/if}

	<!-- Loading State -->
	{#if perfState.loading}
		<div class="flex justify-center items-center py-12">
			<div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
		</div>
	{/if}

	<!-- Error State -->
	{#if perfState.error}
		<Card>
			<div class="text-red-600 dark:text-red-400">
				<p class="font-semibold">Error loading worker performance</p>
				<p class="text-sm">{perfState.error}</p>
				<button
					onclick={() => workerPerformance.fetchAll()}
					class="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
				>
					Retry
				</button>
			</div>
		</Card>
	{/if}

	<!-- Empty State -->
	{#if !perfState.loading && !perfState.error && !workersCurrent}
		<Card>
			<div class="text-center py-12 text-gray-600 dark:text-gray-400">
				<p>No worker performance data available</p>
				<p class="text-sm mt-2">System may not have run any cycles yet</p>
			</div>
		</Card>
	{/if}
</div>
