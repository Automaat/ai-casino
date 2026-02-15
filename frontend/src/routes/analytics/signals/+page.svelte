<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import MetricCard from '$lib/components/ui/MetricCard.svelte';
	import Card from '$lib/components/ui/Card.svelte';
	import LineChart from '$lib/components/charts/LineChart.svelte';
	import BarChart from '$lib/components/charts/BarChart.svelte';
	import SankeyChart from '$lib/components/charts/SankeyChart.svelte';
	import { signalAnalytics } from '$lib/stores/signalAnalytics';

	let startDate = $state(getDefaultStartDate());
	let endDate = $state(new Date().toISOString().split('T')[0]);
	let horizon: '1d' | '5d' | '20d' = $state('5d');
	let refreshInterval: number | null = null;
	let ws: WebSocket | null = null;
	let wsConnected = $state(false);

	let analyticsState = $derived($signalAnalytics);
	let summary = $derived(analyticsState.summary);
	let sankeyData = $derived(analyticsState.sankeyData);
	let accuracyByType = $derived(analyticsState.accuracyByType);
	let calibration = $derived(analyticsState.calibration);
	let timing = $derived(analyticsState.timing);
	let executionRate = $derived(analyticsState.executionRate);

	function getDefaultStartDate(): string {
		const date = new Date();
		date.setDate(date.getDate() - 30);
		return date.toISOString().split('T')[0];
	}

	function setDateRange(days: number) {
		const end = new Date();
		const start = new Date();
		start.setDate(start.getDate() - days);
		startDate = start.toISOString().split('T')[0];
		endDate = end.toISOString().split('T')[0];
		loadData();
	}

	function loadData() {
		signalAnalytics.fetchAll(startDate, endDate, horizon);
	}

	function formatPercent(value: number): string {
		return `${(value * 100).toFixed(1)}%`;
	}

	function formatNumber(value: number): string {
		return value.toLocaleString();
	}

	// Accuracy by type chart data
	let accuracyChartData = $derived.by(() => {
		if (!accuracyByType?.data.length) return null;

		return accuracyByType.data.map((d) => ({
			label: d.signal_type,
			value: d.hit_rate,
			color: d.signal_type === 'BUY' ? '#10b981' : '#ef4444'
		}));
	});

	// Execution rate chart data
	let executionRateChartData = $derived.by(() => {
		if (!executionRate?.data.length) return null;

		return executionRate.data.map((d) => ({
			label: d.confidence_bucket,
			value: d.execution_rate,
			color: '#3b82f6'
		}));
	});

	// Calibration curve data
	let calibrationChartData = $derived.by(() => {
		if (!calibration?.buckets.length) return null;

		return calibration.buckets.map((b) => ({
			time: b.confidence_bucket,
			value: b.actual_accuracy
		}));
	});

	// Timing analysis chart data
	let timingChartData = $derived.by(() => {
		if (!timing?.by_confidence_bucket) return null;

		return Object.entries(timing.by_confidence_bucket).map(([bucket, delay]) => ({
			label: bucket,
			value: delay,
			color: '#8b5cf6'
		}));
	});

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
			<h1 class="text-2xl font-bold">Signal Accuracy Tracking</h1>
			<p class="text-gray-600 dark:text-gray-400 text-sm mt-1">
				Monitor signal performance and execution patterns
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

	<!-- Date Range and Horizon Controls -->
	<Card>
		<div class="flex flex-wrap gap-4 items-end">
			<div class="flex-1 min-w-[200px]">
				<label for="start-date" class="block text-sm font-medium mb-1">Start Date</label>
				<input
					id="start-date"
					type="date"
					bind:value={startDate}
					onchange={loadData}
					class="px-3 py-2 border rounded-md w-full dark:bg-gray-800 dark:border-gray-700"
				/>
			</div>
			<div class="flex-1 min-w-[200px]">
				<label for="end-date" class="block text-sm font-medium mb-1">End Date</label>
				<input
					id="end-date"
					type="date"
					bind:value={endDate}
					onchange={loadData}
					class="px-3 py-2 border rounded-md w-full dark:bg-gray-800 dark:border-gray-700"
				/>
			</div>
			<div class="flex-1 min-w-[150px]">
				<label for="horizon" class="block text-sm font-medium mb-1">Horizon</label>
				<select
					id="horizon"
					bind:value={horizon}
					onchange={loadData}
					class="px-3 py-2 border rounded-md w-full dark:bg-gray-800 dark:border-gray-700"
				>
					<option value="1d">1 Day</option>
					<option value="5d">5 Days</option>
					<option value="20d">20 Days</option>
				</select>
			</div>
			<div class="flex gap-2">
				<button
					onclick={() => setDateRange(7)}
					class="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
				>
					Last 7 Days
				</button>
				<button
					onclick={() => setDateRange(30)}
					class="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
				>
					Last 30 Days
				</button>
				<button
					onclick={() => setDateRange(90)}
					class="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
				>
					Last 90 Days
				</button>
			</div>
		</div>
	</Card>

	<!-- Metric Cards -->
	{#if summary}
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
			<MetricCard title="Total Signals" value={formatNumber(summary.total_signals)} subtitle="BUY/SELL only" />
			<MetricCard
				title="Execution Rate"
				value={formatPercent(summary.execution_rate)}
				subtitle={`${formatNumber(summary.executed_count)} / ${formatNumber(summary.total_signals)}`}
			/>
			<MetricCard
				title="Overall Accuracy"
				value={formatPercent(summary.overall_accuracy)}
				subtitle={`${horizon} horizon`}
			/>
			<MetricCard
				title="Avg Confidence"
				value={formatPercent(summary.avg_confidence)}
			/>
		</div>
	{/if}

	<!-- Sankey Diagram -->
	{#if sankeyData && sankeyData.nodes.length > 0}
		<Card>
			<h3 class="text-lg font-semibold mb-4">Signal Flow: Signal Type → Execution → Outcome</h3>
			<SankeyChart data={sankeyData} height={400} />
			<p class="text-sm text-gray-600 dark:text-gray-400 mt-2">
				Note: Outcome based on {horizon} horizon. Signals without outcome data excluded.
			</p>
		</Card>
	{/if}

	<!-- Accuracy and Execution Rate -->
	<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
		{#if accuracyChartData}
			<Card>
				<h3 class="text-lg font-semibold mb-4">Accuracy by Signal Type ({horizon})</h3>
				<BarChart data={accuracyChartData} yAxisLabel="Hit Rate" height={300} />
			</Card>
		{/if}

		{#if executionRateChartData}
			<Card>
				<h3 class="text-lg font-semibold mb-4">Execution Rate by Confidence</h3>
				<BarChart data={executionRateChartData} yAxisLabel="Execution Rate" height={300} />
			</Card>
		{/if}
	</div>

	<!-- Calibration Curve -->
	{#if calibrationChartData}
		<Card>
			<h3 class="text-lg font-semibold mb-4">Calibration Curve: Confidence vs Actual Accuracy ({horizon})</h3>
			<LineChart data={calibrationChartData} yAxisLabel="Accuracy" height={300} color="#10b981" />
			<p class="text-sm text-gray-600 dark:text-gray-400 mt-2">
				Well-calibrated signals should have actual accuracy matching expected confidence.
			</p>
		</Card>
	{/if}

	<!-- Signal Timing Analysis -->
	{#if timingChartData && timing}
		<Card>
			<h3 class="text-lg font-semibold mb-4">Signal Timing Analysis</h3>
			<div class="mb-4">
				<p class="text-sm text-gray-600 dark:text-gray-400">
					Average delay from signal generation to execution:
					<span class="font-semibold text-gray-900 dark:text-gray-100">
						{timing.avg_execution_delay_hours.toFixed(2)} hours
					</span>
				</p>
			</div>
			<BarChart data={timingChartData} yAxisLabel="Delay (hours)" height={300} />
		</Card>
	{/if}

	<!-- Loading State -->
	{#if analyticsState.loading}
		<div class="flex justify-center items-center py-12">
			<div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
		</div>
	{/if}

	<!-- Error State -->
	{#if analyticsState.error}
		<Card>
			<div class="text-red-600 dark:text-red-400">
				<p class="font-semibold">Error</p>
				<p class="text-sm">{analyticsState.error}</p>
			</div>
		</Card>
	{/if}

	<!-- Empty State -->
	{#if !analyticsState.loading && !analyticsState.error && !summary}
		<Card>
			<div class="text-center py-12 text-gray-600 dark:text-gray-400">
				<p>No signal data available for the selected date range</p>
				<p class="text-sm mt-2">Try adjusting the date range or run some analyses</p>
			</div>
		</Card>
	{/if}
</div>
