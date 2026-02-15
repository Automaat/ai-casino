<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import MetricCard from '$lib/components/ui/MetricCard.svelte';
	import Card from '$lib/components/ui/Card.svelte';
	import LineChart from '$lib/components/charts/LineChart.svelte';
	import BarChart from '$lib/components/charts/BarChart.svelte';
	import PieChart from '$lib/components/charts/PieChart.svelte';
	import { costAnalytics } from '$lib/stores/costAnalytics';

	let startDate = $state(getDefaultStartDate());
	let endDate = $state(new Date().toISOString().split('T')[0]);
	let period: 'daily' | 'weekly' = $state('daily');
	let refreshInterval: number | null = null;
	let ws: WebSocket | null = null;
	let wsConnected = $state(false);

	const state = $derived($costAnalytics);
	const summary = $derived(state.summary);
	const trends = $derived(state.trends);
	const bySymbol = $derived(state.bySymbol);
	const byAgent = $derived(state.byAgent);
	const byModel = $derived(state.byModel);

	function getDefaultStartDate(): string {
		const date = new Date();
		date.setDate(date.getDate() - 7);
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
		costAnalytics.fetchAll(startDate, endDate, period);
	}

	function formatCurrency(value: number): string {
		if (value < 0.01) {
			return `$${value.toFixed(4)}`;
		}
		return `$${value.toFixed(2)}`;
	}

	function formatNumber(value: number): string {
		if (value >= 1000000) {
			return `${(value / 1000000).toFixed(1)}M`;
		}
		if (value >= 1000) {
			return `${(value / 1000).toFixed(1)}K`;
		}
		return value.toString();
	}

	// Line chart data for cost trends
	const trendChartData = $derived.by(() => {
		if (!trends?.trends.length) return null;

		const dates = trends.trends.map(t => new Date(t.timestamp).toLocaleDateString());
		const costs = trends.trends.map(t => t.cost_usd);

		return {
			xAxis: { type: 'category' as const, data: dates },
			yAxis: { type: 'value' as const, name: 'Cost (USD)' },
			series: [
				{
					name: 'Cost',
					type: 'line' as const,
					data: costs,
					smooth: true,
					itemStyle: { color: '#3b82f6' }
				}
			]
		};
	});

	// Bar chart data for cost by symbol
	const symbolChartData = $derived.by(() => {
		if (!bySymbol?.data.length) return null;

		const top10 = bySymbol.data.slice(0, 10);
		const symbols = top10.map(d => d.dimension_value);
		const costs = top10.map(d => d.cost_usd);

		return {
			xAxis: { type: 'category' as const, data: symbols },
			yAxis: { type: 'value' as const, name: 'Cost (USD)' },
			series: [
				{
					name: 'Cost',
					type: 'bar' as const,
					data: costs,
					itemStyle: { color: '#10b981' }
				}
			]
		};
	});

	// Pie chart data for token distribution by agent
	const agentPieData = $derived.by(() => {
		if (!byAgent?.data.length) return null;

		return byAgent.data.map(d => ({
			name: d.dimension_value,
			value: d.tokens
		}));
	});

	// Bar chart data for model cost comparison
	const modelChartData = $derived.by(() => {
		if (!byModel?.data.length) return null;

		const models = byModel.data.map(d => d.dimension_value);
		const costs = byModel.data.map(d => d.cost_usd);

		return {
			xAxis: { type: 'category' as const, data: models },
			yAxis: { type: 'value' as const, name: 'Cost (USD)' },
			series: [
				{
					name: 'Cost',
					type: 'bar' as const,
					data: costs,
					itemStyle: { color: '#8b5cf6' }
				}
			]
		};
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
			<h1 class="text-2xl font-bold">LLM Cost Analytics</h1>
			<p class="text-gray-600 dark:text-gray-400 text-sm mt-1">
				Monitor spending and optimize LLM usage
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

	<!-- Date Range Controls -->
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
			</div>
		</div>
	</Card>

	<!-- Metric Cards -->
	{#if summary}
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
			<MetricCard
				title="Total Cost"
				value={formatCurrency(summary.total_cost_usd)}
				trend={null}
			/>
			<MetricCard
				title="Avg Cost / Signal"
				value={formatCurrency(summary.avg_cost_per_signal)}
				subtitle="BUY/SELL only"
				trend={null}
			/>
			<MetricCard
				title="Total Tokens"
				value={formatNumber(summary.total_tokens)}
				trend={null}
			/>
			<MetricCard
				title="Executions"
				value={summary.total_executions.toString()}
				subtitle={`Avg ${formatCurrency(summary.avg_cost_per_execution)}/exec`}
				trend={null}
			/>
		</div>

		<!-- Forecast Card -->
		<Card>
			<div class="flex items-center justify-between">
				<div>
					<h3 class="text-lg font-semibold">30-Day Forecast</h3>
					<p class="text-2xl font-bold text-blue-600 dark:text-blue-400 mt-2">
						{formatCurrency(summary.forecast_30d_usd)}
					</p>
				</div>
				<div class="text-gray-600 dark:text-gray-400">
					<svg
						class="w-12 h-12"
						fill="none"
						stroke="currentColor"
						viewBox="0 0 24 24"
					>
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							stroke-width="2"
							d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
						/>
					</svg>
				</div>
			</div>
		</Card>
	{/if}

	<!-- Cost Trends -->
	{#if trendChartData}
		<Card>
			<div class="flex justify-between items-center mb-4">
				<h3 class="text-lg font-semibold">Cost Trends</h3>
				<div class="flex gap-2">
					<button
						onclick={() => { period = 'daily'; loadData(); }}
						class="px-3 py-1 rounded-md text-sm"
						class:bg-blue-500={period === 'daily'}
						class:text-white={period === 'daily'}
						class:bg-gray-200={period !== 'daily'}
						class:dark:bg-gray-700={period !== 'daily'}
					>
						Daily
					</button>
					<button
						onclick={() => { period = 'weekly'; loadData(); }}
						class="px-3 py-1 rounded-md text-sm"
						class:bg-blue-500={period === 'weekly'}
						class:text-white={period === 'weekly'}
						class:bg-gray-200={period !== 'weekly'}
						class:dark:bg-gray-700={period !== 'weekly'}
					>
						Weekly
					</button>
				</div>
			</div>
			<LineChart
				title=""
				xAxis={trendChartData.xAxis}
				yAxis={trendChartData.yAxis}
				series={trendChartData.series}
				height={300}
			/>
		</Card>
	{/if}

	<!-- Cost by Symbol and Token Distribution by Agent -->
	<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
		{#if symbolChartData}
			<Card>
				<h3 class="text-lg font-semibold mb-4">Cost by Symbol (Top 10)</h3>
				<BarChart
					title=""
					xAxis={symbolChartData.xAxis}
					yAxis={symbolChartData.yAxis}
					series={symbolChartData.series}
					height={300}
				/>
			</Card>
		{/if}

		{#if agentPieData}
			<Card>
				<h3 class="text-lg font-semibold mb-4">Token Distribution by Agent</h3>
				<PieChart
					title=""
					data={agentPieData}
					height={300}
				/>
			</Card>
		{/if}
	</div>

	<!-- Model Cost Comparison -->
	{#if modelChartData}
		<Card>
			<h3 class="text-lg font-semibold mb-4">Model Cost Comparison</h3>
			<BarChart
				title=""
				xAxis={modelChartData.xAxis}
				yAxis={modelChartData.yAxis}
				series={modelChartData.series}
				height={300}
			/>
			<p class="text-sm text-gray-600 dark:text-gray-400 mt-2">
				Note: Ollama models show $0.00 as they run locally
			</p>
		</Card>
	{/if}

	<!-- Loading State -->
	{#if state.loading}
		<div class="flex justify-center items-center py-12">
			<div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
		</div>
	{/if}

	<!-- Error State -->
	{#if state.error}
		<Card>
			<div class="text-red-600 dark:text-red-400">
				<p class="font-semibold">Error</p>
				<p class="text-sm">{state.error}</p>
			</div>
		</Card>
	{/if}

	<!-- Empty State -->
	{#if !state.loading && !state.error && !summary}
		<Card>
			<div class="text-center py-12 text-gray-600 dark:text-gray-400">
				<p>No data available for the selected date range</p>
				<p class="text-sm mt-2">Try adjusting the date range or run some analyses</p>
			</div>
		</Card>
	{/if}
</div>
