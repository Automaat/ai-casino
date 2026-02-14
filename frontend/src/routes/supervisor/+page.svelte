<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import MetricCard from '$lib/components/ui/MetricCard.svelte';
	import Card from '$lib/components/ui/Card.svelte';
	import DataTable from '$lib/components/ui/DataTable.svelte';
	import {
		recentMetrics,
		summary,
		workerStats,
		errors,
		fetchAllSupervisorMetrics
	} from '$lib/stores/supervisor';

	let wsConnected = $state(false);
	let ws: WebSocket | null = null;
	let refreshInterval: number | null = null;

	const summaryData = $derived($summary);
	const recentData = $derived($recentMetrics?.metrics || []);
	const workerData = $derived($workerStats);
	const errorsData = $derived($errors?.errors || []);

	// Computed metrics
	const avgRoutingTime = $derived(summaryData ? summaryData.avg_routing_ms.toFixed(1) : '0.0');
	const efficiency = $derived(summaryData ? summaryData.avg_efficiency_percent.toFixed(1) : '0');
	const totalCost = $derived(recentData.length > 0
		? recentData.reduce((sum, m) => sum + m.total_cost_usd, 0).toFixed(4)
		: '0.00');
	const avgWorkers = $derived(summaryData ? summaryData.avg_workers_per_cycle.toFixed(1) : '0.0');

	// Routing breakdown
	const routingBreakdown = $derived.by(() => {
		if (!recentData.length) return [];
		const counts: Record<string, number> = {};
		recentData.forEach(m => {
			[...m.required_analyses, ...m.optional_analyses].forEach(a => {
				counts[a] = (counts[a] || 0) + 1;
			});
		});
		const total = Object.values(counts).reduce((sum, v) => sum + v, 0);
		return Object.entries(counts).map(([type, count]) => ({
			type: type.charAt(0).toUpperCase() + type.slice(1),
			count,
			percentage: total > 0 ? ((count / total) * 100).toFixed(1) : '0'
		}));
	});

	// Worker performance table
	const workerPerformance = $derived.by(() => {
		if (!workerData) return [];
		return Object.entries(workerData.worker_stats).map(([name, stats]) => ({
			name,
			executions: stats.total_executions,
			success_rate: stats.success_rate.toFixed(1),
			avg_latency: stats.avg_duration_ms.toFixed(0),
			p95: (stats.avg_duration_ms * 1.5).toFixed(0)
		}));
	});

	// Recent routing columns
	const routingColumns = [
		{ key: 'symbol', label: 'Symbol', class: 'font-medium' },
		{ key: 'timestamp', label: 'Time', format: (v: string) => new Date(v).toLocaleTimeString() },
		{ key: 'total_workers', label: 'Workers' },
		{ key: 'total_cost_usd', label: 'Cost', format: (v: number) => `$${v.toFixed(4)}` },
		{ key: 'routing_time_ms', label: 'Routing (ms)', format: (v: number) => v.toFixed(1) }
	];

	const breakdownColumns = [
		{ key: 'type', label: 'Analysis Type', class: 'font-medium' },
		{ key: 'count', label: 'Count' },
		{ key: 'percentage', label: 'Percentage', format: (v: string) => `${v}%` }
	];

	const workerColumns = [
		{ key: 'name', label: 'Worker', class: 'font-medium' },
		{ key: 'executions', label: 'Executions' },
		{ key: 'success_rate', label: 'Success Rate', format: (v: string) => `${v}%` },
		{ key: 'avg_latency', label: 'Avg Latency (ms)' },
		{ key: 'p95', label: 'P95 (ms)' }
	];

	function connectWebSocket() {
		const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
		const wsUrl = `${protocol}//${window.location.host.replace(':5173', ':8484')}/ws/events`;

		ws = new WebSocket(wsUrl);

		ws.onopen = () => {
			wsConnected = true;
		};

		ws.onmessage = (event) => {
			const msg = JSON.parse(event.data);
			if (msg.event_type === 'CYCLE_COMPLETE' || msg.event_type === 'ANALYSIS_COMPLETE') {
				fetchAllSupervisorMetrics();
			}
		};

		ws.onerror = () => {
			wsConnected = false;
		};

		ws.onclose = () => {
			wsConnected = false;
			setTimeout(connectWebSocket, 5000);
		};
	}

	onMount(() => {
		fetchAllSupervisorMetrics();
		connectWebSocket();
		refreshInterval = setInterval(() => fetchAllSupervisorMetrics(), 30000);
	});

	onDestroy(() => {
		if (ws) ws.close();
		if (refreshInterval) clearInterval(refreshInterval);
	});
</script>

<svelte:head>
	<title>Supervisor Metrics - AI Casino</title>
</svelte:head>

<div class="space-y-8">
	<!-- Header -->
	<div class="flex justify-between items-center">
		<div>
			<h1 class="text-3xl font-bold text-gray-900">Supervisor Metrics Dashboard</h1>
			<p class="text-gray-600 mt-1">Routing decisions and worker performance</p>
		</div>
		<div class="flex items-center gap-2">
			<div class={`w-3 h-3 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
			<span class="text-sm text-gray-600">{wsConnected ? 'Live' : 'Disconnected'}</span>
		</div>
	</div>

	<!-- Stats Cards -->
	<div class="grid grid-cols-1 md:grid-cols-4 gap-6">
		<MetricCard title="Avg Routing Time" value={`${avgRoutingTime}ms`} icon="⚡" />
		<MetricCard title="Efficiency" value={`${efficiency}%`} icon="🎯" />
		<MetricCard title="Total Cost (24h)" value={`$${totalCost}`} icon="💰" />
		<MetricCard title="Avg Workers" value={avgWorkers} icon="👥" />
	</div>

	<!-- Routing Breakdown -->
	<Card title="Routing Breakdown">
		{#if routingBreakdown.length > 0}
			<DataTable data={routingBreakdown} columns={breakdownColumns} />
		{:else}
			<div class="text-center py-12 text-gray-600">No routing data available</div>
		{/if}
	</Card>

	<!-- Worker Performance -->
	<Card title="Worker Performance (24h)">
		{#if workerPerformance.length > 0}
			<DataTable data={workerPerformance} columns={workerColumns} />
		{:else}
			<div class="text-center py-12 text-gray-600">No worker data available</div>
		{/if}
	</Card>

	<!-- Recent Routing Decisions -->
	<Card title="Recent Routing Decisions">
		{#if recentData.length > 0}
			<div class="space-y-4">
				{#each recentData as metric}
					<div class="border border-gray-200 rounded-lg p-4 bg-gray-50">
						<div class="flex justify-between items-start mb-2">
							<div>
								<span class="font-semibold text-gray-900">{metric.symbol}</span>
								<span class="text-sm text-gray-500 ml-2">
									{new Date(metric.timestamp).toLocaleTimeString()}
								</span>
							</div>
							<div class="text-sm text-gray-600">
								{metric.total_workers} workers | ${metric.total_cost_usd.toFixed(4)}
							</div>
						</div>
						<div class="grid grid-cols-2 gap-4 text-sm mb-2">
							<div>
								<span class="text-gray-600">Required:</span>
								<span class="text-gray-900">{metric.required_analyses.join(', ')}</span>
							</div>
							<div>
								<span class="text-gray-600">Optional:</span>
								<span class="text-gray-900">
									{metric.optional_analyses.length > 0 ? metric.optional_analyses.join(', ') : 'None'}
								</span>
							</div>
						</div>
						{#if metric.reasoning}
							<div class="text-sm text-gray-700">
								<span class="text-gray-600">Reasoning:</span>
								{metric.reasoning}
							</div>
						{/if}
						{#if metric.errors.length > 0}
							<div class="text-sm text-red-600 mt-2">
								<span class="font-medium">Errors:</span>
								{metric.errors.join(', ')}
							</div>
						{/if}
					</div>
				{/each}
			</div>
		{:else}
			<div class="text-center py-12 text-gray-600">No recent routing decisions</div>
		{/if}
	</Card>
</div>
