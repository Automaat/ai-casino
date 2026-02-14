<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import MetricCard from '$lib/components/ui/MetricCard.svelte';
	import Card from '$lib/components/ui/Card.svelte';
	import DataTable from '$lib/components/ui/DataTable.svelte';
	import PieChart from '$lib/components/charts/PieChart.svelte';
	import BarChart from '$lib/components/charts/BarChart.svelte';
	import { discoveryInsights, fetchDiscoveryInsights } from '$lib/stores/discovery';

	let wsConnected = $state(false);
	let ws: WebSocket | null = null;
	let refreshInterval: number | null = null;

	const insights = $derived($discoveryInsights);

	// Pie chart data for source breakdown
	const pieChartData = $derived.by(() => {
		if (!insights?.source_breakdown) return [];
		const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4'];
		return insights.source_breakdown.map((source, idx) => ({
			label: source.source.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
			value: source.count,
			color: colors[idx % colors.length]
		}));
	});

	// Bar chart data for composite scores distribution
	const scoreDistribution = $derived.by(() => {
		if (!insights?.recent_discoveries) return [];
		const buckets = { '0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0 };
		insights.recent_discoveries.forEach(d => {
			const score = d.composite_score;
			if (score < 0.2) buckets['0.0-0.2']++;
			else if (score < 0.4) buckets['0.2-0.4']++;
			else if (score < 0.6) buckets['0.4-0.6']++;
			else if (score < 0.8) buckets['0.6-0.8']++;
			else buckets['0.8-1.0']++;
		});
		return Object.entries(buckets).map(([label, value]) => ({ label, value }));
	});

	// Recent discoveries table
	const discoveryColumns = [
		{ key: 'symbol' as const, label: 'Symbol', class: 'font-medium' },
		{ key: 'discovered_at' as const, label: 'Discovered', format: (v: string) => new Date(v).toLocaleDateString() },
		{ key: 'composite_score' as const, label: 'Score', format: (v: number) => v.toFixed(3) },
		{ key: 'sources' as const, label: 'Sources', format: (v: string[]) => v.length.toString() },
		{ key: 'added_to_watchlist' as const, label: 'Added', format: (v: boolean) => v ? '✓' : '–' },
		{ key: 'first_signal' as const, label: 'Signal', format: (v: string | null) => v || '–' },
		{ key: 'outcome_7d' as const, label: '7d Return', format: (v: number | null) => v !== null ? `${(v * 100).toFixed(1)}%` : '–' },
		{ key: 'outcome_30d' as const, label: '30d Return', format: (v: number | null) => v !== null ? `${(v * 100).toFixed(1)}%` : '–' }
	];

	const WS_URL =
		import.meta.env.VITE_WS_URL ||
		(typeof window !== 'undefined'
			? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`
			: 'ws://localhost:8484');

	let reconnectTimeout: number | null = null;
	let reconnectAttempts = 0;
	const MAX_RECONNECT_ATTEMPTS = 5;
	const RECONNECT_DELAY = 3000;

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
				const msg = JSON.parse(event.data);
				if (msg.event_type === 'CYCLE_COMPLETE') {
					fetchDiscoveryInsights();
				}
			} catch (error) {
				console.error('Failed to parse WebSocket message:', error);
			}
		};

		ws.onerror = (error) => {
			console.error('WebSocket error:', error);
			wsConnected = false;
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
	}

	onMount(() => {
		fetchDiscoveryInsights();
		connectWebSocket();
		refreshInterval = setInterval(() => fetchDiscoveryInsights(), 60000); // Refresh every minute
	});

	onDestroy(() => {
		if (reconnectTimeout) clearTimeout(reconnectTimeout);
		if (ws) ws.close();
		if (refreshInterval) clearInterval(refreshInterval);
	});
</script>

<svelte:head>
	<title>Discovery Insights - AI Casino</title>
</svelte:head>

<div class="space-y-8">
	<!-- Header -->
	<div class="flex justify-between items-center">
		<div>
			<h1 class="text-3xl font-bold text-gray-900">Discovery Insights Dashboard</h1>
			<p class="text-gray-600 mt-1">Multi-source stock discovery analytics and success tracking</p>
		</div>
		<div class="flex items-center gap-2">
			<div class={`w-3 h-3 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
			<span class="text-sm text-gray-600">{wsConnected ? 'Live' : 'Disconnected'}</span>
		</div>
	</div>

	<!-- Stats Cards -->
	<div class="grid grid-cols-1 md:grid-cols-4 gap-6">
		<MetricCard title="Total Discovered" value={insights?.total_discoveries?.toString() || '0'} icon="🔍" />
		<MetricCard title="Added to Watchlist" value={insights?.success_metrics?.added_to_watchlist?.toString() || '0'} icon="⭐" />
		<MetricCard title="Received Signal" value={insights?.success_metrics?.received_signal?.toString() || '0'} icon="📊" />
		<MetricCard title="Signal Rate" value={`${insights?.success_metrics?.signal_rate || 0}%`} icon="🎯" />
	</div>

	<!-- Charts Row -->
	<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
		<!-- Source Attribution Pie Chart -->
		<Card title="Discovery Source Breakdown">
			{#if pieChartData.length > 0}
				<PieChart data={pieChartData} height={350} />
			{:else}
				<div class="text-center py-12 text-gray-600">No discovery data available</div>
			{/if}
		</Card>

		<!-- Composite Score Distribution -->
		<Card title="Composite Score Distribution">
			{#if scoreDistribution.length > 0}
				<BarChart data={scoreDistribution} height={350} yAxisLabel="Count" xAxisLabel="Score Range" />
			{:else}
				<div class="text-center py-12 text-gray-600">No score data available</div>
			{/if}
		</Card>
	</div>

	<!-- Success Metrics Summary -->
	<Card title="Success Rate Tracking">
		<div class="grid grid-cols-1 md:grid-cols-3 gap-6 p-4">
			<div class="text-center">
				<div class="text-3xl font-bold text-blue-600">
					{insights?.total_discoveries || 0}
				</div>
				<div class="text-sm text-gray-600 mt-1">Total Discovered</div>
			</div>
			<div class="text-center">
				<div class="text-3xl font-bold text-green-600">
					{insights?.success_metrics?.added_to_watchlist || 0}
				</div>
				<div class="text-sm text-gray-600 mt-1">Added to Watchlist</div>
				<div class="text-xs text-gray-500 mt-1">
					{insights?.total_discoveries ?
						((insights?.success_metrics?.added_to_watchlist / insights?.total_discoveries) * 100).toFixed(1)
						: '0'}% conversion
				</div>
			</div>
			<div class="text-center">
				<div class="text-3xl font-bold text-purple-600">
					{insights?.success_metrics?.received_signal || 0}
				</div>
				<div class="text-sm text-gray-600 mt-1">Received Trading Signal</div>
				<div class="text-xs text-gray-500 mt-1">
					{insights?.success_metrics?.signal_rate || 0}% signal rate
				</div>
			</div>
		</div>
	</Card>

	<!-- Recent Discoveries Table -->
	<Card title="Recent Discoveries (Last 50)">
		{#if insights?.recent_discoveries && insights.recent_discoveries.length > 0}
			<DataTable data={insights.recent_discoveries} columns={discoveryColumns} />
		{:else}
			<div class="text-center py-12 text-gray-600">No recent discoveries</div>
		{/if}
	</Card>

	<!-- Discovery Timeline Stats -->
	<Card title="Discovery Statistics">
		<div class="grid grid-cols-1 md:grid-cols-2 gap-6 p-4">
			<div>
				<h3 class="text-lg font-semibold text-gray-900 mb-3">Average Composite Score</h3>
				<div class="text-4xl font-bold text-blue-600">
					{insights?.avg_composite_score?.toFixed(3) || '0.000'}
				</div>
				<p class="text-sm text-gray-600 mt-2">
					Higher scores indicate stronger multi-source confirmation
				</p>
			</div>
			<div>
				<h3 class="text-lg font-semibold text-gray-900 mb-3">Source Diversity</h3>
				<div class="text-4xl font-bold text-green-600">
					{pieChartData.length}
				</div>
				<p class="text-sm text-gray-600 mt-2">
					Active discovery sources contributing to watchlist
				</p>
			</div>
		</div>
	</Card>
</div>
