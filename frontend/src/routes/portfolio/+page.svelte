<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import Card from '$lib/components/ui/Card.svelte';
	import MetricCard from '$lib/components/ui/MetricCard.svelte';
	import DataTable from '$lib/components/ui/DataTable.svelte';
	import TreemapChart from '$lib/components/charts/TreemapChart.svelte';
	import LineChart from '$lib/components/charts/LineChart.svelte';
	import RebalanceChart from '$lib/components/charts/RebalanceChart.svelte';
	import { positions } from '$lib/stores/dashboard';
	import { api } from '$lib/api/client';
	import { formatCurrency, formatPercent } from '$lib/utils/format';
	import type { PositionResponse, SnapshotsResponse, RebalanceResponse } from '$lib/types/api';

	type EnhancedPosition = PositionResponse & {
		market_value: number;
		unrealized_pnl: number;
		unrealized_pnl_percent: number;
	};

	$: portfolio = $positions;
	$: positionsList = (portfolio?.positions || []).map((p): EnhancedPosition => ({
		...p,
		market_value: p.current_qty * p.current_price,
		unrealized_pnl: (p.current_price - p.entry_price) * p.current_qty,
		unrealized_pnl_percent: p.entry_price ? ((p.current_price / p.entry_price) - 1) * 100 : 0
	}));

	// Portfolio totals
	$: portfolioValue = positionsList.reduce((sum, p) => sum + p.market_value, 0);
	$: totalPnl = positionsList.reduce((sum, p) => sum + p.unrealized_pnl, 0);

	let snapshotsData: SnapshotsResponse | null = null;
	let rebalance: RebalanceResponse | null = null;
	let loading = true;

	async function loadData() {
		loading = true;
		try {
			const [snaps, rebal] = await Promise.all([
				api.getSnapshots(30),
				api.getRebalance()
			]);
			snapshotsData = snaps;
			rebalance = rebal;
		} catch (error) {
			console.error('Failed to load portfolio data:', error);
		} finally {
			loading = false;
		}
	}

	onMount(() => {
		loadData();
	});

	// Refetch when route changes (handles tab switching)
	$: if ($page.url.pathname === '/portfolio') {
		loadData();
	}

	// Equity curve data
	$: equityData = (snapshotsData?.snapshots || []).map(s => ({
		time: new Date(s.timestamp),
		value: s.portfolio_value
	}));

	// Treemap data
	$: treemapData = positionsList.map(p => ({
		name: p.symbol,
		value: p.market_value
	}));

	// Stop-loss coverage percentage
	$: stopLossCoverage = positionsList.length > 0
		? (positionsList.filter(p => p.current_stop_loss > 0).length / positionsList.length) * 100
		: 0;

	// Average entry confidence
	$: avgConfidence = positionsList.length > 0
		? positionsList.reduce((sum, p) => sum + p.entry_confidence, 0) / positionsList.length
		: 0;

	// Position age distribution
	$: positionAgeDistribution = (() => {
		const buckets = { '0-7d': 0, '8-30d': 0, '31-90d': 0, '90+d': 0 };
		positionsList.forEach((p: EnhancedPosition) => {
			if (p.days_held <= 7) buckets['0-7d']++;
			else if (p.days_held <= 30) buckets['8-30d']++;
			else if (p.days_held <= 90) buckets['31-90d']++;
			else buckets['90+d']++;
		});
		return Object.entries(buckets)
			.filter(([_, count]) => count > 0)
			.map(([range, count]) => `${range}: ${count}`)
			.join(' | ') || 'No positions';
	})();

	const positionColumns: Array<{
		key: keyof EnhancedPosition;
		label: string;
		format?: (value: any) => string;
		class?: string;
	}> = [
		{
			key: 'symbol',
			label: 'Symbol',
			class: 'font-medium'
		},
		{
			key: 'current_qty',
			label: 'Quantity',
			format: (value: number) => value.toFixed(0)
		},
		{
			key: 'entry_price',
			label: 'Avg Entry',
			format: (value: number) => formatCurrency(value)
		},
		{
			key: 'current_price',
			label: 'Current Price',
			format: (value: number) => formatCurrency(value)
		},
		{
			key: 'market_value',
			label: 'Market Value',
			format: (value: number) => formatCurrency(value)
		},
		{
			key: 'unrealized_pnl',
			label: 'P&L',
			format: (value: number) => formatCurrency(value),
			class: 'font-semibold'
		},
		{
			key: 'unrealized_pnl_percent',
			label: 'P&L %',
			format: (value: number) => formatPercent(value / 100)
		},
		{
			key: 'current_stop_loss',
			label: 'Stop Loss',
			format: (value: number) => (value === 0 ? 'Not set' : formatCurrency(value))
		},
		{
			key: 'entry_confidence',
			label: 'Entry Conf',
			format: (value: number) => formatPercent(value)
		},
		{
			key: 'days_held',
			label: 'Days Held',
			format: (value: number) => `${value}d`
		}
	];
</script>

<svelte:head>
	<title>Portfolio - AI Casino</title>
</svelte:head>

<div class="space-y-8">
	<!-- Portfolio Metrics -->
	<div class="grid grid-cols-1 md:grid-cols-3 gap-6">
		<MetricCard
			title="Portfolio Value"
			value={formatCurrency(portfolioValue)}
			icon="💰"
		/>
		<MetricCard
			title="Total P&L"
			value={formatCurrency(totalPnl)}
			icon="💵"
		/>
		<MetricCard
			title="Positions"
			value={positionsList.length}
			icon="📈"
		/>
	</div>

	<!-- Additional Metrics -->
	<div class="grid grid-cols-1 md:grid-cols-3 gap-6">
		<MetricCard
			title="Avg Confidence"
			value={formatPercent(avgConfidence)}
			icon="🎯"
		/>
		<MetricCard
			title="Stop-Loss Coverage"
			value={formatPercent(stopLossCoverage / 100)}
			icon="🛡️"
		/>
		<MetricCard
			title="Position Age"
			value="Distribution"
			subtitle={positionAgeDistribution}
			icon="📅"
		/>
	</div>

	<!-- Charts Row -->
	<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
		<!-- Equity Curve -->
		<Card title="Equity Curve">
			{#if !loading && equityData.length > 0}
				<LineChart
					data={equityData}
					height={300}
					color="#059669"
					yAxisLabel="Portfolio Value ($)"
				/>
			{:else if loading}
				<div class="text-center py-12 text-gray-600">Loading...</div>
			{:else if !snapshotsData?.database_enabled}
				<div class="text-center py-12 text-gray-600">
					<div class="font-medium">Database persistence disabled</div>
					<div class="text-sm mt-2">Enable in daemon config: <code class="bg-gray-100 px-1 rounded">database.enable_persistence: true</code></div>
				</div>
			{:else if !snapshotsData?.has_trades}
				<div class="text-center py-12 text-gray-600">
					<div class="font-medium">No trades executed yet</div>
					<div class="text-sm mt-2">Equity curve will populate after first trade execution</div>
				</div>
			{:else}
				<div class="text-center py-12 text-gray-600">No historical data</div>
			{/if}
		</Card>

		<!-- Allocation Treemap -->
		<Card title="Allocation">
			{#if treemapData.length > 0}
				<TreemapChart data={treemapData} height={300} title="" />
			{:else}
				<div class="text-center py-12 text-gray-600">No positions</div>
			{/if}
		</Card>
	</div>

	<!-- Rebalance Chart -->
	<Card title="Rebalance Analysis">
		{#if rebalance && rebalance.allocations.length > 0}
			<RebalanceChart allocations={rebalance.allocations} height={350} />
		{:else if loading}
			<div class="text-center py-12 text-gray-600">Loading...</div>
		{:else if !rebalance?.enabled}
			<div class="text-center py-12 text-gray-600">
				<div class="font-medium">Rebalancing disabled</div>
				<div class="text-sm mt-2">Enable in daemon config: <code class="bg-gray-100 px-1 rounded">rebalancing.enabled: true</code></div>
			</div>
		{:else}
			<div class="text-center py-12 text-gray-600">
				<div class="font-medium">No rebalancing data yet</div>
				<div class="text-sm mt-2">Waiting for first scheduled rebalancing run</div>
			</div>
		{/if}
	</Card>

	<!-- Positions Table -->
	<Card title="Positions">
		{#if positionsList.length > 0}
			<DataTable data={positionsList} columns={positionColumns} />
		{:else}
			<div class="text-center py-12 text-gray-600">
				No active positions. Start trading to see positions here.
			</div>
		{/if}
	</Card>
</div>
