<script lang="ts">
	import { onMount } from 'svelte';
	import Card from '$lib/components/ui/Card.svelte';
	import MetricCard from '$lib/components/ui/MetricCard.svelte';
	import DataTable from '$lib/components/ui/DataTable.svelte';
	import TreemapChart from '$lib/components/charts/TreemapChart.svelte';
	import LineChart from '$lib/components/charts/LineChart.svelte';
	import { positions } from '$lib/stores/dashboard';
	import { api } from '$lib/api/client';
	import { formatCurrency, formatPercent } from '$lib/utils/format';
	import type { PositionResponse, SnapshotRecord } from '$lib/types/api';

	$: portfolio = $positions;
	$: positionsList = portfolio?.positions || [];
	
	let snapshots: SnapshotRecord[] = [];
	let loading = true;

	onMount(async () => {
		try {
			const data = await api.getSnapshots(30);
			snapshots = data.snapshots;
		} catch (error) {
			console.error('Failed to load snapshots:', error);
		} finally {
			loading = false;
		}
	});

	// Equity curve data
	$: equityData = snapshots.map(s => ({
		time: new Date(s.timestamp),
		value: s.portfolio_value
	}));

	// Treemap data
	$: treemapData = positionsList.map(p => ({
		name: p.symbol,
		value: p.market_value
	}));

	const positionColumns = [
		{ 
			key: 'symbol' as keyof PositionResponse, 
			label: 'Symbol',
			class: 'font-medium'
		},
		{ 
			key: 'quantity' as keyof PositionResponse, 
			label: 'Quantity',
			format: (value: number) => value.toFixed(0)
		},
		{ 
			key: 'avg_entry_price' as keyof PositionResponse, 
			label: 'Avg Entry',
			format: (value: number) => formatCurrency(value)
		},
		{ 
			key: 'current_price' as keyof PositionResponse, 
			label: 'Current Price',
			format: (value: number) => formatCurrency(value)
		},
		{ 
			key: 'market_value' as keyof PositionResponse, 
			label: 'Market Value',
			format: (value: number) => formatCurrency(value)
		},
		{ 
			key: 'unrealized_pnl' as keyof PositionResponse, 
			label: 'P&L',
			format: (value: number) => formatCurrency(value),
			class: 'font-semibold'
		},
		{ 
			key: 'unrealized_pnl_percent' as keyof PositionResponse, 
			label: 'P&L %',
			format: (value: number) => formatPercent(value / 100)
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
			value={portfolio ? formatCurrency(portfolio.portfolio_value) : '$0'}
			icon="💰"
		/>
		<MetricCard
			title="Cash"
			value={portfolio ? formatCurrency(portfolio.cash) : '$0'}
			icon="💵"
		/>
		<MetricCard
			title="Positions"
			value={portfolio?.positions.length ?? 0}
			icon="📈"
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
					color="#10b981"
					yAxisLabel="Portfolio Value ($)"
				/>
			{:else if loading}
				<div class="text-center py-12 text-slate-400">Loading...</div>
			{:else}
				<div class="text-center py-12 text-slate-400">No historical data</div>
			{/if}
		</Card>

		<!-- Allocation Treemap -->
		<Card title="Allocation">
			{#if treemapData.length > 0}
				<TreemapChart data={treemapData} height={300} title="" />
			{:else}
				<div class="text-center py-12 text-slate-400">No positions</div>
			{/if}
		</Card>
	</div>

	<!-- Positions Table -->
	<Card title="Positions">
		{#if positionsList.length > 0}
			<DataTable data={positionsList} columns={positionColumns} />
		{:else}
			<div class="text-center py-12 text-slate-400">
				No active positions. Start trading to see positions here.
			</div>
		{/if}
	</Card>
</div>
