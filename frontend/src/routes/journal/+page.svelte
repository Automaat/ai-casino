<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import Card from '$lib/components/ui/Card.svelte';
	import Badge from '$lib/components/ui/Badge.svelte';
	import DataTable from '$lib/components/ui/DataTable.svelte';
	import MetricCard from '$lib/components/ui/MetricCard.svelte';
	import { api } from '$lib/api/client';
	import { formatCurrency, formatPercent, formatDate } from '$lib/utils/format';
	import type { TradeResponse, TradesResponse } from '$lib/types/api';

	let tradesData: TradesResponse | null = null;
	let loading = true;
	let error = '';

	// Filters
	let selectedSymbol = '';
	let selectedStatus = '';
	let selectedRisk = '';
	let window: 'all' | '30d' | '7d' = 'all';

	$: allTrades = tradesData?.trades || [];

	// Filter trades
	$: filteredTrades = allTrades.filter(t => {
		if (selectedSymbol && t.symbol !== selectedSymbol) return false;
		if (selectedStatus && t.status !== selectedStatus) return false;
		if (selectedRisk && t.risk_level !== selectedRisk) return false;
		return true;
	});

	// Unique symbols for filter
	$: symbols = [...new Set(allTrades.map(t => t.symbol))].sort();

	// Stats
	$: closedTrades = filteredTrades.filter(t => t.status === 'CLOSED');
	$: winningTrades = closedTrades.filter(t => t.pnl && t.pnl > 0);
	$: losingTrades = closedTrades.filter(t => t.pnl && t.pnl < 0);
	$: winRate = closedTrades.length > 0 ? winningTrades.length / closedTrades.length : 0;
	$: totalPnl = closedTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);
	$: avgPnl = closedTrades.length > 0 ? totalPnl / closedTrades.length : 0;
	$: avgWin = winningTrades.length > 0 ? winningTrades.reduce((sum, t) => sum + (t.pnl || 0), 0) / winningTrades.length : 0;
	$: avgLoss = losingTrades.length > 0 ? Math.abs(losingTrades.reduce((sum, t) => sum + (t.pnl || 0), 0)) / losingTrades.length : 0;
	$: profitFactor = avgLoss > 0 ? avgWin / avgLoss : 0;

	async function loadData() {
		loading = true;
		error = '';
		try {
			tradesData = await api.getTrades({ limit: 500, window });
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load trades';
			console.error('Failed to load trades:', e);
		} finally {
			loading = false;
		}
	}

	function clearFilters() {
		selectedSymbol = '';
		selectedStatus = '';
		selectedRisk = '';
	}

	onMount(() => {
		loadData();
	});

	// Refetch when route changes
	$: if ($page.url.pathname === '/journal') {
		loadData();
	}

	// Refetch when window changes
	$: if (window) {
		loadData();
	}

	const columns = [
		{
			key: 'timestamp' as keyof TradeResponse,
			label: 'Date',
			format: (value: string) => formatDate(value)
		},
		{
			key: 'symbol' as keyof TradeResponse,
			label: 'Symbol',
			class: 'font-medium'
		},
		{
			key: 'action' as keyof TradeResponse,
			label: 'Action',
			format: (value: string) => value
		},
		{
			key: 'status' as keyof TradeResponse,
			label: 'Status',
			format: (value: string) => value
		},
		{
			key: 'entry_price' as keyof TradeResponse,
			label: 'Entry',
			format: (value: number) => formatCurrency(value)
		},
		{
			key: 'exit_price' as keyof TradeResponse,
			label: 'Exit',
			format: (value: number | null) => value ? formatCurrency(value) : '-'
		},
		{
			key: 'shares' as keyof TradeResponse,
			label: 'Shares',
			format: (value: number) => value.toString()
		},
		{
			key: 'pnl' as keyof TradeResponse,
			label: 'P/L',
			format: (value: number | null) => {
				if (value === null) return '-';
				return formatCurrency(value);
			},
			cellClass: (value: number | null) => {
				if (value === null) return '';
				return value > 0 ? 'text-green-600 font-semibold' : value < 0 ? 'text-red-600 font-semibold' : '';
			}
		},
		{
			key: 'pnl_percent' as keyof TradeResponse,
			label: 'P/L %',
			format: (value: number | null) => {
				if (value === null) return '-';
				return formatPercent(value);
			},
			cellClass: (value: number | null) => {
				if (value === null) return '';
				return value > 0 ? 'text-green-600' : value < 0 ? 'text-red-600' : '';
			}
		},
		{
			key: 'confidence' as keyof TradeResponse,
			label: 'Confidence',
			format: (value: number) => formatPercent(value)
		},
		{
			key: 'risk_level' as keyof TradeResponse,
			label: 'Risk',
			format: (value: string) => value
		},
		{
			key: 'strategy_name' as keyof TradeResponse,
			label: 'Strategy',
			format: (value: string | null) => value || '-'
		}
	];
</script>

<svelte:head>
	<title>Trade Journal - AI Casino</title>
</svelte:head>

<div class="space-y-8">
	<!-- Filters -->
	<Card title="Filters">
		<div class="space-y-4">
			<!-- Time Window Presets -->
			<div>
				<label class="block text-sm font-medium text-gray-600 mb-2">
					Time Window
				</label>
				<div class="flex gap-2">
					<button
						on:click={() => window = '7d'}
						class="px-4 py-2 {window === '7d' ? 'bg-blue-700 text-white' : 'bg-gray-100 text-black hover:bg-gray-50'} border border-gray-300 rounded-lg text-sm transition-colors"
					>
						Last 7 days
					</button>
					<button
						on:click={() => window = '30d'}
						class="px-4 py-2 {window === '30d' ? 'bg-blue-700 text-white' : 'bg-gray-100 text-black hover:bg-gray-50'} border border-gray-300 rounded-lg text-sm transition-colors"
					>
						Last 30 days
					</button>
					<button
						on:click={() => window = 'all'}
						class="px-4 py-2 {window === 'all' ? 'bg-blue-700 text-white' : 'bg-gray-100 text-black hover:bg-gray-50'} border border-gray-300 rounded-lg text-sm transition-colors"
					>
						All time
					</button>
				</div>
			</div>

			<!-- Symbol, Status, Risk Filters -->
			<div class="grid grid-cols-1 md:grid-cols-3 gap-4">
				<div>
					<label for="symbol-filter" class="block text-sm font-medium text-gray-600 mb-2">
						Symbol
					</label>
					<select
						id="symbol-filter"
						bind:value={selectedSymbol}
						class="w-full px-3 py-2 bg-gray-100 border border-gray-300 rounded-lg text-black focus:outline-none focus:ring-2 focus:ring-blue-700"
					>
						<option value="">All Symbols</option>
						{#each symbols as symbol}
							<option value={symbol}>{symbol}</option>
						{/each}
					</select>
				</div>
				<div>
					<label for="status-filter" class="block text-sm font-medium text-gray-600 mb-2">
						Status
					</label>
					<select
						id="status-filter"
						bind:value={selectedStatus}
						class="w-full px-3 py-2 bg-gray-100 border border-gray-300 rounded-lg text-black focus:outline-none focus:ring-2 focus:ring-blue-700"
					>
						<option value="">All Statuses</option>
						<option value="OPEN">OPEN</option>
						<option value="CLOSED">CLOSED</option>
						<option value="REJECTED">REJECTED</option>
					</select>
				</div>
				<div>
					<label for="risk-filter" class="block text-sm font-medium text-gray-600 mb-2">
						Risk Level
					</label>
					<select
						id="risk-filter"
						bind:value={selectedRisk}
						class="w-full px-3 py-2 bg-gray-100 border border-gray-300 rounded-lg text-black focus:outline-none focus:ring-2 focus:ring-blue-700"
					>
						<option value="">All Risk Levels</option>
						<option value="LOW">LOW</option>
						<option value="MEDIUM">MEDIUM</option>
						<option value="HIGH">HIGH</option>
					</select>
				</div>
			</div>

			<!-- Clear Filters Button -->
			<div>
				<button
					on:click={clearFilters}
					class="px-4 py-2 bg-red-100 hover:bg-red-200 border border-red-300 rounded-lg text-red-800 text-sm transition-colors"
				>
					Clear All Filters
				</button>
			</div>
		</div>
	</Card>

	<!-- Performance Stats -->
	<div class="grid grid-cols-1 md:grid-cols-4 gap-4">
		<MetricCard title="Total Trades" value={filteredTrades.length.toString()} />
		<MetricCard
			title="Win Rate"
			value={formatPercent(winRate)}
			trend={winRate >= 0.5 ? 'up' : 'down'}
		/>
		<MetricCard
			title="Total P/L"
			value={formatCurrency(totalPnl)}
			trend={totalPnl > 0 ? 'up' : totalPnl < 0 ? 'down' : 'neutral'}
		/>
		<MetricCard
			title="Avg P/L"
			value={formatCurrency(avgPnl)}
			trend={avgPnl > 0 ? 'up' : avgPnl < 0 ? 'down' : 'neutral'}
		/>
	</div>

	<!-- Additional Stats -->
	<div class="grid grid-cols-1 md:grid-cols-4 gap-4">
		<MetricCard title="Winning Trades" value={winningTrades.length.toString()} trend="up" />
		<MetricCard title="Losing Trades" value={losingTrades.length.toString()} trend="down" />
		<MetricCard title="Avg Win" value={formatCurrency(avgWin)} trend="up" />
		<MetricCard title="Avg Loss" value={formatCurrency(avgLoss)} trend="down" />
	</div>

	<!-- Trades Table -->
	<Card title="Trade History">
		{#if loading}
			<div class="text-center py-12 text-gray-600">
				Loading trades...
			</div>
		{:else if error}
			<div class="text-center py-12 text-red-600">
				{error}
			</div>
		{:else if !tradesData?.database_enabled}
			<div class="text-center py-12 text-gray-600">
				Database not enabled. Enable state persistence in config to track trades.
			</div>
		{:else if filteredTrades.length > 0}
			<DataTable data={filteredTrades} columns={columns} />
		{:else}
			<div class="text-center py-12 text-gray-600">
				No trades match the selected filters.
			</div>
		{/if}
	</Card>
</div>
