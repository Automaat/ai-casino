<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import Card from '$lib/components/ui/Card.svelte';
	import Badge from '$lib/components/ui/Badge.svelte';
	import DataTable from '$lib/components/ui/DataTable.svelte';
	import { analyses } from '$lib/stores/dashboard';
	import { formatPercent, formatDate } from '$lib/utils/format';
	import type { AnalysisRecordResponse } from '$lib/types/api';

	let selectedSymbol = '';
	let selectedSignal = '';

	$: allAnalyses = $analyses || [];

	// Filter analyses
	$: filteredAnalyses = allAnalyses.filter(a => {
		if (selectedSymbol && a.symbol !== selectedSymbol) return false;
		if (selectedSignal && a.signal !== selectedSignal) return false;
		return true;
	});

	// Unique symbols for filter
	$: symbols = [...new Set(allAnalyses.map(a => a.symbol))].sort();

	async function loadData() {
		await analyses.fetch({ limit: 100 });
	}

	onMount(() => {
		loadData();
	});

	// Refetch when route changes (handles tab switching)
	$: if ($page.url.pathname === '/signals') {
		loadData();
	}

	const columns = [
		{ 
			key: 'timestamp' as keyof AnalysisRecordResponse, 
			label: 'Date',
			format: (value: string) => formatDate(value)
		},
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
			key: 'risk_level' as keyof AnalysisRecordResponse, 
			label: 'Risk',
			format: (value: string) => value
		},
		{ 
			key: 'trading_session' as keyof AnalysisRecordResponse, 
			label: 'Session',
			format: (value: string) => value === 'PRE_MARKET' ? '🌅 Pre' : '📊 Regular'
		}
	];
</script>

<svelte:head>
	<title>Signals - AI Casino</title>
</svelte:head>

<div class="space-y-8">
	<!-- Filters -->
	<Card title="Filters">
		<div class="flex gap-4">
			<div class="flex-1">
				<label for="symbol-filter" class="block text-sm font-medium text-slate-400 mb-2">
					Symbol
				</label>
				<select
					id="symbol-filter"
					bind:value={selectedSymbol}
					class="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
				>
					<option value="">All Symbols</option>
					{#each symbols as symbol}
						<option value={symbol}>{symbol}</option>
					{/each}
				</select>
			</div>
			<div class="flex-1">
				<label for="signal-filter" class="block text-sm font-medium text-slate-400 mb-2">
					Signal
				</label>
				<select
					id="signal-filter"
					bind:value={selectedSignal}
					class="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
				>
					<option value="">All Signals</option>
					<option value="BUY">BUY</option>
					<option value="SELL">SELL</option>
					<option value="HOLD">HOLD</option>
				</select>
			</div>
		</div>
	</Card>

	<!-- Stats -->
	<div class="grid grid-cols-1 md:grid-cols-4 gap-4">
		{#each (['BUY', 'SELL', 'HOLD'] as const) as signal}
			{@const count = filteredAnalyses.filter(a => a.signal === signal).length}
			<div class="bg-slate-800 rounded-lg border border-slate-700 p-4">
				<div class="flex items-center justify-between">
					<Badge variant={signal}>{signal}</Badge>
					<span class="text-2xl font-bold text-slate-100">{count}</span>
				</div>
			</div>
		{/each}
		<div class="bg-slate-800 rounded-lg border border-slate-700 p-4">
			<div class="flex items-center justify-between">
				<span class="text-sm font-medium text-slate-400">Total</span>
				<span class="text-2xl font-bold text-slate-100">{filteredAnalyses.length}</span>
			</div>
		</div>
	</div>

	<!-- Signals Table -->
	<Card title="Trading Signals">
		{#if filteredAnalyses.length > 0}
			<DataTable data={filteredAnalyses} columns={columns} />
		{:else}
			<div class="text-center py-12 text-slate-400">
				No signals match the selected filters.
			</div>
		{/if}
	</Card>
</div>
