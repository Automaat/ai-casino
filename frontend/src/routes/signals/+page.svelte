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
	let startDate = '';
	let endDate = '';

	$: allAnalyses = $analyses || [];

	// Date range presets
	function setDatePreset(preset: 'today' | '7days' | '30days' | 'all') {
		const now = new Date();
		const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());

		switch (preset) {
			case 'today':
				startDate = today.toISOString().split('T')[0];
				endDate = new Date(today.getTime() + 24 * 60 * 60 * 1000).toISOString().split('T')[0];
				break;
			case '7days':
				startDate = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
				endDate = new Date(today.getTime() + 24 * 60 * 60 * 1000).toISOString().split('T')[0];
				break;
			case '30days':
				startDate = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
				endDate = new Date(today.getTime() + 24 * 60 * 60 * 1000).toISOString().split('T')[0];
				break;
			case 'all':
				startDate = '';
				endDate = '';
				break;
		}
	}

	function clearFilters() {
		selectedSymbol = '';
		selectedSignal = '';
		startDate = '';
		endDate = '';
	}

	// Filter analyses
	$: filteredAnalyses = allAnalyses.filter(a => {
		if (selectedSymbol && a.symbol !== selectedSymbol) return false;
		if (selectedSignal && a.signal !== selectedSignal) return false;

		// Date range filter
		if (startDate || endDate) {
			const timestamp = new Date(a.timestamp);
			if (startDate) {
				const start = new Date(startDate);
				if (timestamp < start) return false;
			}
			if (endDate) {
				const end = new Date(endDate);
				if (timestamp > end) return false;
			}
		}

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
		<div class="space-y-4">
			<!-- Quick Presets -->
			<div>
				<label class="block text-sm font-medium text-gray-600 mb-2">
					Quick Presets
				</label>
				<div class="flex gap-2">
					<button
						on:click={() => setDatePreset('today')}
						class="px-4 py-2 bg-gray-100 hover:bg-gray-50 border border-gray-300 rounded-lg text-black text-sm transition-colors"
					>
						Today
					</button>
					<button
						on:click={() => setDatePreset('7days')}
						class="px-4 py-2 bg-gray-100 hover:bg-gray-50 border border-gray-300 rounded-lg text-black text-sm transition-colors"
					>
						Last 7 days
					</button>
					<button
						on:click={() => setDatePreset('30days')}
						class="px-4 py-2 bg-gray-100 hover:bg-gray-50 border border-gray-300 rounded-lg text-black text-sm transition-colors"
					>
						Last 30 days
					</button>
					<button
						on:click={() => setDatePreset('all')}
						class="px-4 py-2 bg-gray-100 hover:bg-gray-50 border border-gray-300 rounded-lg text-black text-sm transition-colors"
					>
						All time
					</button>
				</div>
			</div>

			<!-- Date Range -->
			<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
				<div>
					<label for="start-date" class="block text-sm font-medium text-gray-600 mb-2">
						Start Date
					</label>
					<input
						id="start-date"
						type="date"
						bind:value={startDate}
						class="w-full px-3 py-2 bg-gray-100 border border-gray-300 rounded-lg text-black focus:outline-none focus:ring-2 focus:ring-blue-700"
					/>
				</div>
				<div>
					<label for="end-date" class="block text-sm font-medium text-gray-600 mb-2">
						End Date
					</label>
					<input
						id="end-date"
						type="date"
						bind:value={endDate}
						class="w-full px-3 py-2 bg-gray-100 border border-gray-300 rounded-lg text-black focus:outline-none focus:ring-2 focus:ring-blue-700"
					/>
				</div>
			</div>

			<!-- Symbol and Signal Filters -->
			<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
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
					<label for="signal-filter" class="block text-sm font-medium text-gray-600 mb-2">
						Signal
					</label>
					<select
						id="signal-filter"
						bind:value={selectedSignal}
						class="w-full px-3 py-2 bg-gray-100 border border-gray-300 rounded-lg text-black focus:outline-none focus:ring-2 focus:ring-blue-700"
					>
						<option value="">All Signals</option>
						<option value="BUY">BUY</option>
						<option value="SELL">SELL</option>
						<option value="HOLD">HOLD</option>
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

	<!-- Stats -->
	<div class="grid grid-cols-1 md:grid-cols-4 gap-4">
		{#each (['BUY', 'SELL', 'HOLD'] as const) as signal}
			{@const count = filteredAnalyses.filter(a => a.signal === signal).length}
			<div class="bg-white rounded-lg border border-gray-300 p-4">
				<div class="flex items-center justify-between">
					<Badge variant={signal}>{signal}</Badge>
					<span class="text-2xl font-bold text-black">{count}</span>
				</div>
			</div>
		{/each}
		<div class="bg-white rounded-lg border border-gray-300 p-4">
			<div class="flex items-center justify-between">
				<span class="text-sm font-medium text-gray-600">Total</span>
				<span class="text-2xl font-bold text-black">{filteredAnalyses.length}</span>
			</div>
		</div>
	</div>

	<!-- Signals Table -->
	<Card title="Trading Signals">
		{#if filteredAnalyses.length > 0}
			<DataTable data={filteredAnalyses} columns={columns} />
		{:else}
			<div class="text-center py-12 text-gray-600">
				No signals match the selected filters.
			</div>
		{/if}
	</Card>
</div>
