<script lang="ts">
	import type { WatchlistResponse } from '$lib/types/api';

	export let watchlist: WatchlistResponse | null = null;

	let expanded = false;

	function toggleExpanded() {
		expanded = !expanded;
	}

	$: hasSymbols = watchlist && watchlist.count > 0;
</script>

<div class="bg-white dark:bg-gray-800 rounded-lg shadow p-4 border border-gray-200 dark:border-gray-700">
	<h3 class="text-lg font-semibold mb-3 text-gray-900 dark:text-white">Watchlist</h3>

	{#if hasSymbols && watchlist}
		<div class="space-y-3">
			<!-- Total Count -->
			<div class="text-2xl font-bold text-gray-900 dark:text-white">
				{watchlist.count} {watchlist.count === 1 ? 'Symbol' : 'Symbols'}
			</div>

			<!-- Source Breakdown -->
			<div class="grid grid-cols-3 gap-2">
				<div class="bg-blue-50 dark:bg-blue-900/20 rounded p-2">
					<div class="text-xs text-gray-600 dark:text-gray-400">Config</div>
					<div class="text-lg font-semibold text-blue-600 dark:text-blue-400">
						{watchlist.sources.config || 0}
					</div>
				</div>
				<div class="bg-green-50 dark:bg-green-900/20 rounded p-2">
					<div class="text-xs text-gray-600 dark:text-gray-400">Broker</div>
					<div class="text-lg font-semibold text-green-600 dark:text-green-400">
						{watchlist.sources.broker || 0}
					</div>
				</div>
				<div class="bg-purple-50 dark:bg-purple-900/20 rounded p-2">
					<div class="text-xs text-gray-600 dark:text-gray-400">Screening</div>
					<div class="text-lg font-semibold text-purple-600 dark:text-purple-400">
						{watchlist.sources.screening || 0}
					</div>
				</div>
			</div>

			<!-- Expandable Symbol List -->
			{#if watchlist.symbols.length > 0}
				<button
					on:click={toggleExpanded}
					class="w-full text-left text-sm text-blue-600 dark:text-blue-400 hover:underline"
				>
					{expanded ? '▼ Hide symbols' : '▶ Show symbols'}
				</button>

				{#if expanded}
					<div class="flex flex-wrap gap-2 max-h-40 overflow-y-auto p-2 bg-gray-50 dark:bg-gray-900/50 rounded">
						{#each watchlist.symbols as symbol}
							<span class="px-2 py-1 bg-white dark:bg-gray-800 text-xs font-mono rounded border border-gray-200 dark:border-gray-700">
								{symbol}
							</span>
						{/each}
					</div>
				{/if}
			{/if}
		</div>
	{:else}
		<p class="text-sm text-gray-500 dark:text-gray-400">No watchlist data available</p>
	{/if}
</div>
