<script lang="ts">
	import type { WatchlistResponse } from '$lib/types/api';

	export let watchlist: WatchlistResponse | null = null;

	let expanded = false;

	function toggleExpanded() {
		expanded = !expanded;
	}

	$: hasSymbols = watchlist && watchlist.count > 0;
</script>

<div class="bg-white rounded-lg shadow p-4 border border-gray-300">
	<h3 class="text-lg font-semibold mb-3 text-black">Watchlist</h3>

	{#if hasSymbols && watchlist}
		<div class="space-y-3">
			<!-- Total Count -->
			<div class="text-2xl font-bold text-black">
				{watchlist.count} {watchlist.count === 1 ? 'Symbol' : 'Symbols'}
			</div>

			<!-- Source Breakdown -->
			<div class="grid grid-cols-3 gap-2">
				<div class="bg-blue-50 rounded p-2">
					<div class="text-xs text-gray-600">Config</div>
					<div class="text-lg font-semibold text-blue-700">
						{watchlist.sources.config || 0}
					</div>
				</div>
				<div class="bg-green-50 rounded p-2">
					<div class="text-xs text-gray-600">Broker</div>
					<div class="text-lg font-semibold text-green-700">
						{watchlist.sources.broker || 0}
					</div>
				</div>
				<div class="bg-purple-50 rounded p-2">
					<div class="text-xs text-gray-600">Discovery</div>
					<div class="text-lg font-semibold text-purple-700">
						{watchlist.sources.discovery || 0}
					</div>
				</div>
			</div>

			<!-- Expandable Symbol List -->
			{#if watchlist.symbols.length > 0}
				<button
					on:click={toggleExpanded}
					class="w-full text-left text-sm text-blue-700 hover:underline"
					aria-expanded={expanded}
					aria-controls="watchlist-symbols"
				>
					{expanded ? '▼ Hide symbols' : '▶ Show symbols'}
				</button>

				{#if expanded}
					<div
						id="watchlist-symbols"
						class="flex flex-wrap gap-2 max-h-40 overflow-y-auto p-2 bg-gray-50 rounded"
					>
						{#each watchlist.symbols as symbol}
							<span class="px-2 py-1 bg-white text-xs font-mono rounded border border-gray-300">
								{symbol}
							</span>
						{/each}
					</div>
				{/if}
			{/if}
		</div>
	{:else}
		<p class="text-sm text-gray-600">No watchlist data available</p>
	{/if}
</div>
