<script lang="ts">
	import LiveStatusTab from '$lib/components/workflow/LiveStatusTab.svelte';
	import HistoricalAnalysisTab from '$lib/components/workflow/HistoricalAnalysisTab.svelte';

	let activeSubTab = $state<'live' | 'historical'>('live');
</script>

<div class="space-y-6">
	<h2 class="text-2xl font-bold text-black">Workflow Execution</h2>

	<!-- Sub-tab buttons -->
	<div class="flex gap-4 border-b border-gray-300">
		<button
			class="px-4 py-2 text-sm font-medium transition-colors {activeSubTab === 'live'
				? 'text-blue-600 border-b-2 border-blue-600'
				: 'text-gray-600 hover:text-gray-700'}"
			onclick={() => activeSubTab = 'live'}
		>
			Live Status
		</button>
		<button
			class="px-4 py-2 text-sm font-medium transition-colors {activeSubTab === 'historical'
				? 'text-blue-600 border-b-2 border-blue-600'
				: 'text-gray-600 hover:text-gray-700'}"
			onclick={() => activeSubTab = 'historical'}
		>
			Historical Analysis
		</button>
	</div>

	<!-- Tab content -->
	{#if activeSubTab === 'live'}
		<LiveStatusTab />
	{:else}
		<HistoricalAnalysisTab />
	{/if}
</div>
