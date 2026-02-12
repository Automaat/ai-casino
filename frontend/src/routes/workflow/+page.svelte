<script lang="ts">
	import LiveStatusTab from '$lib/components/workflow/LiveStatusTab.svelte';
	import HistoricalAnalysisTab from '$lib/components/workflow/HistoricalAnalysisTab.svelte';

	let activeSubTab = $state<'live' | 'historical'>('live');
</script>

<div class="space-y-6">
	<h2 class="text-2xl font-bold text-slate-100">Workflow Execution</h2>

	<!-- Sub-tab buttons -->
	<div class="flex gap-4 border-b border-slate-700">
		<button
			class="px-4 py-2 text-sm font-medium transition-colors {activeSubTab === 'live'
				? 'text-blue-400 border-b-2 border-blue-400'
				: 'text-slate-400 hover:text-slate-300'}"
			onclick={() => activeSubTab = 'live'}
		>
			Live Status
		</button>
		<button
			class="px-4 py-2 text-sm font-medium transition-colors {activeSubTab === 'historical'
				? 'text-blue-400 border-b-2 border-blue-400'
				: 'text-slate-400 hover:text-slate-300'}"
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
