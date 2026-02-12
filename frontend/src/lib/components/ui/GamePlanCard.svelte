<script lang="ts">
	import type { GamePlanResponse } from '$lib/types/api';

	export let plan: GamePlanResponse | null = null;

	// Risk stance colors
	const riskColors: Record<string, string> = {
		AGGRESSIVE: 'bg-red-100 text-red-800 border-red-300',
		MODERATE: 'bg-yellow-100 text-yellow-800 border-yellow-300',
		DEFENSIVE: 'bg-green-100 text-green-800 border-green-300'
	};

	$: riskClass = plan ? riskColors[plan.risk_stance] || riskColors.MODERATE : '';
</script>

<div class="bg-white rounded-lg shadow p-4 border border-gray-300">
	<h3 class="text-lg font-semibold mb-3 text-black">Game Plan</h3>

	{#if plan}
		<div class="space-y-3">
			<!-- Risk Stance & Confidence -->
			<div class="flex items-center gap-3">
				<span class="px-3 py-1 rounded-full border text-sm font-medium {riskClass}">
					{plan.risk_stance}
				</span>
				<span class="text-sm text-gray-600">
					Confidence: {(plan.confidence * 100).toFixed(1)}%
				</span>
			</div>

			<!-- Priority Symbols -->
			{#if plan.priority_symbols.length > 0}
				<div>
					<span class="text-xs font-medium text-gray-600 uppercase">
						Priority Symbols
					</span>
					<div class="flex flex-wrap gap-2 mt-1">
						{#each plan.priority_symbols as symbol}
							<span
								class="px-2 py-1 bg-blue-100 text-blue-800 text-xs font-mono rounded"
							>
								{symbol}
							</span>
						{/each}
					</div>
				</div>
			{/if}

			<!-- Sector Focus -->
			{#if plan.sector_focus.length > 0}
				<div>
					<span class="text-xs font-medium text-gray-600 uppercase">
						Sector Focus
					</span>
					<div class="flex flex-wrap gap-2 mt-1">
						{#each plan.sector_focus as sector}
							<span class="px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded">
								{sector}
							</span>
						{/each}
					</div>
				</div>
			{/if}

			<!-- Reasoning -->
			<div>
				<span class="text-xs font-medium text-gray-600 uppercase">Reasoning</span>
				<p class="text-sm text-gray-700 mt-1">{plan.reasoning}</p>
			</div>

			<!-- Metadata -->
			<div class="text-xs text-gray-600 pt-2 border-t border-gray-300">
				Generated: {new Date(plan.generated_at).toLocaleString()}
			</div>
		</div>
	{:else}
		<p class="text-sm text-gray-600">
			No game plan available. Enable in daemon config.
		</p>
	{/if}
</div>
