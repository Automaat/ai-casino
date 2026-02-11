<script lang="ts">
	import type { DegradationResponse } from '$lib/types/api';

	export let degradation: DegradationResponse | null = null;

	// Tier colors
	const tierColors: Record<string, string> = {
		FULL: 'bg-green-100 text-green-800 border-green-300',
		DEGRADED: 'bg-yellow-100 text-yellow-800 border-yellow-300',
		MINIMAL: 'bg-orange-100 text-orange-800 border-orange-300',
		HALTED: 'bg-red-100 text-red-800 border-red-300'
	};

	$: shouldShow = degradation && degradation.tier !== 'FULL';
	$: tierClass = degradation ? tierColors[degradation.tier] || tierColors.DEGRADED : '';
</script>

{#if shouldShow && degradation}
	<div
		class="bg-white dark:bg-gray-800 rounded-lg shadow p-4 border-l-4 border-yellow-400 dark:border-yellow-600"
		role="alert"
	>
		<div class="flex items-start gap-3">
			<span class="text-2xl">⚠️</span>
			<div class="flex-1">
				<div class="flex items-center gap-3 mb-2">
					<h3 class="text-lg font-semibold text-gray-900 dark:text-white">Service Degradation</h3>
					<span class="px-3 py-1 rounded-full border text-sm font-medium {tierClass}">
						{degradation.tier}
					</span>
				</div>

				<!-- Confidence Adjustment -->
				<div class="mb-2">
					<span class="text-sm font-medium text-gray-700 dark:text-gray-300">
						Confidence Adjustment: {(degradation.confidence_adjustment * 100).toFixed(0)}%
					</span>
				</div>

				<!-- Unavailable Services -->
				{#if degradation.unavailable_services.length > 0}
					<div class="mb-2">
						<span class="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
							Unavailable Services
						</span>
						<div class="flex flex-wrap gap-2 mt-1">
							{#each degradation.unavailable_services as service}
								<span class="px-2 py-1 bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 text-xs rounded">
									{service}
								</span>
							{/each}
						</div>
					</div>
				{/if}

				<!-- Halt Reason -->
				{#if degradation.halt_reason}
					<div class="mt-2 p-2 bg-red-50 dark:bg-red-900/20 rounded">
						<span class="text-xs font-medium text-red-700 dark:text-red-300 uppercase">
							Trading Halted
						</span>
						<p class="text-sm text-red-600 dark:text-red-400 mt-1">
							{degradation.halt_reason}
						</p>
					</div>
				{/if}
			</div>
		</div>
	</div>
{/if}
