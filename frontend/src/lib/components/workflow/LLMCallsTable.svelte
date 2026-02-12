<script lang="ts">
	import type { LLMCallMetric } from '$lib/types/api';

	interface Props {
		data: LLMCallMetric[];
	}

	let { data }: Props = $props();

	function formatTimestamp(timestamp: string): string {
		const date = new Date(timestamp);
		return date.toLocaleTimeString();
	}

	function formatCost(cost: number | null): string {
		if (cost === null) return 'N/A';
		return `$${cost.toFixed(4)}`;
	}

	function formatTokens(tokens: number | null): string {
		if (tokens === null) return 'N/A';
		return tokens.toString();
	}
</script>

<div class="overflow-x-auto">
	{#if data && data.length > 0}
		<table class="w-full text-sm text-left text-gray-700">
			<thead class="text-xs text-gray-600 uppercase bg-gray-100">
				<tr>
					<th class="px-4 py-3">Time</th>
					<th class="px-4 py-3">Agent</th>
					<th class="px-4 py-3">Method</th>
					<th class="px-4 py-3">Model</th>
					<th class="px-4 py-3">Latency (ms)</th>
					<th class="px-4 py-3">In Tokens</th>
					<th class="px-4 py-3">Out Tokens</th>
					<th class="px-4 py-3">Cost</th>
					<th class="px-4 py-3">Status</th>
				</tr>
			</thead>
			<tbody>
				{#each data as call}
					<tr class="border-b border-gray-300 hover:bg-gray-100/50">
						<td class="px-4 py-3 whitespace-nowrap">{formatTimestamp(call.timestamp)}</td>
						<td class="px-4 py-3">{call.agent_name}</td>
						<td class="px-4 py-3">{call.method}</td>
						<td class="px-4 py-3 text-xs">{call.model}</td>
						<td class="px-4 py-3 text-right">{call.latency_ms.toFixed(0)}</td>
						<td class="px-4 py-3 text-right">{formatTokens(call.input_tokens)}</td>
						<td class="px-4 py-3 text-right">{formatTokens(call.output_tokens)}</td>
						<td class="px-4 py-3 text-right">{formatCost(call.estimated_cost_usd)}</td>
						<td class="px-4 py-3 text-center">
							{#if call.success}
								<span class="text-green-600">✓</span>
							{:else}
								<span class="text-red-600" title={call.error || 'Unknown error'}>✗</span>
							{/if}
						</td>
					</tr>
				{/each}
			</tbody>
		</table>
	{:else}
		<p class="text-gray-600 text-center py-8">No LLM calls recorded</p>
	{/if}
</div>
