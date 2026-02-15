<script lang="ts">
	import Badge from '$lib/components/ui/Badge.svelte';
	import { formatCurrency, formatPercent } from '$lib/utils/format';
	import type { RebalanceAllocation, PositionResponse } from '$lib/types/api';

	interface Props {
		allocations: RebalanceAllocation[];
		portfolioValue: number;
		positions: PositionResponse[];
	}

	let { allocations, portfolioValue, positions }: Props = $props();

	interface EnrichedAction {
		symbol: string;
		action: string;
		current_pct: number;
		target_pct: number;
		delta_pct: number;
		shares: number;
		amount: number;
		price: number;
		price_timestamp: string | null;
		is_stale: boolean;
	}

	// Build enriched actions with price data
	const enrichedActions = $derived.by(() => {
		const now = new Date();
		const staleThresholdMs = 60 * 60 * 1000; // 1 hour

		return allocations
			.map((alloc): EnrichedAction => {
				const position = positions.find(p => p.symbol === alloc.symbol);
				const currentPrice = position?.current_price || 0;
				const priceTimestamp = position?.entry_timestamp || null;
				const deltaAbs = Math.abs(alloc.delta);
				const shares = currentPrice > 0 ? Math.floor((deltaAbs * portfolioValue) / currentPrice) : 0;
				const amount = shares * currentPrice;

				let isStale = false;
				if (priceTimestamp) {
					const priceDate = new Date(priceTimestamp);
					isStale = now.getTime() - priceDate.getTime() > staleThresholdMs;
				}

				return {
					symbol: alloc.symbol,
					action: alloc.action,
					current_pct: alloc.current_weight * 100,
					target_pct: alloc.target_weight * 100,
					delta_pct: alloc.delta * 100,
					shares,
					amount,
					price: currentPrice,
					price_timestamp: priceTimestamp,
					is_stale: isStale
				};
			})
			.sort((a, b) => Math.abs(b.delta_pct) - Math.abs(a.delta_pct));
	});

	const hasStaleData = $derived(enrichedActions.some(a => a.is_stale));

	function getActionVariant(action: string): 'success' | 'error' | 'neutral' {
		switch (action) {
			case 'INCREASE':
				return 'success';
			case 'REDUCE':
				return 'error';
			default:
				return 'neutral';
		}
	}

	function getRowClass(action: string): string {
		switch (action) {
			case 'INCREASE':
				return 'bg-green-50 hover:bg-green-100';
			case 'REDUCE':
				return 'bg-red-50 hover:bg-red-100';
			default:
				return 'hover:bg-gray-50';
		}
	}
</script>

{#if hasStaleData}
	<div class="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded">
		<div class="flex items-center gap-2 text-sm text-yellow-800">
			<span class="font-medium">⚠️ Warning:</span>
			<span>Some position prices are stale (&gt;1hr old). Calculations shown but may be approximate.</span>
		</div>
	</div>
{/if}

<div class="overflow-x-auto">
	<table class="min-w-full divide-y divide-gray-300">
		<thead class="bg-gray-50">
			<tr>
				<th class="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
					Symbol
				</th>
				<th class="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
					Action
				</th>
				<th class="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
					Current %
				</th>
				<th class="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
					Target %
				</th>
				<th class="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
					Delta %
				</th>
				<th class="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
					Shares
				</th>
				<th class="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
					$ Amount
				</th>
			</tr>
		</thead>
		<tbody class="bg-white divide-y divide-gray-300">
			{#each enrichedActions as action}
				<tr class="{getRowClass(action.action)} transition-colors">
					<td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-black">
						{action.symbol}
						{#if action.is_stale}
							<span class="ml-1 text-yellow-600" title="Price data >1hr old">⚠️</span>
						{/if}
					</td>
					<td class="px-6 py-4 whitespace-nowrap text-sm">
						<Badge variant={getActionVariant(action.action)}>{action.action}</Badge>
					</td>
					<td class="px-6 py-4 whitespace-nowrap text-sm text-black">
						{formatPercent(action.current_pct / 100)}
					</td>
					<td class="px-6 py-4 whitespace-nowrap text-sm text-black">
						{formatPercent(action.target_pct / 100)}
					</td>
					<td class="px-6 py-4 whitespace-nowrap text-sm font-medium {action.delta_pct > 0 ? 'text-red-600' : action.delta_pct < 0 ? 'text-green-600' : 'text-gray-600'}">
						{action.delta_pct > 0 ? '+' : ''}{formatPercent(action.delta_pct / 100)}
					</td>
					<td class="px-6 py-4 whitespace-nowrap text-sm text-black">
						{action.shares.toLocaleString()}
					</td>
					<td class="px-6 py-4 whitespace-nowrap text-sm text-black">
						{formatCurrency(action.amount)}
					</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>
