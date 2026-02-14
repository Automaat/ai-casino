<script lang="ts">
	import type { PositionManagementActionResponse } from '$lib/types/api';
	import { formatDistanceToNow } from 'date-fns';

	export let actions: PositionManagementActionResponse[];

	// Map action types to icons and colors
	const actionConfig: Record<string, { icon: string; color: string; label: string }> = {
		TRAILING_STOP: { icon: '📈', color: 'text-blue-600', label: 'Trailing Stop' },
		BREAKEVEN: { icon: '⚖️', color: 'text-green-600', label: 'Breakeven' },
		PARTIAL_PROFIT: { icon: '💰', color: 'text-yellow-600', label: 'Partial Profit' },
		TIME_EXIT: { icon: '⏰', color: 'text-red-600', label: 'Time Exit' },
		CONVICTION_SCALE: { icon: '📉', color: 'text-purple-600', label: 'Conviction Scale' }
	};

	function getActionConfig(actionType: string) {
		return actionConfig[actionType] || { icon: '🔹', color: 'text-gray-600', label: actionType };
	}

	function formatTimestamp(timestamp: string): string {
		const date = new Date(timestamp);
		return formatDistanceToNow(date, { addSuffix: true });
	}

	function formatPrice(price: number): string {
		return `$${price.toFixed(2)}`;
	}
</script>

{#if actions.length === 0}
	<div class="text-center py-8 text-gray-500">
		<p class="text-lg">No management actions yet</p>
		<p class="text-sm mt-2">Actions like trailing stops and partial exits will appear here</p>
	</div>
{:else}
	<div class="space-y-4">
		{#each actions as action}
			{@const config = getActionConfig(action.action_type)}
			<div class="flex gap-4">
				<!-- Timeline dot/line -->
				<div class="flex flex-col items-center">
					<div class="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center text-xl">
						{config.icon}
					</div>
					{#if action !== actions[actions.length - 1]}
						<div class="w-0.5 h-full bg-gray-200 my-1"></div>
					{/if}
				</div>

				<!-- Action card -->
				<div class="flex-1 pb-4">
					<div class="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
						<div class="flex items-start justify-between mb-2">
							<h3 class="font-semibold {config.color}">{config.label}</h3>
							<span class="text-sm text-gray-500">{formatTimestamp(action.timestamp)}</span>
						</div>

						<p class="text-sm text-gray-700 mb-3">{action.reason}</p>

						<div class="grid grid-cols-2 gap-2 text-sm">
							{#if action.old_stop_loss !== null && action.new_stop_loss !== null}
								<div>
									<span class="text-gray-500">Stop Loss:</span>
									<span class="ml-1 font-mono">
										{formatPrice(action.old_stop_loss)} → {formatPrice(action.new_stop_loss)}
									</span>
								</div>
							{/if}

							{#if action.qty_sold !== null}
								<div>
									<span class="text-gray-500">Qty Sold:</span>
									<span class="ml-1 font-mono">{action.qty_sold.toFixed(2)}</span>
								</div>
							{/if}

							<div>
								<span class="text-gray-500">Price:</span>
								<span class="ml-1 font-mono">{formatPrice(action.price)}</span>
							</div>

							<div>
								<span class="text-gray-500">Status:</span>
								<span
									class="ml-1 px-2 py-0.5 rounded-full text-xs {action.executed
										? 'bg-green-100 text-green-800'
										: 'bg-yellow-100 text-yellow-800'}"
								>
									{action.executed ? 'Executed' : 'Pending'}
								</span>
							</div>

							{#if action.order_id}
								<div class="col-span-2">
									<span class="text-gray-500">Order ID:</span>
									<span class="ml-1 font-mono text-xs">{action.order_id}</span>
								</div>
							{/if}
						</div>
					</div>
				</div>
			</div>
		{/each}
	</div>
{/if}
