<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { events } from '$lib/stores/dashboard';
	import Badge from '$lib/components/ui/Badge.svelte';
	import Card from '$lib/components/ui/Card.svelte';
	import type { SystemEvent } from '$lib/types/api';

	let intervalId: ReturnType<typeof setInterval> | null = null;

	const eventTypeBadgeMap: Record<string, 'success' | 'error' | 'warning' | 'info' | 'neutral'> = {
		CYCLE_START: 'info',
		ANALYSIS_COMPLETE: 'success',
		ERROR: 'error',
		TRADE_EXECUTED: 'success',
		DEGRADATION_CHANGE: 'warning',
		SERVICE_UNAVAILABLE: 'error',
		SERVICE_RESTORED: 'success'
	};

	function getBadgeVariant(eventType: string): 'success' | 'error' | 'warning' | 'info' | 'neutral' {
		return eventTypeBadgeMap[eventType] || 'neutral';
	}

	function formatRelativeTime(timestamp: string): string {
		const now = new Date();
		const eventTime = new Date(timestamp);
		const diffMs = now.getTime() - eventTime.getTime();
		const diffSec = Math.floor(diffMs / 1000);
		const diffMin = Math.floor(diffSec / 60);
		const diffHour = Math.floor(diffMin / 60);
		const diffDay = Math.floor(diffHour / 24);

		if (diffSec < 60) return `${diffSec}s ago`;
		if (diffMin < 60) return `${diffMin}m ago`;
		if (diffHour < 24) return `${diffHour}h ago`;
		return `${diffDay}d ago`;
	}

	async function fetchEvents() {
		await events.fetch({ limit: 15 });
	}

	onMount(() => {
		fetchEvents();
		intervalId = setInterval(fetchEvents, 5000);
	});

	onDestroy(() => {
		if (intervalId) clearInterval(intervalId);
	});

	const latestEvent = $derived($events?.events?.[0]);
	const eventsList = $derived($events?.events || []);
</script>

<div class="space-y-6">
	<!-- Status Banner -->
	{#if latestEvent}
		<Card>
			<div class="flex items-center gap-4">
				<Badge variant={getBadgeVariant(latestEvent.event_type)}>
					{latestEvent.event_type}
				</Badge>
				<div class="flex-1">
					<p class="text-slate-300">
						{latestEvent.data.message || JSON.stringify(latestEvent.data)}
					</p>
					<p class="text-sm text-slate-500 mt-1">
						{formatRelativeTime(latestEvent.timestamp)}
					</p>
				</div>
			</div>
		</Card>
	{/if}

	<!-- Activity Feed -->
	<Card title="Recent Activity">
		{#if eventsList.length > 0}
			<div class="space-y-3">
				{#each eventsList as event}
					<div class="flex items-start gap-4 py-2 border-b border-slate-700 last:border-b-0">
						<div class="flex-shrink-0">
							<Badge variant={getBadgeVariant(event.event_type)}>
								{event.event_type}
							</Badge>
						</div>
						<div class="flex-1 min-w-0">
							<p class="text-sm text-slate-300 truncate">
								{event.data.message || event.data.symbol || JSON.stringify(event.data)}
							</p>
							<p class="text-xs text-slate-500 mt-1">
								{formatRelativeTime(event.timestamp)}
							</p>
						</div>
					</div>
				{/each}
			</div>
		{:else}
			<p class="text-slate-400 text-center py-8">No recent events</p>
		{/if}
	</Card>
</div>
