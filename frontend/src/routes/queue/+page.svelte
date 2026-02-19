<script lang="ts">
	import { onMount } from 'svelte';
	import MetricCard from '$lib/components/ui/MetricCard.svelte';
	import Card from '$lib/components/ui/Card.svelte';
	import DataTable from '$lib/components/ui/DataTable.svelte';
	import BarChart from '$lib/components/charts/BarChart.svelte';
	import { queueStats, queueEvents } from '$lib/stores/dashboard';
	import type { QueueEventItem } from '$lib/types/api';

	type StatusFilter = 'all' | 'pending' | 'consumed' | 'expired';

	let selectedStatus = $state<StatusFilter>('all');

	const stats = $derived($queueStats);
	const eventsData = $derived($queueEvents);

	// Bar chart data: pending events by type
	const byTypeChartData = $derived.by(() => {
		if (!stats?.by_type) return [];
		const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4'];
		return stats.by_type.map((item, idx) => ({
			label: item.event_type,
			value: item.count,
			color: colors[idx % colors.length]
		}));
	});

	const statusTabs: { id: StatusFilter; label: string }[] = [
		{ id: 'all', label: 'All' },
		{ id: 'pending', label: 'Pending' },
		{ id: 'consumed', label: 'Consumed' },
		{ id: 'expired', label: 'Expired' }
	];

	const columns = [
		{
			key: 'enqueued_at' as keyof QueueEventItem,
			label: 'Time',
			format: (v: string) => new Date(v).toLocaleString()
		},
		{ key: 'event_type' as keyof QueueEventItem, label: 'Type' },
		{
			key: 'symbols' as keyof QueueEventItem,
			label: 'Symbols',
			format: (v: string[]) => v.join(', ') || '–'
		},
		{ key: 'urgency' as keyof QueueEventItem, label: 'Urgency' },
		{ key: 'sentiment' as keyof QueueEventItem, label: 'Sentiment' },
		{
			key: 'confidence' as keyof QueueEventItem,
			label: 'Confidence',
			format: (v: number) => `${(v * 100).toFixed(0)}%`
		},
		{
			key: 'status' as keyof QueueEventItem,
			label: 'Status',
			cellClass: (_v: string, row: QueueEventItem) =>
				row.status === 'pending'
					? 'text-blue-600 font-medium'
					: row.status === 'consumed'
						? 'text-green-600 font-medium'
						: 'text-red-500 font-medium'
		},
		{
			key: 'ttl_remaining_seconds' as keyof QueueEventItem,
			label: 'Expires In',
			format: (v: number | null) => {
				if (v === null) return '–';
				const mins = Math.floor(v / 60);
				const secs = Math.floor(v % 60);
				return `${mins}m ${secs}s`;
			}
		}
	];

	async function loadEvents(status: StatusFilter) {
		selectedStatus = status;
		await queueEvents.fetch(status, 100);
	}

	onMount(() => {
		queueEvents.fetch('all', 100);
	});
</script>

<svelte:head>
	<title>Queue - AI Casino</title>
</svelte:head>

<div class="space-y-8">
	<!-- Header -->
	<div>
		<h1 class="text-3xl font-bold text-gray-900">Market Event Queue</h1>
		<p class="text-gray-600 mt-1">Live state of the PostgreSQL-backed market signal queue</p>
	</div>

	<!-- Metric Cards -->
	<div class="grid grid-cols-1 md:grid-cols-4 gap-6">
		<MetricCard
			title="Pending"
			value={stats?.pending_count?.toString() ?? '–'}
			subtitle="Awaiting processing"
			icon="⏳"
		/>
		<MetricCard
			title="Consumed (24h)"
			value={stats?.consumed_count_24h?.toString() ?? '–'}
			subtitle="Processed last 24h"
			icon="✅"
		/>
		<MetricCard
			title="Stale"
			value={stats?.stale_count?.toString() ?? '–'}
			subtitle="Expired, not yet purged"
			icon="⚠️"
		/>
		<MetricCard
			title="Total in DB"
			value={stats?.total_in_db?.toString() ?? '–'}
			subtitle="All rows in table"
			icon="🗄️"
		/>
	</div>

	<!-- Pending by Event Type -->
	<Card title="Pending Events by Type">
		{#if byTypeChartData.length > 0}
			<BarChart data={byTypeChartData} height={300} yAxisLabel="Count" xAxisLabel="Event Type" />
		{:else}
			<div class="text-center py-12 text-gray-600">No pending events</div>
		{/if}
	</Card>

	<!-- Events Table -->
	<Card title="Queue Events">
		<div class="space-y-4">
			<!-- Status filter tabs -->
			<div class="flex gap-2 px-4 pt-2">
				{#each statusTabs as tab}
					<button
						class="px-4 py-1.5 rounded-full text-sm font-medium border transition-colors {selectedStatus === tab.id
							? 'bg-black text-white border-black'
							: 'bg-white text-gray-700 border-gray-300 hover:border-gray-500'}"
						onclick={() => loadEvents(tab.id)}
					>
						{tab.label}
					</button>
				{/each}
				<span class="ml-auto text-sm text-gray-500 self-center">
					{eventsData?.returned_count ?? 0} events
				</span>
			</div>

			{#if eventsData?.events && eventsData.events.length > 0}
				<DataTable data={eventsData.events} {columns} />
			{:else}
				<div class="text-center py-12 text-gray-600">No events found</div>
			{/if}
		</div>
	</Card>
</div>
