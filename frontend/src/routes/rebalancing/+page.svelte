<script lang="ts">
	import { onMount } from 'svelte';
	import Card from '$lib/components/ui/Card.svelte';
	import Badge from '$lib/components/ui/Badge.svelte';
	import RebalanceChart from '$lib/components/charts/RebalanceChart.svelte';
	import RebalancingActionsTable from '$lib/components/rebalancing/RebalancingActionsTable.svelte';
	import MetricsComparisonCard from '$lib/components/rebalancing/MetricsComparisonCard.svelte';
	import DeviationTimelineChart from '$lib/components/rebalancing/DeviationTimelineChart.svelte';
	import MetricsTrendChart from '$lib/components/rebalancing/MetricsTrendChart.svelte';
	import { api } from '$lib/api/client';
	import type { RebalancingHistoryResponse, PositionsResponse } from '$lib/types/api';

	let rebalancingData = $state<RebalancingHistoryResponse | null>(null);
	let positions = $state<PositionsResponse | null>(null);
	let loading = $state(true);

	async function loadData() {
		loading = true;
		try {
			const [rebalancingResult, positionsResult] = await Promise.allSettled([
				api.getRebalancingHistory(),
				api.getPositions()
			]);

			if (rebalancingResult.status === 'fulfilled') {
				rebalancingData = rebalancingResult.value;
			}

			if (positionsResult.status === 'fulfilled') {
				positions = positionsResult.value;
			}
		} finally {
			loading = false;
		}
	}

	onMount(() => {
		loadData();
	});

	const lastUpdatedText = $derived.by(() => {
		if (!rebalancingData?.latest) return 'N/A';
		const date = new Date(rebalancingData.latest.timestamp);
		return date.toLocaleString();
	});
</script>

<svelte:head>
	<title>Rebalancing - AI Casino</title>
</svelte:head>

<div class="space-y-8">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h1 class="text-3xl font-bold text-black">Rebalancing</h1>
			<p class="text-gray-600 mt-1">Portfolio rebalancing analysis and execution recommendations</p>
		</div>
		<div>
			{#if rebalancingData}
				<Badge variant={rebalancingData.enabled ? 'success' : 'neutral'}>
					{rebalancingData.enabled ? 'Enabled' : 'Disabled'}
				</Badge>
			{/if}
		</div>
	</div>

	{#if loading}
		<div class="text-center py-12 text-gray-600">Loading...</div>
	{:else if !rebalancingData}
		<Card>
			<div class="text-center py-12 text-gray-600">
				<div class="font-medium">Failed to load rebalancing data</div>
				<div class="text-sm mt-2">Please try refreshing the page</div>
			</div>
		</Card>
	{:else if !rebalancingData.enabled}
		<!-- Disabled State -->
		<Card>
			<div class="text-center py-12 text-gray-600">
				<div class="font-medium text-lg">Rebalancing Disabled</div>
				<div class="text-sm mt-2">
					Enable in daemon config: <code class="bg-gray-100 px-2 py-1 rounded">rebalancing.enabled: true</code>
				</div>
			</div>
		</Card>
	{:else if !rebalancingData.latest}
		<!-- No Data State -->
		<Card>
			<div class="text-center py-12 text-gray-600">
				<div class="font-medium text-lg">No Rebalancing Data Yet</div>
				<div class="text-sm mt-2">Check back after daemon runs rebalancing analysis</div>
			</div>
		</Card>
	{:else}
		<!-- Main Content -->
		<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
			<!-- Left Column (2/3) -->
			<div class="lg:col-span-2 space-y-6">
				<!-- Recommended Actions -->
				<Card title="Recommended Actions">
					<RebalancingActionsTable
						allocations={rebalancingData.latest.allocations}
						portfolioValue={rebalancingData.current_portfolio_value}
						positions={positions?.positions || []}
					/>
				</Card>

				<!-- Current vs Target Allocation -->
				<Card title="Current vs Target Allocation">
					<RebalanceChart allocations={rebalancingData.latest.allocations} height={350} />
				</Card>

				<!-- Allocation Drift Over Time -->
				{#if rebalancingData.history.length > 0}
					<Card title="Allocation Drift Over Time">
						<DeviationTimelineChart
							history={rebalancingData.history}
							threshold={rebalancingData.rebalance_threshold}
							height={350}
						/>
					</Card>

					<!-- Portfolio Metrics Trend -->
					<Card title="Portfolio Metrics Trend">
						<MetricsTrendChart history={rebalancingData.history} height={350} />
					</Card>
				{:else}
					<Card title="Allocation Drift Over Time">
						<div class="text-center py-12 text-gray-600">
							<div class="font-medium">Insufficient Historical Data</div>
							<div class="text-sm mt-2">History will populate after multiple rebalancing runs</div>
						</div>
					</Card>
				{/if}
			</div>

			<!-- Right Column (1/3) -->
			<div class="space-y-6">
				<!-- Metrics Comparison -->
				<MetricsComparisonCard
					current={rebalancingData.current_metrics}
					expected={{
						expected_return: rebalancingData.latest.expected_return,
						expected_volatility: rebalancingData.latest.expected_volatility,
						sharpe_ratio: rebalancingData.latest.sharpe_ratio
					}}
				/>

				<!-- Last Updated -->
				<Card title="Last Updated">
					<div class="space-y-3">
						<div>
							<div class="text-xs text-gray-500">Timestamp</div>
							<div class="text-sm font-medium text-black">{lastUpdatedText}</div>
						</div>
						<div>
							<div class="text-xs text-gray-500">Method</div>
							<div class="text-sm">
								<Badge variant="info">{rebalancingData.latest.method}</Badge>
							</div>
						</div>
						<div>
							<div class="text-xs text-gray-500">Portfolio Value</div>
							<div class="text-sm font-medium text-black">
								${rebalancingData.current_portfolio_value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
							</div>
						</div>
						<div>
							<div class="text-xs text-gray-500">Rebalance Threshold</div>
							<div class="text-sm font-medium text-black">
								{(rebalancingData.rebalance_threshold * 100).toFixed(1)}%
							</div>
						</div>
					</div>
				</Card>
			</div>
		</div>
	{/if}
</div>
