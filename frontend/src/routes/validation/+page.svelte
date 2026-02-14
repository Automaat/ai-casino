<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import Card from '$lib/components/ui/Card.svelte';
	import MetricCard from '$lib/components/ui/MetricCard.svelte';
	import Badge from '$lib/components/ui/Badge.svelte';
	import { api } from '$lib/api/client';
	import type { PaperTradingValidation, ValidationCriterion } from '$lib/types/api';

	let validation: PaperTradingValidation | null = $state(null);
	let loading = $state(true);
	let error: string | null = $state(null);

	async function loadData() {
		loading = true;
		error = null;
		try {
			validation = await api.getPaperTradingValidation();
		} catch (e) {
			console.error('Failed to load validation data:', e);
			error = e instanceof Error ? e.message : 'Failed to load validation data';
		} finally {
			loading = false;
		}
	}

	onMount(() => {
		loadData();
	});

	$effect(() => {
		if ($page.url.pathname === '/validation') {
			loadData();
		}
	});

	function isLowerBetterMetric(criterion: ValidationCriterion): boolean {
		// Only "Max Drawdown" is a lower-is-better metric
		return criterion.name === 'Max Drawdown';
	}

	function getProgressPercent(criterion: ValidationCriterion): number {
		if (criterion.threshold === 0) return 0;

		const ratio = criterion.current_value / criterion.threshold;
		let risk: number;

		if (isLowerBetterMetric(criterion)) {
			// For lower-is-better metrics (e.g. Max Drawdown),
			// risk increases as current_value approaches/exceeds the threshold
			risk = Math.max(0, Math.min(ratio, 1));
		} else {
			// For higher-is-better metrics, risk increases as current_value
			// falls below the threshold. At/above threshold => no risk
			if (ratio >= 1) {
				risk = 0;
			} else {
				risk = 1 - Math.max(0, ratio);
			}
		}

		return risk * 100;
	}

	function getProgressColor(criterion: ValidationCriterion): string {
		if (criterion.passed) return 'bg-green-600';
		const percent = getProgressPercent(criterion);
		if (percent >= 75) return 'bg-yellow-500';
		if (percent >= 50) return 'bg-orange-500';
		return 'bg-red-500';
	}

	function formatValue(value: number, name: string): string {
		if (name === 'Duration') return `${value.toFixed(0)} days`;
		if (name === 'Min Trades') return value.toFixed(0);
		if (name === 'Win Rate') return `${(value * 100).toFixed(1)}%`;
		if (name === 'Max Drawdown') return `${value.toFixed(1)}%`;
		return value.toFixed(2);
	}
</script>

<svelte:head>
	<title>Paper Trading Validation - AI Casino</title>
</svelte:head>

<div class="space-y-8">
	<div>
		<h1 class="text-3xl font-bold text-black">Paper Trading Validation</h1>
		<p class="mt-2 text-gray-600">Track progress toward live trading promotion</p>
	</div>

	{#if loading}
		<div class="text-center py-12">
			<div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gray-600"></div>
			<p class="mt-4 text-gray-600">Loading validation data...</p>
		</div>
	{:else if error}
		<Card title="Error">
			<p class="text-red-600">{error}</p>
		</Card>
	{:else if validation}
		<!-- Status Badge -->
		<Card title="Live Trading Status">
			<div class="flex items-center justify-center py-6">
				{#if validation.ready_for_live}
					<Badge variant="success" class="text-lg px-6 py-3">
						✓ Ready for Live Trading
					</Badge>
				{:else}
					<Badge variant="neutral" class="text-lg px-6 py-3">
						⏳ In Progress
					</Badge>
				{/if}
			</div>
		</Card>

		<!-- Summary Metrics -->
		<div class="grid grid-cols-1 md:grid-cols-3 gap-6">
			<MetricCard
				title="Duration"
				value={validation.paper_trading_duration_days}
				subtitle="days in paper trading"
				icon="📅"
			/>
			<MetricCard
				title="Total Trades"
				value={validation.total_paper_trades}
				subtitle="trades executed"
				icon="📊"
			/>
			<MetricCard
				title="Criteria Met"
				value={`${validation.criteria.filter(c => c.passed).length}/${validation.criteria.length}`}
				subtitle="validation criteria"
				icon="✓"
			/>
		</div>

		<!-- Progress Bars -->
		<Card title="Validation Criteria">
			<div class="space-y-6">
				{#each validation.criteria as criterion}
					{@const percent = getProgressPercent(criterion)}
					{@const color = getProgressColor(criterion)}

					<div>
						<div class="flex items-center justify-between mb-2">
							<div class="flex items-center gap-2">
								<span class="font-semibold text-black">{criterion.name}</span>
								{#if criterion.passed}
									<Badge variant="success">✓ Passed</Badge>
								{:else}
									<Badge variant="neutral">In Progress</Badge>
								{/if}
							</div>
							<span class="text-sm text-gray-600">
								{formatValue(criterion.current_value, criterion.name)} /
								{formatValue(criterion.threshold, criterion.name)}
							</span>
						</div>

						<!-- Progress bar -->
						<div class="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
							<div
								class="h-full transition-all duration-300 {color}"
								style="width: {percent}%"
							></div>
						</div>

						<p class="mt-1 text-sm text-gray-600">{criterion.message}</p>
					</div>
				{/each}
			</div>
		</Card>

		<!-- Recommendations -->
		{#if validation.recommendations.length > 0}
			<Card title="Recommendations">
				<ul class="space-y-3">
					{#each validation.recommendations as recommendation}
						<li class="flex items-start gap-3">
							<span class="text-gray-600 mt-0.5">•</span>
							<span class="text-gray-700">{recommendation}</span>
						</li>
					{/each}
				</ul>
			</Card>
		{/if}

		<!-- Assessment Info -->
		<div class="text-center text-sm text-gray-600">
			Last assessed: {new Date(validation.assessment_date).toLocaleString()}
		</div>
	{/if}
</div>

<style>
	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	.animate-spin {
		animation: spin 1s linear infinite;
	}
</style>
