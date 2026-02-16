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

<div class="max-w-5xl mx-auto px-6 py-8">
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
		<div class="bg-white rounded-lg border border-gray-200 p-8 space-y-6">
			<!-- Header -->
			<div class="flex items-center justify-between">
				<h1 class="text-2xl font-bold text-gray-900">Paper Trading Validation</h1>
				{#if validation.ready_for_live}
					<Badge variant="success" class="text-sm px-4 py-2">Ready</Badge>
				{:else}
					<Badge variant="neutral" class="text-sm px-4 py-2">Not Ready</Badge>
				{/if}
			</div>

			<!-- Summary Info -->
			<div class="grid grid-cols-3 gap-8 text-sm">
				<div>
					<span class="text-gray-500">Assessment Date:</span>
					<span class="ml-2 text-gray-900 font-medium">
						{new Date(validation.assessment_date).toLocaleDateString()}
					</span>
				</div>
				<div>
					<span class="text-gray-500">Paper Trading Duration:</span>
					<span class="ml-2 text-gray-900 font-medium">
						{validation.paper_trading_duration_days} days
					</span>
				</div>
				<div>
					<span class="text-gray-500">Total Paper Trades:</span>
					<span class="ml-2 text-gray-900 font-medium">{validation.total_paper_trades}</span>
				</div>
			</div>

			<!-- Validation Criteria -->
			<div>
				<h2 class="text-base font-semibold text-gray-900 mb-4">Validation Criteria:</h2>
				<div class="space-y-2">
					{#each validation.criteria as criterion}
						<div class="space-y-1">
							<div
								class="bg-gray-800 rounded-md px-4 py-3 flex items-center justify-between"
							>
								<div class="flex items-center gap-3">
									{#if criterion.passed}
										<svg
											class="w-5 h-5 text-green-500 flex-shrink-0"
											fill="currentColor"
											viewBox="0 0 20 20"
										>
											<path
												fill-rule="evenodd"
												d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
												clip-rule="evenodd"
											/>
										</svg>
									{:else}
										<svg
											class="w-5 h-5 text-red-500 flex-shrink-0"
											fill="currentColor"
											viewBox="0 0 20 20"
										>
											<path
												fill-rule="evenodd"
												d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
												clip-rule="evenodd"
											/>
										</svg>
									{/if}
									<span class="text-white font-medium">{criterion.name}</span>
								</div>
								<span class="text-gray-300 text-sm font-mono">
									{formatValue(criterion.current_value, criterion.name)} /
									{formatValue(criterion.threshold, criterion.name)}
								</span>
							</div>
							<p class="text-sm text-gray-500 pl-12">{criterion.message}</p>
						</div>
					{/each}
				</div>
			</div>

			<!-- Recommendations -->
			{#if validation.recommendations.length > 0}
				<div>
					<h2 class="text-base font-semibold text-gray-900 mb-3">Recommendations:</h2>
					<ul class="space-y-2">
						{#each validation.recommendations as recommendation}
							<li class="flex items-start gap-2 text-sm text-gray-500">
								<span class="mt-1">•</span>
								<span>{recommendation}</span>
							</li>
						{/each}
					</ul>
				</div>
			{/if}
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
