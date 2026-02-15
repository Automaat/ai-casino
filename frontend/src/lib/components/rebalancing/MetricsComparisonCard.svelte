<script lang="ts">
	import Card from '$lib/components/ui/Card.svelte';
	import Badge from '$lib/components/ui/Badge.svelte';
	import { formatPercent } from '$lib/utils/format';
	import type { MetricsSnapshot } from '$lib/types/api';

	interface Props {
		current: MetricsSnapshot | null;
		expected: MetricsSnapshot;
	}

	let { current, expected }: Props = $props();

	interface MetricRow {
		label: string;
		currentValue: number;
		expectedValue: number;
		delta: number;
		deltaPercent: number;
		isImprovement: boolean;
	}

	const metrics = $derived.by((): MetricRow[] => {
		if (!current) {
			return [
				{
					label: 'Expected Return',
					currentValue: 0,
					expectedValue: expected.expected_return,
					delta: 0,
					deltaPercent: 0,
					isImprovement: false
				},
				{
					label: 'Expected Volatility',
					currentValue: 0,
					expectedValue: expected.expected_volatility,
					delta: 0,
					deltaPercent: 0,
					isImprovement: false
				},
				{
					label: 'Sharpe Ratio',
					currentValue: 0,
					expectedValue: expected.sharpe_ratio,
					delta: 0,
					deltaPercent: 0,
					isImprovement: false
				}
			];
		}

		const returnDelta = expected.expected_return - current.expected_return;
		const volDelta = expected.expected_volatility - current.expected_volatility;
		const sharpeDelta = expected.sharpe_ratio - current.sharpe_ratio;

		return [
			{
				label: 'Expected Return',
				currentValue: current.expected_return,
				expectedValue: expected.expected_return,
				delta: returnDelta,
				deltaPercent: current.expected_return !== 0 ? (returnDelta / current.expected_return) * 100 : 0,
				isImprovement: returnDelta > 0
			},
			{
				label: 'Expected Volatility',
				currentValue: current.expected_volatility,
				expectedValue: expected.expected_volatility,
				delta: volDelta,
				deltaPercent: current.expected_volatility !== 0 ? (volDelta / current.expected_volatility) * 100 : 0,
				isImprovement: volDelta < 0
			},
			{
				label: 'Sharpe Ratio',
				currentValue: current.sharpe_ratio,
				expectedValue: expected.sharpe_ratio,
				delta: sharpeDelta,
				deltaPercent: current.sharpe_ratio !== 0 ? (sharpeDelta / current.sharpe_ratio) * 100 : 0,
				isImprovement: sharpeDelta > 0
			}
		];
	});
</script>

<Card title="Portfolio Metrics Comparison">
	<div class="space-y-4">
		{#each metrics as metric}
			<div class="border-b border-gray-200 pb-4 last:border-b-0 last:pb-0">
				<div class="text-sm font-medium text-gray-700 mb-2">{metric.label}</div>
				<div class="grid grid-cols-2 gap-4">
					<div>
						<div class="text-xs text-gray-500">Current</div>
						<div class="text-lg font-semibold text-black">
							{formatPercent(metric.currentValue / 100)}
						</div>
					</div>
					<div>
						<div class="text-xs text-gray-500">After Rebalancing</div>
						<div class="text-lg font-semibold text-black">
							{formatPercent(metric.expectedValue / 100)}
						</div>
					</div>
				</div>
				{#if current && metric.delta !== 0}
					<div class="mt-2">
						<Badge variant={metric.isImprovement ? 'success' : 'error'}>
							{metric.delta > 0 ? '+' : ''}{formatPercent(metric.deltaPercent / 100)} ({metric.delta > 0 ? '+' : ''}{(metric.delta * 100).toFixed(2)}pp)
						</Badge>
					</div>
				{/if}
			</div>
		{/each}
	</div>
</Card>
