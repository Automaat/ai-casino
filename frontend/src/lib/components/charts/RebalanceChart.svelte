<script lang="ts">
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import type { ECharts, EChartsOption } from 'echarts';
	import type { RebalanceAllocation } from '$lib/types/api';

	interface Props {
		allocations: RebalanceAllocation[];
		height?: number;
	}

	let { allocations, height = 350 }: Props = $props();

	let chartContainer = null as unknown as HTMLElement;
	let chart: ECharts | null = null;

	function updateChart() {
		if (!chart || allocations.length === 0) return;

		const symbols = allocations.map(a => a.symbol);
		const targets = allocations.map(a => a.target_weight * 100);
		const actuals = allocations.map(a => a.current_weight * 100);
		const deltas = allocations.map(a => a.delta * 100);

		// Color-code current bars: green for underweight, red for overweight
		const barColors = deltas.map(d =>
			d < 0 ? '#10b981' : d > 0 ? '#ef4444' : '#6b7280'
		);

		const option: EChartsOption = {
			backgroundColor: 'transparent',
			title: {
				text: 'Rebalance Analysis',
				subtext: 'Green = Underweight, Red = Overweight',
				textStyle: { color: '#e2e8f0', fontSize: 16 },
				subtextStyle: { color: '#94a3b8', fontSize: 12 }
			},
			tooltip: {
				trigger: 'axis',
				axisPointer: { type: 'shadow' },
				formatter: (params: any) => {
					const idx = params[0].dataIndex;
					const symbol = symbols[idx];
					const target = targets[idx];
					const actual = actuals[idx];
					const delta = deltas[idx];
					const action = allocations[idx].action;

					return `
						<strong>${symbol}</strong><br/>
						Target: ${target.toFixed(1)}%<br/>
						Current: ${actual.toFixed(1)}%<br/>
						Delta: ${delta > 0 ? '+' : ''}${delta.toFixed(1)}%<br/>
						Action: ${action}
					`;
				}
			},
			legend: {
				data: ['Target', 'Current'],
				textStyle: { color: '#e2e8f0' }
			},
			xAxis: {
				type: 'category',
				data: symbols,
				axisLabel: { color: '#e2e8f0' },
				axisLine: { lineStyle: { color: '#475569' } }
			},
			yAxis: {
				type: 'value',
				name: 'Weight (%)',
				nameTextStyle: { color: '#e2e8f0' },
				axisLabel: { color: '#e2e8f0' },
				axisLine: { lineStyle: { color: '#475569' } },
				splitLine: { lineStyle: { color: '#334155' } }
			},
			series: [
				{
					name: 'Target',
					type: 'bar',
					data: targets,
					itemStyle: { color: '#3b82f6' }
				},
				{
					name: 'Current',
					type: 'bar',
					data: actuals,
					itemStyle: {
						color: (params: any) => barColors[params.dataIndex]
					}
				}
			]
		};

		chart.setOption(option);
	}

	onMount(() => {
		if (chartContainer) {
			chart = echarts.init(chartContainer, 'dark');
			updateChart();

			const resizeObserver = new ResizeObserver(() => {
				chart?.resize();
			});
			resizeObserver.observe(chartContainer);

			return () => {
				resizeObserver.disconnect();
				chart?.dispose();
			};
		}
	});

	$effect(() => {
		updateChart();
	});
</script>

<div bind:this={chartContainer} style="width: 100%; height: {height}px;"></div>
