<script lang="ts">
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import type { RebalanceHistoryEntry } from '$lib/types/api';

	interface Props {
		history: RebalanceHistoryEntry[];
		height?: number;
	}

	let { history, height = 350 }: Props = $props();

	let chartContainer: HTMLDivElement;
	let chart: echarts.ECharts | null = null;
	let selectedMetric = $state<'return' | 'volatility' | 'sharpe'>('return');

	function initChart() {
		if (!chartContainer) return;

		chart = echarts.init(chartContainer);
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

	function updateChart() {
		if (!chart || !history.length) return;

		const timestamps = history.map(h => new Date(h.timestamp).toLocaleDateString());

		let data: number[];
		let name: string;
		let color: string;
		let yAxisLabel: string;

		switch (selectedMetric) {
			case 'return':
				data = history.map(h => h.metrics.expected_return * 100);
				name = 'Expected Return';
				color = '#10b981';
				yAxisLabel = 'Return %';
				break;
			case 'volatility':
				data = history.map(h => h.metrics.expected_volatility * 100);
				name = 'Expected Volatility';
				color = '#ef4444';
				yAxisLabel = 'Volatility %';
				break;
			case 'sharpe':
				data = history.map(h => h.metrics.sharpe_ratio * 100);
				name = 'Sharpe Ratio';
				color = '#3b82f6';
				yAxisLabel = 'Sharpe Ratio';
				break;
		}

		const option: echarts.EChartsOption = {
			tooltip: {
				trigger: 'axis',
				formatter: (params: any) => {
					const param = params[0];
					return `${param.name}<br/>${param.seriesName}: ${param.value.toFixed(2)}%`;
				}
			},
			grid: {
				left: '3%',
				right: '4%',
				bottom: '3%',
				containLabel: true
			},
			xAxis: {
				type: 'category',
				boundaryGap: false,
				data: timestamps
			},
			yAxis: {
				type: 'value',
				name: yAxisLabel,
				axisLabel: {
					formatter: '{value}%'
				}
			},
			series: [
				{
					name,
					type: 'line',
					smooth: true,
					data,
					itemStyle: {
						color
					},
					areaStyle: {
						color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
							{ offset: 0, color: `${color}4D` },
							{ offset: 1, color: `${color}0D` }
						])
					}
				}
			]
		};

		chart.setOption(option);
	}

	onMount(() => {
		return initChart();
	});

	$effect(() => {
		updateChart();
	});
</script>

<div class="space-y-4">
	<div class="flex gap-2">
		<button
			onclick={() => selectedMetric = 'return'}
			class="px-3 py-1 text-sm rounded {selectedMetric === 'return' ? 'bg-green-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}"
		>
			Expected Return
		</button>
		<button
			onclick={() => selectedMetric = 'volatility'}
			class="px-3 py-1 text-sm rounded {selectedMetric === 'volatility' ? 'bg-red-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}"
		>
			Volatility
		</button>
		<button
			onclick={() => selectedMetric = 'sharpe'}
			class="px-3 py-1 text-sm rounded {selectedMetric === 'sharpe' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}"
		>
			Sharpe Ratio
		</button>
	</div>
	<div bind:this={chartContainer} style="width: 100%; height: {height}px;"></div>
</div>
