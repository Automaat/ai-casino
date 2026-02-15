<script lang="ts">
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import type { RebalanceHistoryEntry } from '$lib/types/api';

	interface Props {
		history: RebalanceHistoryEntry[];
		threshold: number;
		height?: number;
	}

	let { history, threshold, height = 350 }: Props = $props();

	let chartContainer: HTMLDivElement;
	let chart: echarts.ECharts | null = null;

	function initChart() {
		if (!chartContainer) return;

		chart = echarts.init(chartContainer);

		const timestamps = history.map(h => new Date(h.timestamp).toLocaleDateString());
		const avgDeviations = history.map(h => h.avg_deviation_pct);
		const maxDeviations = history.map(h => h.max_deviation_pct);

		const option: echarts.EChartsOption = {
			tooltip: {
				trigger: 'axis',
				axisPointer: {
					type: 'cross'
				}
			},
			legend: {
				data: ['Avg Deviation', 'Max Deviation', 'Threshold'],
				bottom: 0
			},
			grid: {
				left: '3%',
				right: '4%',
				bottom: '10%',
				containLabel: true
			},
			xAxis: {
				type: 'category',
				boundaryGap: false,
				data: timestamps
			},
			yAxis: {
				type: 'value',
				name: 'Deviation %',
				axisLabel: {
					formatter: '{value}%'
				}
			},
			series: [
				{
					name: 'Avg Deviation',
					type: 'line',
					smooth: true,
					data: avgDeviations,
					itemStyle: {
						color: '#3b82f6'
					},
					areaStyle: {
						color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
							{ offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
							{ offset: 1, color: 'rgba(59, 130, 246, 0.05)' }
						])
					}
				},
				{
					name: 'Max Deviation',
					type: 'line',
					smooth: true,
					data: maxDeviations,
					itemStyle: {
						color: '#ef4444'
					},
					areaStyle: {
						color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
							{ offset: 0, color: 'rgba(239, 68, 68, 0.3)' },
							{ offset: 1, color: 'rgba(239, 68, 68, 0.05)' }
						])
					}
				},
				{
					name: 'Threshold',
					type: 'line',
					data: Array.from({ length: timestamps.length }).fill(threshold * 100),
					lineStyle: {
						type: 'dashed',
						color: '#f59e0b',
						width: 2
					},
					itemStyle: {
						color: '#f59e0b'
					},
					symbol: 'none'
				}
			]
		};

		chart.setOption(option);

		const resizeObserver = new ResizeObserver(() => {
			chart?.resize();
		});
		resizeObserver.observe(chartContainer);

		return () => {
			resizeObserver.disconnect();
			chart?.dispose();
		};
	}

	onMount(() => {
		return initChart();
	});

	$effect(() => {
		if (chart && history) {
			initChart();
		}
	});
</script>

<div bind:this={chartContainer} style="width: 100%; height: {height}px;"></div>
