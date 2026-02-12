<script lang="ts">
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import type { ECharts, EChartsOption } from 'echarts';

	interface DataPoint {
		label: string;
		value: number;
		color?: string;
	}

	interface Props {
		data: DataPoint[];
		title?: string;
		height?: number;
		defaultColor?: string;
		yAxisLabel?: string;
		xAxisLabel?: string;
	}

	let { data, title, height = 300, defaultColor = '#3b82f6', yAxisLabel, xAxisLabel }: Props = $props();

	let chartContainer = null as unknown as HTMLElement;
	let chart: ECharts | null = null;

	onMount(() => {
		chart = echarts.init(chartContainer);

		const option: EChartsOption = {
			backgroundColor: 'transparent',
			title: title ? {
				text: title,
				textStyle: { color: '#000000', fontSize: 14 }
			} : undefined,
			tooltip: {
				trigger: 'axis',
				backgroundColor: '#ffffff',
				borderColor: '#d1d5db',
				textStyle: { color: '#000000' },
				axisPointer: {
					type: 'shadow'
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
				data: data.map(d => d.label),
				name: xAxisLabel,
				nameTextStyle: { color: '#4b5563' },
				axisLine: { lineStyle: { color: '#d1d5db' } },
				axisLabel: { color: '#4b5563', rotate: 45 }
			},
			yAxis: {
				type: 'value',
				name: yAxisLabel,
				nameTextStyle: { color: '#4b5563' },
				axisLine: { lineStyle: { color: '#d1d5db' } },
				axisLabel: { color: '#4b5563' },
				splitLine: { lineStyle: { color: '#e5e7eb' } }
			},
			series: [
				{
					data: data.map(d => ({
						value: d.value,
						itemStyle: { color: d.color || defaultColor }
					})),
					type: 'bar',
					barWidth: '60%'
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
	});

	$effect(() => {
		if (chart && data) {
			chart.setOption({
				xAxis: {
					data: data.map(d => d.label)
				},
				series: [{
					data: data.map(d => ({
						value: d.value,
						itemStyle: { color: d.color || defaultColor }
					}))
				}]
			});
		}
	});
</script>

<div bind:this={chartContainer} style="width: 100%; height: {height}px;"></div>
