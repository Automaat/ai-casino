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
		chart = echarts.init(chartContainer, 'dark');

		const option: EChartsOption = {
			backgroundColor: 'transparent',
			title: title ? {
				text: title,
				textStyle: { color: '#e2e8f0', fontSize: 14 }
			} : undefined,
			tooltip: {
				trigger: 'axis',
				backgroundColor: '#1e293b',
				borderColor: '#334155',
				textStyle: { color: '#e2e8f0' },
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
				nameTextStyle: { color: '#94a3b8' },
				axisLine: { lineStyle: { color: '#334155' } },
				axisLabel: { color: '#94a3b8', rotate: 45 }
			},
			yAxis: {
				type: 'value',
				name: yAxisLabel,
				nameTextStyle: { color: '#94a3b8' },
				axisLine: { lineStyle: { color: '#334155' } },
				axisLabel: { color: '#94a3b8' },
				splitLine: { lineStyle: { color: '#334155' } }
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
