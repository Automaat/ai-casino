<script lang="ts">
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import type { ECharts, EChartsOption } from 'echarts';

	interface DataPoint {
		time: string | Date;
		value: number;
	}

	interface Props {
		data: DataPoint[];
		title?: string;
		height?: number;
		color?: string;
		yAxisLabel?: string;
	}

	let { data, title, height = 300, color = '#3b82f6', yAxisLabel }: Props = $props();

	let chartContainer: HTMLElement;
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
				textStyle: { color: '#e2e8f0' }
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
				data: data.map(d => typeof d.time === 'string' ? d.time : d.time.toLocaleString()),
				axisLine: { lineStyle: { color: '#334155' } },
				axisLabel: { color: '#94a3b8' }
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
					data: data.map(d => d.value),
					type: 'line',
					smooth: true,
					lineStyle: { color, width: 2 },
					itemStyle: { color },
					areaStyle: {
						color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
							{ offset: 0, color: color + '40' },
							{ offset: 1, color: color + '00' }
						])
					}
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
					data: data.map(d => typeof d.time === 'string' ? d.time : d.time.toLocaleString())
				},
				series: [{ data: data.map(d => d.value) }]
			});
		}
	});
</script>

<div bind:this={chartContainer} style="width: 100%; height: {height}px;"></div>
