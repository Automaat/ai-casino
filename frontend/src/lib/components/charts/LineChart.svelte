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
				textStyle: { color: '#000000' }
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
				axisLine: { lineStyle: { color: '#d1d5db' } },
				axisLabel: { color: '#4b5563' }
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
