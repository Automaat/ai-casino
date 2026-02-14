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
		showLegend?: boolean;
	}

	let { data, title, height = 300, showLegend = true }: Props = $props();

	let chartContainer = null as unknown as HTMLElement;
	let chart: ECharts | null = null;

	onMount(() => {
		chart = echarts.init(chartContainer);

		const option: EChartsOption = {
			backgroundColor: 'transparent',
			title: title ? {
				text: title,
				textStyle: { color: '#000000', fontSize: 14 },
				left: 'center'
			} : undefined,
			tooltip: {
				trigger: 'item',
				backgroundColor: '#ffffff',
				borderColor: '#d1d5db',
				textStyle: { color: '#000000' },
				formatter: '{a} <br/>{b}: {c} ({d}%)'
			},
			legend: showLegend ? {
				orient: 'horizontal',
				bottom: '0',
				textStyle: { color: '#4b5563' }
			} : undefined,
			series: [
				{
					name: title || 'Distribution',
					type: 'pie',
					radius: showLegend ? '60%' : '70%',
					center: ['50%', '50%'],
					data: data.map(d => ({
						value: d.value,
						name: d.label,
						itemStyle: { color: d.color }
					})),
					emphasis: {
						itemStyle: {
							shadowBlur: 10,
							shadowOffsetX: 0,
							shadowColor: 'rgba(0, 0, 0, 0.5)'
						}
					},
					label: {
						color: '#4b5563',
						formatter: '{b}: {d}%'
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
				series: [{
					data: data.map(d => ({
						value: d.value,
						name: d.label,
						itemStyle: { color: d.color }
					}))
				}]
			});
		}
	});
</script>

<div bind:this={chartContainer} style="width: 100%; height: {height}px;"></div>
