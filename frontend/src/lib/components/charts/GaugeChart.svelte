<script lang="ts">
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import type { ECharts, EChartsOption } from 'echarts';

	interface Thresholds {
		low: number;
		medium: number;
		high: number;
	}

	interface Props {
		value: number;
		min?: number;
		max?: number;
		thresholds: Thresholds;
		title: string;
		unit?: string;
	}

	let { value, min = 0, max = 100, thresholds, title, unit = '%' }: Props = $props();

	let chartContainer = null as unknown as HTMLElement;
	let chart: ECharts | null = null;

	onMount(() => {
		chart = echarts.init(chartContainer, 'dark');

		const option: EChartsOption = {
			backgroundColor: 'transparent',
			series: [
				{
					type: 'gauge',
					min,
					max,
					startAngle: 200,
					endAngle: -20,
					radius: '80%',
					center: ['50%', '60%'],
					splitNumber: 4,
					axisLine: {
						lineStyle: {
							width: 20,
							color: [
								[thresholds.low / max, '#10b981'],
								[thresholds.medium / max, '#f59e0b'],
								[1, '#ef4444']
							]
						}
					},
					pointer: {
						itemStyle: {
							color: '#e2e8f0'
						},
						width: 5,
						length: '60%'
					},
					axisTick: {
						distance: -20,
						length: 6,
						lineStyle: {
							color: '#334155',
							width: 1
						}
					},
					splitLine: {
						distance: -20,
						length: 12,
						lineStyle: {
							color: '#334155',
							width: 2
						}
					},
					axisLabel: {
						color: '#94a3b8',
						distance: 20,
						fontSize: 11,
						formatter: (value: number) => value.toFixed(0)
					},
					title: {
						offsetCenter: [0, '85%'],
						fontSize: 14,
						color: '#e2e8f0'
					},
					detail: {
						fontSize: 22,
						offsetCenter: [0, '50%'],
						valueAnimation: true,
						formatter: (value: number) => `{value|${value.toFixed(1)}}{unit|${unit}}`,
						rich: {
							value: {
								fontSize: 24,
								fontWeight: 'bold',
								color: '#e2e8f0'
							},
							unit: {
								fontSize: 16,
								color: '#94a3b8',
								padding: [0, 0, 0, 4]
							}
						}
					},
					data: [
						{
							value,
							name: title
						}
					]
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
		if (chart) {
			chart.setOption({
				series: [
					{
						data: [{ value, name: title }]
					}
				]
			});
		}
	});
</script>

<div bind:this={chartContainer} style="width: 100%; height: 250px;"></div>
