<script lang="ts">
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import type { ECharts, EChartsOption } from 'echarts';
	import type { PipelineStageMetric } from '$lib/types/api';

	interface Props {
		data: PipelineStageMetric[];
	}

	let { data }: Props = $props();

	let chartContainer = null as unknown as HTMLElement;
	let chart: ECharts | null = null;

	onMount(() => {
		chart = echarts.init(chartContainer);

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
		if (chart && data && data.length > 0) {
			// Calculate cumulative times for Gantt visualization
			let cumulativeTime = 0;
			const ganttData = data.map((stage, index) => {
				const start = cumulativeTime;
				const end = start + stage.latency_ms;
				cumulativeTime = end;
				return {
					name: stage.stage,
					value: [index, start, end, end - start],
					itemStyle: {
						color: '#3b82f6'
					}
				};
			});

			const option: EChartsOption = {
				backgroundColor: 'transparent',
				tooltip: {
					trigger: 'item',
					backgroundColor: '#ffffff',
					borderColor: '#e5e7eb',
					textStyle: { color: '#374151' },
					formatter: (params: any) => {
						const duration = (params.value[3] / 1000).toFixed(2);
						return `${params.name}<br/>Duration: ${duration}s`;
					}
				},
				grid: {
					left: '15%',
					right: '4%',
					bottom: '3%',
					top: '3%',
					containLabel: true
				},
				xAxis: {
					type: 'value',
					name: 'Time (ms)',
					nameTextStyle: { color: '#6b7280' },
					axisLine: { lineStyle: { color: '#d1d5db' } },
					axisLabel: { color: '#6b7280' },
					splitLine: { lineStyle: { color: '#e5e7eb' } }
				},
				yAxis: {
					type: 'category',
					data: data.map(s => s.stage),
					axisLine: { lineStyle: { color: '#d1d5db' } },
					axisLabel: { color: '#6b7280' }
				},
				series: [
					{
						type: 'custom',
						renderItem: (params: any, api: any) => {
							const categoryIndex = api.value(0);
							const start = api.coord([api.value(1), categoryIndex]);
							const end = api.coord([api.value(2), categoryIndex]);
							const height = api.size([0, 1])[1] * 0.6;

							return {
								type: 'rect',
								shape: {
									x: start[0],
									y: start[1] - height / 2,
									width: end[0] - start[0],
									height: height
								},
								style: api.style()
							};
						},
						encode: {
							x: [1, 2],
							y: 0
						},
						data: ganttData
					}
				]
			};

			chart.setOption(option);
		}
	});
</script>

<div bind:this={chartContainer} style="width: 100%; height: 300px;"></div>
