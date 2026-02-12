<script lang="ts">
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import type { ECharts, EChartsOption } from 'echarts';
	import type { AgentTimingMetric } from '$lib/types/api';

	interface Props {
		data: AgentTimingMetric[];
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
			// Sort by latency descending
			const sorted = [...data].sort((a, b) => b.latency_ms - a.latency_ms);

			const option: EChartsOption = {
				backgroundColor: 'transparent',
				tooltip: {
					trigger: 'axis',
					backgroundColor: '#ffffff',
					borderColor: '#e5e7eb',
					textStyle: { color: '#374151' },
					axisPointer: {
						type: 'shadow'
					},
					formatter: (params: any) => {
						const value = params[0].value.toFixed(2);
						return `${params[0].name}<br/>Latency: ${value}s`;
					}
				},
				grid: {
					left: '3%',
					right: '4%',
					bottom: '3%',
					top: '3%',
					containLabel: true
				},
				xAxis: {
					type: 'value',
					name: 'Latency (s)',
					nameTextStyle: { color: '#6b7280' },
					axisLine: { lineStyle: { color: '#d1d5db' } },
					axisLabel: { color: '#6b7280' },
					splitLine: { lineStyle: { color: '#e5e7eb' } }
				},
				yAxis: {
					type: 'category',
					data: sorted.map(a => a.agent_name),
					axisLine: { lineStyle: { color: '#d1d5db' } },
					axisLabel: { color: '#6b7280' }
				},
				series: [
					{
						type: 'bar',
						data: sorted.map(a => a.latency_ms / 1000),
						itemStyle: {
							color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
								{ offset: 0, color: '#3b82f6' },
								{ offset: 1, color: '#60a5fa' }
							])
						}
					}
				]
			};

			chart.setOption(option);
		}
	});
</script>

<div bind:this={chartContainer} style="width: 100%; height: 300px;"></div>
