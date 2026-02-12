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
			const option: EChartsOption = {
				backgroundColor: 'transparent',
				tooltip: {
					trigger: 'axis',
					backgroundColor: '#ffffff',
					borderColor: '#e5e7eb',
					textStyle: { color: '#374151' },
					axisPointer: {
						type: 'cross'
					}
				},
				legend: {
					data: ['Latency (ms)', 'LLM Calls'],
					textStyle: { color: '#6b7280' },
					top: 0
				},
				grid: {
					left: '3%',
					right: '4%',
					bottom: '3%',
					top: '12%',
					containLabel: true
				},
				xAxis: {
					type: 'category',
					data: data.map(a => a.agent_name),
					axisLine: { lineStyle: { color: '#d1d5db' } },
					axisLabel: { color: '#6b7280', rotate: 45 }
				},
				yAxis: [
					{
						type: 'value',
						name: 'Latency (ms)',
						nameTextStyle: { color: '#6b7280' },
						axisLine: { lineStyle: { color: '#d1d5db' } },
						axisLabel: { color: '#6b7280' },
						splitLine: { lineStyle: { color: '#e5e7eb' } }
					},
					{
						type: 'value',
						name: 'LLM Calls',
						nameTextStyle: { color: '#6b7280' },
						axisLine: { lineStyle: { color: '#d1d5db' } },
						axisLabel: { color: '#6b7280' },
						splitLine: { show: false }
					}
				],
				series: [
					{
						name: 'Latency (ms)',
						type: 'bar',
						data: data.map(a => a.latency_ms),
						itemStyle: { color: '#3b82f6' },
						yAxisIndex: 0
					},
					{
						name: 'LLM Calls',
						type: 'line',
						data: data.map(a => a.llm_calls),
						itemStyle: { color: '#059669' },
						lineStyle: { color: '#059669', width: 2 },
						yAxisIndex: 1
					}
				]
			};

			chart.setOption(option);
		}
	});
</script>

<div bind:this={chartContainer} style="width: 100%; height: 400px;"></div>
