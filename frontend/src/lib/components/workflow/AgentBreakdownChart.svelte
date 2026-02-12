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
		chart = echarts.init(chartContainer, 'dark');

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
					backgroundColor: '#1e293b',
					borderColor: '#334155',
					textStyle: { color: '#e2e8f0' },
					axisPointer: {
						type: 'cross'
					}
				},
				legend: {
					data: ['Latency (ms)', 'LLM Calls'],
					textStyle: { color: '#94a3b8' },
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
					axisLine: { lineStyle: { color: '#334155' } },
					axisLabel: { color: '#94a3b8', rotate: 45 }
				},
				yAxis: [
					{
						type: 'value',
						name: 'Latency (ms)',
						nameTextStyle: { color: '#94a3b8' },
						axisLine: { lineStyle: { color: '#334155' } },
						axisLabel: { color: '#94a3b8' },
						splitLine: { lineStyle: { color: '#334155' } }
					},
					{
						type: 'value',
						name: 'LLM Calls',
						nameTextStyle: { color: '#94a3b8' },
						axisLine: { lineStyle: { color: '#334155' } },
						axisLabel: { color: '#94a3b8' },
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
						itemStyle: { color: '#10b981' },
						lineStyle: { color: '#10b981', width: 2 },
						yAxisIndex: 1
					}
				]
			};

			chart.setOption(option);
		}
	});
</script>

<div bind:this={chartContainer} style="width: 100%; height: 400px;"></div>
