<script lang="ts">
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import type { ECharts, EChartsOption } from 'echarts';

	interface TreeNode {
		name: string;
		value: number;
	}

	interface Props {
		data: TreeNode[];
		title?: string;
		height?: number;
	}

	let { data, title = 'Portfolio Allocation', height = 400 }: Props = $props();

	let chartContainer: HTMLElement;
	let chart: ECharts | null = null;

	function updateChart() {
		if (!chart || data.length === 0) return;

		const option: EChartsOption = {
			backgroundColor: 'transparent',
			title: {
				text: title,
				textStyle: { color: '#e2e8f0', fontSize: 16 }
			},
			tooltip: {
				formatter: (params: any) => {
					return `${params.name}: $${params.value.toLocaleString()}`;
				}
			},
			series: [
				{
					type: 'treemap',
					data,
					roam: false,
					breadcrumb: { show: false },
					label: {
						show: true,
						formatter: (params: any) => {
							return `${params.name}\n$${params.value.toLocaleString()}`;
						},
						color: '#e2e8f0'
					},
					itemStyle: {
						borderColor: '#1e293b',
						borderWidth: 2
					},
					levels: [
						{
							itemStyle: {
								borderWidth: 0,
								gapWidth: 5
							}
						},
						{
							itemStyle: {
								gapWidth: 1
							},
							colorSaturation: [0.35, 0.5]
						}
					],
					visualDimension: 0,
					visualMin: 0,
					visualMax: Math.max(...data.map(d => d.value))
				}
			]
		};

		chart.setOption(option);
	}

	onMount(() => {
		chart = echarts.init(chartContainer, 'dark');
		updateChart();

		const resizeObserver = new ResizeObserver(() => {
			chart?.resize();
		});
		resizeObserver.observe(chartContainer);

		return () => {
			resizeObserver.disconnect();
			chart?.dispose();
		};
	});

	// Update chart when data or title changes
	$effect(() => {
		updateChart();
	});
</script>

<div bind:this={chartContainer} style="width: 100%; height: {height}px;"></div>
