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

	let chartContainer = null as unknown as HTMLElement;
	let chart: ECharts | null = null;

	function updateChart() {
		if (!chart || data.length === 0) return;

		const option: EChartsOption = {
			backgroundColor: 'transparent',
			color: ['#e5e7eb', '#d1d5db', '#9ca3af', '#6b7280', '#4b5563'],
			title: {
				text: title,
				textStyle: { color: '#000000', fontSize: 16 }
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
						color: '#000000'
					},
					itemStyle: {
						borderColor: '#ffffff',
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
							colorSaturation: [0.15, 0.25]
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
		chart = echarts.init(chartContainer);
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
