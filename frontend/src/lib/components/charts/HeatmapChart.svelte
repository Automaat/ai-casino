<script lang="ts">
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import type { ECharts, EChartsOption } from 'echarts';

	interface Props {
		symbols: string[];
		matrix: number[][];
		title?: string;
		height?: number;
	}

	let { symbols, matrix, title = 'Correlation Matrix', height = 500 }: Props = $props();

	let chartContainer = null as unknown as HTMLElement;
	let chart: ECharts | null = null;

	function updateChart() {
		if (!chart || symbols.length === 0 || matrix.length === 0) return;

		// Convert matrix to ECharts format: [x, y, value]
		const data: [number, number, number][] = [];
		for (let i = 0; i < symbols.length; i++) {
			for (let j = 0; j < symbols.length; j++) {
				data.push([i, j, matrix[i][j]]);
			}
		}

		const option: EChartsOption = {
			backgroundColor: 'transparent',
			title: {
				text: title,
				textStyle: { color: '#000000', fontSize: 16 }
			},
			tooltip: {
				position: 'top',
				formatter: (params: any) => {
					const x = symbols[params.data[0]];
					const y = symbols[params.data[1]];
					const val = params.data[2].toFixed(2);
					return `${x} vs ${y}: ${val}`;
				}
			},
			grid: {
				height: '80%',
				top: '10%'
			},
			xAxis: {
				type: 'category',
				data: symbols,
				splitArea: { show: true },
				axisLabel: { color: '#4b5563' }
			},
			yAxis: {
				type: 'category',
				data: symbols,
				splitArea: { show: true },
				axisLabel: { color: '#4b5563' }
			},
			visualMap: {
				min: -1,
				max: 1,
				calculable: true,
				orient: 'horizontal',
				left: 'center',
				bottom: '5%',
				inRange: {
					color: ['#ef4444', '#f59e0b', '#059669']
				},
				textStyle: { color: '#4b5563' }
			},
			series: [
				{
					name: 'Correlation',
					type: 'heatmap',
					data,
					label: {
						show: true,
						formatter: (params: any) => params.data[2].toFixed(2),
						color: '#000000'
					},
					emphasis: {
						itemStyle: {
							shadowBlur: 10,
							shadowColor: 'rgba(0, 0, 0, 0.5)'
						}
					}
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

	// Update chart when symbols or matrix changes
	$effect(() => {
		updateChart();
	});
</script>

<div bind:this={chartContainer} style="width: 100%; height: {height}px;"></div>
