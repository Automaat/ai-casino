<script lang="ts">
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import type { ECharts, EChartsOption } from 'echarts';

	interface SankeyNode {
		name: string;
		itemStyle?: { color: string };
	}

	interface SankeyLink {
		source: string;
		target: string;
		value: number;
	}

	interface SankeyData {
		nodes: (SankeyNode | { name: string; itemStyle: { color: string } })[];
		links: (SankeyLink | { source: string; target: string; value: number })[];
	}

	interface Props {
		data: SankeyData | null;
		title?: string;
		height?: number;
	}

	let { data, title, height = 400 }: Props = $props();

	let chartContainer = null as unknown as HTMLElement;
	let chart: ECharts | null = null;

	onMount(() => {
		chart = echarts.init(chartContainer);

		const option: EChartsOption = {
			backgroundColor: 'transparent',
			title: title
				? {
						text: title,
						textStyle: { color: '#000000', fontSize: 14 }
					}
				: undefined,
			tooltip: {
				trigger: 'item',
				backgroundColor: '#ffffff',
				borderColor: '#d1d5db',
				textStyle: { color: '#000000' },
				formatter: (params: any) => {
					if (params.dataType === 'edge') {
						return `${params.data.source} → ${params.data.target}<br/>Count: ${params.data.value}`;
					} else {
						return params.name;
					}
				}
			},
			series: [
				{
					type: 'sankey',
					data: data?.nodes || [],
					links: data?.links || [],
					emphasis: {
						focus: 'adjacency'
					},
					lineStyle: {
						color: 'gradient',
						curveness: 0.5
					},
					label: {
						color: '#000000',
						fontSize: 12
					},
					nodeGap: 20,
					layoutIterations: 32
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
				series: [
					{
						data: data.nodes,
						links: data.links
					}
				]
			});
		}
	});
</script>

<div bind:this={chartContainer} style="width: 100%; height: {height}px;"></div>
