<script lang="ts">
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import type { ECharts, EChartsOption } from 'echarts';

	interface DataPoint {
		time: string | Date;
		value: number;
	}

	interface SeriesData {
		name: string;
		data: number[];
		color: string;
	}

	interface Props {
		data?: DataPoint[];
		series?: SeriesData[];
		title?: string;
		height?: number;
		color?: string;
		yAxisLabel?: string;
		showLegend?: boolean;
		markLine?: { value: number; label: string; color?: string };
		areaFill?: boolean;
	}

	let { data, series, title, height = 300, color = '#3b82f6', yAxisLabel, showLegend = false, markLine, areaFill = true }: Props = $props();

	let chartContainer = null as unknown as HTMLElement;
	let chart: ECharts | null = null;

	onMount(() => {
		chart = echarts.init(chartContainer);

		// Determine x-axis data
		const xAxisData = data?.map(d => typeof d.time === 'string' ? d.time : d.time.toLocaleString()) || [];

		// Build series config
		let seriesConfig: any[];
		if (series) {
			// Multi-series mode
			seriesConfig = series.map(s => ({
				name: s.name,
				data: s.data,
				type: 'line',
				smooth: true,
				lineStyle: { color: s.color, width: 2 },
				itemStyle: { color: s.color },
				areaStyle: areaFill ? {
					color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
						{ offset: 0, color: s.color + '40' },
						{ offset: 1, color: s.color + '00' }
					])
				} : undefined
			}));
		} else if (data) {
			// Single series mode
			seriesConfig = [
				{
					data: data.map(d => d.value),
					type: 'line',
					smooth: true,
					lineStyle: { color, width: 2 },
					itemStyle: { color },
					markLine: markLine ? {
						silent: true,
						symbol: 'none',
						lineStyle: { type: 'dashed', color: markLine.color || '#ef4444', width: 2 },
						data: [{ yAxis: markLine.value, label: { show: true, formatter: markLine.label } }]
					} : undefined,
					areaStyle: areaFill ? {
						color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
							{ offset: 0, color: color + '40' },
							{ offset: 1, color: color + '00' }
						])
					} : undefined
				}
			];
		} else {
			seriesConfig = [];
		}

		const option: EChartsOption = {
			backgroundColor: 'transparent',
			title: title ? {
				text: title,
				textStyle: { color: '#000000', fontSize: 14 }
			} : undefined,
			legend: showLegend ? {
				top: 'bottom',
				textStyle: { color: '#4b5563' }
			} : undefined,
			tooltip: {
				trigger: 'axis',
				backgroundColor: '#ffffff',
				borderColor: '#d1d5db',
				textStyle: { color: '#000000' }
			},
			grid: {
				left: '3%',
				right: '4%',
				bottom: showLegend ? '12%' : '3%',
				containLabel: true
			},
			xAxis: {
				type: 'category',
				boundaryGap: false,
				data: xAxisData,
				axisLine: { lineStyle: { color: '#d1d5db' } },
				axisLabel: { color: '#4b5563' }
			},
			yAxis: {
				type: 'value',
				name: yAxisLabel,
				nameTextStyle: { color: '#4b5563' },
				axisLine: { lineStyle: { color: '#d1d5db' } },
				axisLabel: { color: '#4b5563' },
				splitLine: { lineStyle: { color: '#e5e7eb' } }
			},
			series: seriesConfig
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
		if (!chart) return;

		if (series) {
			// Update multi-series
			const xAxisData = data?.map(d => typeof d.time === 'string' ? d.time : d.time.toLocaleString()) || [];
			chart.setOption({
				xAxis: { data: xAxisData },
				series: series.map(s => ({ data: s.data }))
			});
		} else if (data) {
			// Update single series
			chart.setOption({
				xAxis: {
					data: data.map(d => typeof d.time === 'string' ? d.time : d.time.toLocaleString())
				},
				series: [{ data: data.map(d => d.value) }]
			});
		}
	});
</script>

<div bind:this={chartContainer} style="width: 100%; height: {height}px;"></div>
