<script lang="ts">
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import type { ECharts, EChartsOption } from 'echarts';

	interface Props {
		sectorStrengths: Record<string, number>;
		sectorMomenta: Record<string, string>;
		leadingSectors: string[];
		laggingSectors: string[];
		flaggedPositions: string[];
	}

	let { sectorStrengths, sectorMomenta, leadingSectors, laggingSectors, flaggedPositions }: Props = $props();

	// Sector ETF ticker to name mapping
	const SECTOR_NAMES: Record<string, string> = {
		'XLK': 'Technology',
		'XLV': 'Healthcare',
		'XLF': 'Financials',
		'XLY': 'Consumer Discretionary',
		'XLP': 'Consumer Staples',
		'XLE': 'Energy',
		'XLI': 'Industrials',
		'XLU': 'Utilities',
		'XLB': 'Materials',
		'XLRE': 'Real Estate',
		'XLC': 'Communication Services'
	};

	let chartContainer = null as unknown as HTMLElement;
	let chart: ECharts | null = null;

	function updateChart() {
		if (!chart || Object.keys(sectorStrengths).length === 0) return;

		// Sort sectors by strength (descending)
		const sortedSectors = Object.entries(sectorStrengths)
			.sort(([, a], [, b]) => b - a)
			.map(([ticker]) => ticker);

		// Convert to 2D array for heatmap: [[strength1], [strength2], ...]
		const data: [number, number, number][] = sortedSectors.map((ticker, idx) => [
			0, // x-axis (single column)
			idx, // y-axis (sector index)
			sectorStrengths[ticker] * 100 // Convert to percentage
		]);

		// Y-axis labels with sector names and momentum indicators
		const yAxisLabels = sortedSectors.map((ticker) => {
			const name = SECTOR_NAMES[ticker] || ticker;
			const momentum = sectorMomenta[ticker] || 'NEUTRAL';
			const momentumIcon = momentum === 'ACCELERATING' ? '▲' : momentum === 'DECELERATING' ? '▼' : '─';
			const isLeading = leadingSectors.includes(ticker);
			const prefix = isLeading ? '★ ' : '';
			return `${prefix}${name} ${momentumIcon}`;
		});

		// Build title with leading sectors
		const leadingNames = leadingSectors.map(ticker => SECTOR_NAMES[ticker] || ticker).join(', ');
		const chartTitle = `Sector Rotation${leadingNames ? ` (Leading: ${leadingNames})` : ''}`;

		const option: EChartsOption = {
			backgroundColor: 'transparent',
			title: {
				text: chartTitle,
				textStyle: { color: '#e2e8f0', fontSize: 14 },
				left: 'center'
			},
			tooltip: {
				position: 'top',
				backgroundColor: '#1e293b',
				borderColor: '#334155',
				textStyle: { color: '#e2e8f0' },
				formatter: (params: any) => {
					const sectorIdx = params.data[1];
					const ticker = sortedSectors[sectorIdx];
					const name = SECTOR_NAMES[ticker] || ticker;
					const strength = params.data[2].toFixed(1);
					const momentum = sectorMomenta[ticker] || 'NEUTRAL';
					const isLeading = leadingSectors.includes(ticker);
					const isLagging = laggingSectors.includes(ticker);
					const status = isLeading ? '(Leading)' : isLagging ? '(Lagging)' : '';
					const flagged = flaggedPositions.filter(pos => pos.includes(ticker));
					const flaggedText = flagged.length > 0 ? `<br/>Flagged: ${flagged.join(', ')}` : '';
					return `<b>${name}</b> ${status}<br/>Strength: ${strength}%<br/>Momentum: ${momentum}${flaggedText}`;
				}
			},
			grid: {
				height: '80%',
				top: '12%',
				left: '30%',
				right: '10%'
			},
			xAxis: {
				type: 'category',
				data: ['Relative Strength'],
				splitArea: { show: false },
				axisLabel: { color: '#94a3b8', fontSize: 12 }
			},
			yAxis: {
				type: 'category',
				data: yAxisLabels,
				splitArea: { show: true },
				axisLabel: { color: '#94a3b8', fontSize: 11 }
			},
			visualMap: {
				min: 0,
				max: 100,
				calculable: true,
				orient: 'horizontal',
				left: 'center',
				bottom: '3%',
				inRange: {
					color: ['#ef4444', '#f59e0b', '#84cc16', '#10b981']
				},
				textStyle: { color: '#94a3b8' },
				formatter: '{value}%' as any
			},
			series: [
				{
					name: 'Sector Strength',
					type: 'heatmap',
					data,
					label: {
						show: true,
						formatter: (params: any) => `${params.data[2].toFixed(1)}%`,
						color: '#e2e8f0',
						fontSize: 11
					},
					emphasis: {
						itemStyle: {
							shadowBlur: 10,
							shadowColor: 'rgba(0, 0, 0, 0.5)',
							borderColor: '#e2e8f0',
							borderWidth: 2
						}
					},
					itemStyle: {
						borderColor: ((params: any) => {
							const sectorIdx = params.data[1];
							const ticker = sortedSectors[sectorIdx];
							return leadingSectors.includes(ticker) ? '#fbbf24' : '#334155';
						}) as any,
						borderWidth: ((params: any) => {
							const sectorIdx = params.data[1];
							const ticker = sortedSectors[sectorIdx];
							return leadingSectors.includes(ticker) ? 2 : 0.5;
						}) as any
					}
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

	$effect(() => {
		updateChart();
	});
</script>

<div bind:this={chartContainer} style="width: 100%; height: 450px;"></div>
