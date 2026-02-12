<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { createChart, type IChartApi, type ISeriesApi, ColorType } from 'lightweight-charts';

	interface CandlestickData {
		time: string;
		open: number;
		high: number;
		low: number;
		close: number;
	}

	interface Props {
		data: CandlestickData[];
		symbol?: string;
		height?: number;
	}

	let { data, symbol = '', height = 400 }: Props = $props();

	let chartContainer = null as unknown as HTMLElement;
	let chart: IChartApi | null = null;
	let candlestickSeries: ISeriesApi<'Candlestick'> | null = null;

	onMount(() => {
		const chartInstance = createChart(chartContainer, {
			layout: {
				background: { type: ColorType.Solid, color: 'transparent' },
				textColor: '#4b5563'
			},
			grid: {
				vertLines: { color: '#e5e7eb' },
				horzLines: { color: '#e5e7eb' }
			},
			width: chartContainer.clientWidth,
			height,
			timeScale: {
				timeVisible: true,
				secondsVisible: false
			}
		});
		chart = chartInstance;

		candlestickSeries = (chartInstance as any).addCandlestickSeries({
			upColor: '#059669',
			downColor: '#ef4444',
			borderVisible: false,
			wickUpColor: '#059669',
			wickDownColor: '#ef4444'
		});

		if (data.length > 0 && candlestickSeries) {
			candlestickSeries.setData(data);
		}

		// Auto-resize
		const resizeObserver = new ResizeObserver((entries) => {
			if (chart && entries[0]) {
				chart.applyOptions({ width: entries[0].contentRect.width });
			}
		});
		resizeObserver.observe(chartContainer);

		return () => {
			resizeObserver.disconnect();
		};
	});

	// Update series when data changes
	$effect(() => {
		if (candlestickSeries && data.length > 0) {
			candlestickSeries.setData(data);
		}
	});

	onDestroy(() => {
		chart?.remove();
	});
</script>

<div class="w-full">
	{#if symbol}
		<div class="mb-2 text-sm font-medium text-gray-700">{symbol}</div>
	{/if}
	<div bind:this={chartContainer} class="rounded-lg overflow-hidden bg-white"></div>
</div>
