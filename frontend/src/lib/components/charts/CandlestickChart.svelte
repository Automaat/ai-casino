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

	let chartContainer: HTMLElement = undefined as unknown as HTMLElement;
	let chart: IChartApi | null = null;
	let candlestickSeries: ISeriesApi<'Candlestick'> | null = null;

	onMount(() => {
		chart = createChart(chartContainer, {
			layout: {
				background: { type: ColorType.Solid, color: 'transparent' },
				textColor: '#94a3b8'
			},
			grid: {
				vertLines: { color: '#334155' },
				horzLines: { color: '#334155' }
			},
			width: chartContainer.clientWidth,
			height,
			timeScale: {
				timeVisible: true,
				secondsVisible: false
			}
		});

		candlestickSeries = chart.addCandlestickSeries({
			upColor: '#10b981',
			downColor: '#ef4444',
			borderVisible: false,
			wickUpColor: '#10b981',
			wickDownColor: '#ef4444'
		});

		if (data.length > 0) {
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
		<div class="mb-2 text-sm font-medium text-slate-300">{symbol}</div>
	{/if}
	<div bind:this={chartContainer} class="rounded-lg overflow-hidden bg-slate-900"></div>
</div>
