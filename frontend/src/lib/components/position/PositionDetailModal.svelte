<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { api } from '$lib/api/client';
	import type { PositionTimelineResponse } from '$lib/types/api';
	import TimelineView from './TimelineView.svelte';

	export let symbol: string;
	export let onClose: () => void;

	let timeline: PositionTimelineResponse | null = null;
	let loading = true;
	let error: string | null = null;

	onMount(() => {
		loadTimeline();

		// Prevent body scroll
		document.body.classList.add('modal-open');

		// Close on escape key
		const handleEscape = (e: KeyboardEvent) => {
			if (e.key === 'Escape') {
				onClose();
			}
		};
		window.addEventListener('keydown', handleEscape);
		return () => window.removeEventListener('keydown', handleEscape);
	});

	onDestroy(() => {
		// Restore body scroll
		document.body.classList.remove('modal-open');
	});

	async function loadTimeline() {
		loading = true;
		error = null;
		try {
			timeline = await api.getPositionTimeline(symbol);
		} catch (e) {
			// Provide user-friendly message for 404
			const anyError = e as any;
			const status = anyError?.status ?? anyError?.response?.status;

			if (status === 404) {
				error = 'Position not found. It may have been closed.';
			} else {
				error = e instanceof Error ? e.message : 'Failed to load position timeline';
			}
		} finally {
			loading = false;
		}
	}

	function handleOverlayClick(e: MouseEvent) {
		if (e.target === e.currentTarget) {
			onClose();
		}
	}

	function formatPrice(price: number): string {
		return `$${price.toFixed(2)}`;
	}

	function calculatePnL(entryPrice: number, currentPrice: number): number {
		return ((currentPrice - entryPrice) / entryPrice) * 100;
	}
</script>

<!-- Modal overlay -->
<div
	class="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
	on:click={handleOverlayClick}
	role="dialog"
	aria-modal="true"
	aria-labelledby="modal-title"
>
	<!-- Modal content -->
	<div
		class="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col"
	>
		<!-- Header -->
		<div class="flex items-center justify-between p-6 border-b border-gray-200">
			<h2 id="modal-title" class="text-2xl font-bold text-gray-900">
				{symbol} Position Timeline
			</h2>
			<button
				on:click={onClose}
				class="text-gray-400 hover:text-gray-600 transition-colors"
				aria-label="Close modal"
			>
				<svg
					class="w-6 h-6"
					fill="none"
					stroke="currentColor"
					viewBox="0 0 24 24"
				>
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						stroke-width="2"
						d="M6 18L18 6M6 6l12 12"
					/>
				</svg>
			</button>
		</div>

		<!-- Body -->
		<div class="flex-1 overflow-y-auto p-6">
			{#if loading}
				<div class="flex items-center justify-center py-12">
					<div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
				</div>
			{:else if error}
				<div class="bg-red-50 border border-red-200 text-red-800 rounded-lg p-4">
					<p class="font-semibold">Error loading timeline</p>
					<p class="text-sm mt-1">{error}</p>
				</div>
			{:else if timeline}
				<!-- Warning banner if database disabled -->
				{#if !timeline.database_enabled}
					<div class="bg-yellow-50 border border-yellow-200 text-yellow-800 rounded-lg p-4 mb-6">
						<p class="font-semibold">⚠️ Database Disabled</p>
						<p class="text-sm mt-1">
							Position management actions are not being tracked. Enable database persistence in
							configuration to see timeline data.
						</p>
					</div>
				{/if}

				<!-- Summary cards -->
				<div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
					<div class="bg-gray-50 rounded-lg p-4">
						<p class="text-sm text-gray-600 mb-1">Entry Price</p>
						<p class="text-xl font-bold text-gray-900">{formatPrice(timeline.entry_price)}</p>
					</div>

					<div class="bg-gray-50 rounded-lg p-4">
						<p class="text-sm text-gray-600 mb-1">Current Price</p>
						<p class="text-xl font-bold text-gray-900">{formatPrice(timeline.current_price)}</p>
					</div>

					{#if timeline}
						{@const pnl = calculatePnL(timeline.entry_price, timeline.current_price)}
						<div class="bg-gray-50 rounded-lg p-4">
							<p class="text-sm text-gray-600 mb-1">Unrealized P&L</p>
							<p
								class="text-xl font-bold {pnl >= 0 ? 'text-green-600' : 'text-red-600'}"
							>
								{pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}%
							</p>
						</div>
					{/if}

					<div class="bg-gray-50 rounded-lg p-4">
						<p class="text-sm text-gray-600 mb-1">Days Held</p>
						<p class="text-xl font-bold text-gray-900">{timeline.days_held}</p>
					</div>
				</div>

				<!-- Timeline -->
				<div>
					<h3 class="text-lg font-semibold text-gray-900 mb-4">
						Management Actions ({timeline.count})
					</h3>
					<TimelineView actions={timeline.actions} />
				</div>
			{/if}
		</div>
	</div>
</div>

<style>
	/* Prevent body scroll when modal is open */
	:global(body.modal-open) {
		overflow: hidden;
	}
</style>
