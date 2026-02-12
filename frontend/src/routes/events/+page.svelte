<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import Card from '$lib/components/ui/Card.svelte';
	import { events, marketEvents, degradationHistory, analyses, risk, degradation } from '$lib/stores/dashboard';
	import { formatDate } from '$lib/utils/format';
	import type { SystemEvent, MarketEvent } from '$lib/types/api';
	import * as echarts from 'echarts';
	import type { ECharts, EChartsOption } from 'echarts';

	// Constants
	const CONSECUTIVE_SIGNALS_THRESHOLD = 5;
	const HIGH_DRAWDOWN_THRESHOLD = 0.10;

	// Filters
	let selectedCategories = $state<string[]>(['ANALYSIS', 'NEWS', 'SOCIAL', 'ANOMALY', 'ERROR']);
	let startDate = $state(new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]);
	let endDate = $state(new Date().toISOString().split('T')[0]);

	// Combined events
	type CombinedEvent = {
		timestamp: string;
		event_type: string;
		category: string;
		source: 'system' | 'market';
		details: string;
		severity: 'ERROR' | 'INFO' | 'TRADE' | 'SYSTEM';
		color: string;
		icon: string;
	};

	let allEvents = $state<CombinedEvent[]>([]);
	let filteredEvents = $state<CombinedEvent[]>([]);
	let availableCategories = $state<string[]>([]);

	// Degradation chart
	let degradationChartContainer = $state<HTMLElement | null>(null as unknown as HTMLElement);
	let degradationChart: ECharts | null = null;

	// Warnings
	interface Warning {
		severity: 'danger' | 'warning';
		icon: string;
		message: string;
		details: string;
	}
	let warnings = $state<Warning[]>([]);

	$effect(() => {
		// Combine system and market events
		allEvents = [];

		if ($events?.events) {
			for (const event of $events.events) {
				allEvents.push(processSystemEvent(event));
			}
		}

		if ($marketEvents?.events) {
			for (const event of $marketEvents.events) {
				allEvents.push(processMarketEvent(event));
			}
		}

		// Sort by timestamp descending
		allEvents.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

		// Extract categories
		availableCategories = [...new Set(allEvents.map(e => e.category))].sort();

		// Apply filters
		filteredEvents = applyFilters(allEvents, selectedCategories, startDate, endDate);
	});

	// Generate warnings
	$effect(() => {
		warnings = [];

		// Check degradation
		if ($degradation) {
			if ($degradation.tier === 'HALTED') {
				warnings.push({
					severity: 'danger',
					icon: '🔴',
					message: `HALTED: ${$degradation.halt_reason}`,
					details: `Unavailable: ${$degradation.unavailable_services.join(', ')}`
				});
			} else if ($degradation.tier === 'DEGRADED' || $degradation.tier === 'MINIMAL') {
				warnings.push({
					severity: 'warning',
					icon: '🟡',
					message: `${$degradation.tier} mode active`,
					details: `Confidence adjustment: ${($degradation.confidence_adjustment * 100).toFixed(0)}%`
				});
			}
		}

		// Check consecutive signals
		if ($analyses && $analyses.length >= CONSECUTIVE_SIGNALS_THRESHOLD) {
			const recentSignals = $analyses.slice(0, CONSECUTIVE_SIGNALS_THRESHOLD).map(a => a.signal);
			if (recentSignals.every(s => s === 'SELL' || s === 'HOLD')) {
				warnings.push({
					severity: 'warning',
					icon: '🟡',
					message: `Consecutive non-BUY signals: ${recentSignals.length}`,
					details: 'Portfolio may be risk-averse or markets bearish'
				});
			}
		}

		// Check high drawdown
		if ($risk && Math.abs($risk.max_drawdown) > HIGH_DRAWDOWN_THRESHOLD) {
			warnings.push({
				severity: 'danger',
				icon: '🔴',
				message: `High drawdown: ${(Math.abs($risk.max_drawdown) * 100).toFixed(1)}%`,
				details: `Risk status: ${$risk.risk_status}`
			});
		}
	});

	function processSystemEvent(event: SystemEvent): CombinedEvent {
		const category = categorizeEvent(event.event_type);
		const severity = getSeverity(category, event.event_type);
		const { color, icon } = getSeverityStyle(severity);

		return {
			timestamp: event.timestamp,
			event_type: event.event_type,
			category,
			source: 'system',
			details: JSON.stringify(event.data),
			severity,
			color,
			icon
		};
	}

	function processMarketEvent(event: MarketEvent): CombinedEvent {
		const eventType = event.event?.event_type || 'unknown';
		const category = categorizeEvent(eventType);
		const severity = getSeverity(category, eventType);
		const { color, icon } = getSeverityStyle(severity);

		return {
			timestamp: event.signal_timestamp,
			event_type: eventType,
			category,
			source: 'market',
			details: event.summary || event.event?.summary || '-',
			severity,
			color,
			icon
		};
	}

	function categorizeEvent(eventType: string): string {
		const type = eventType.toUpperCase();

		if (type.includes('NEWS')) return 'NEWS';
		if (type.includes('SOCIAL')) return 'SOCIAL';
		if (type.includes('ANOMALY')) return 'ANOMALY';
		if (type.includes('FILING')) return 'FILING';
		if (type.includes('ERROR') || type.includes('DEGRADATION')) return 'ERROR';
		if (type.includes('ANALYSIS') || type.includes('TRADE')) return 'ANALYSIS';

		return 'SYSTEM';
	}

	function getSeverity(category: string, eventType: string): 'ERROR' | 'INFO' | 'TRADE' | 'SYSTEM' {
		if (category === 'ERROR' || eventType.includes('ERROR')) return 'ERROR';
		if (category === 'NEWS' || category === 'SOCIAL' || category === 'ANOMALY') return 'INFO';
		if (category === 'ANALYSIS' && eventType.includes('TRADE')) return 'TRADE';
		return 'SYSTEM';
	}

	function getSeverityStyle(severity: 'ERROR' | 'INFO' | 'TRADE' | 'SYSTEM'): { color: string; icon: string } {
		switch (severity) {
			case 'ERROR':
				return { color: '#ef4444', icon: '🔴' };
			case 'INFO':
				return { color: '#3b82f6', icon: '🔵' };
			case 'TRADE':
				return { color: '#22c55e', icon: '🟢' };
			case 'SYSTEM':
			default:
				return { color: '#6b7280', icon: '⚪' };
		}
	}

	function applyFilters(
		events: CombinedEvent[],
		categories: string[],
		start: string,
		end: string
	): CombinedEvent[] {
		let filtered = events;

		// Category filter
		if (categories.length > 0) {
			filtered = filtered.filter(e => categories.includes(e.category));
		}

		// Date filter
		if (start && end) {
			const startDate = new Date(start + 'T00:00:00');
			const endDate = new Date(end + 'T23:59:59');
			endDate.setMilliseconds(999);

			filtered = filtered.filter(e => {
				const eventDate = new Date(e.timestamp);
				return eventDate >= startDate && eventDate <= endDate;
			});
		}

		return filtered;
	}

	function truncate(str: string, maxLength: number = 150): string {
		if (str.length <= maxLength) return str;
		return str.substring(0, maxLength - 3) + '...';
	}

	// Load data
	function loadData() {
		events.fetch({ limit: 100 });
		marketEvents.fetch({ limit: 100 });
		degradationHistory.fetch({ limit: 50 });
		analyses.fetch({ limit: 100 });
		risk.fetch();
	}

	onMount(() => {
		loadData();

		return () => {
			degradationChart?.dispose();
			degradationChart = null;
		};
	});

	// Refetch when route changes (handles tab switching)
	$effect(() => {
		if ($page.url.pathname === '/events') {
			loadData();
		}
	});

	// Initialize degradation chart reactively when container becomes available
	$effect(() => {
		if (degradationChartContainer && !degradationChart) {
			degradationChart = echarts.init(degradationChartContainer, 'dark');
			if ($degradationHistory) {
				updateDegradationChart();
			}
		}
	});

	$effect(() => {
		if (degradationChart && $degradationHistory) {
			updateDegradationChart();
		}
	});

	function updateDegradationChart() {
		if (!degradationChart || !$degradationHistory?.records || $degradationHistory.records.length === 0) {
			return;
		}

		const tierOrder = ['FULL', 'DEGRADED', 'MINIMAL', 'HALTED'];
		const tierColors: Record<string, string> = {
			FULL: '#22c55e',
			DEGRADED: '#fbbf24',
			MINIMAL: '#f97316',
			HALTED: '#ef4444'
		};

		const records = $degradationHistory.records;
		const timestamps = records.map(r => new Date(r.timestamp).toLocaleString());
		const tiers = records.map(r => r.tier);
		const services = records.map(r => r.unavailable_services.join(', ') || 'All healthy');

		const option: EChartsOption = {
			backgroundColor: 'transparent',
			title: {
				text: 'API Degradation Timeline',
				textStyle: { color: '#e2e8f0', fontSize: 14 }
			},
			tooltip: {
				trigger: 'axis',
				backgroundColor: '#1e293b',
				borderColor: '#334155',
				textStyle: { color: '#e2e8f0' },
				formatter: (params: any) => {
					const p = Array.isArray(params) ? params[0] : params;
					return `<b>${p.value}</b><br/>${p.name}<br/>${services[p.dataIndex]}`;
				}
			},
			grid: {
				left: '3%',
				right: '4%',
				bottom: '3%',
				containLabel: true
			},
			xAxis: {
				type: 'category',
				data: timestamps,
				axisLine: { lineStyle: { color: '#334155' } },
				axisLabel: { color: '#94a3b8', rotate: 45 }
			},
			yAxis: {
				type: 'category',
				data: tierOrder,
				axisLine: { lineStyle: { color: '#334155' } },
				axisLabel: { color: '#94a3b8' }
			},
			series: [
				{
					data: tiers,
					type: 'line',
					smooth: false,
					lineStyle: { color: '#6b7280', width: 2 },
					itemStyle: {
						color: (params: any) => tierColors[params.value] || '#6b7280'
					},
					symbolSize: 10
				}
			]
		};

		degradationChart.setOption(option);
	}

</script>

<svelte:head>
	<title>Events - AI Casino</title>
</svelte:head>

<div class="space-y-8">
	<!-- Warnings Banner -->
	{#if warnings.length > 0}
		<div class="space-y-2">
			<h2 class="text-xl font-semibold text-slate-100">⚠️ Active Warnings</h2>
			{#each warnings as warning}
				<div class="{warning.severity === 'danger' ? 'bg-red-900/20 border-red-700' : 'bg-yellow-900/20 border-yellow-700'} border rounded-lg p-4">
					<div class="flex items-start gap-3">
						<span class="text-2xl">{warning.icon}</span>
						<div>
							<p class="font-semibold text-slate-100">{warning.message}</p>
							<p class="text-sm text-slate-400 mt-1">{warning.details}</p>
						</div>
					</div>
				</div>
			{/each}
		</div>
	{/if}

	<!-- Degradation Timeline -->
	{#if $degradationHistory?.records && $degradationHistory.records.length > 0}
		<Card title="API Degradation Timeline">
			<div bind:this={degradationChartContainer} style="width: 100%; height: 300px;"></div>
		</Card>
	{/if}

	<!-- Filters -->
	<Card title="Filters">
		<div class="space-y-4">
			<div>
				<!-- svelte-ignore a11y_label_has_associated_control -->
				<label class="block text-sm font-medium text-slate-400 mb-2">
					Event Types
				</label>
				<div class="flex flex-wrap gap-2">
					{#each availableCategories as category}
						<label class="flex items-center gap-2 px-3 py-2 bg-slate-700 rounded-lg cursor-pointer hover:bg-slate-600 transition-colors">
							<input
								type="checkbox"
								bind:group={selectedCategories}
								value={category}
								class="rounded border-slate-500 text-blue-500 focus:ring-blue-500"
							/>
							<span class="text-sm text-slate-200">{category}</span>
						</label>
					{/each}
				</div>
			</div>
			<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
				<div>
					<label for="start-date" class="block text-sm font-medium text-slate-400 mb-2">
						Start Date
					</label>
					<input
						id="start-date"
						type="date"
						bind:value={startDate}
						class="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
					/>
				</div>
				<div>
					<label for="end-date" class="block text-sm font-medium text-slate-400 mb-2">
						End Date
					</label>
					<input
						id="end-date"
						type="date"
						bind:value={endDate}
						class="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
					/>
				</div>
			</div>
		</div>
	</Card>

	<!-- Event Log -->
	<Card title="Event Log ({filteredEvents.length} events)">
		{#if filteredEvents.length > 0}
			<div class="overflow-x-auto">
				<table class="w-full text-sm">
					<thead class="border-b border-slate-700">
						<tr>
							<th class="text-left py-3 px-4 font-medium text-slate-400">Timestamp</th>
							<th class="text-left py-3 px-4 font-medium text-slate-400">Type</th>
							<th class="text-left py-3 px-4 font-medium text-slate-400">Category</th>
							<th class="text-left py-3 px-4 font-medium text-slate-400">Details</th>
						</tr>
					</thead>
					<tbody>
						{#each filteredEvents.slice(0, 100) as event}
							<tr class="border-b border-slate-800 hover:bg-slate-800/50">
								<td class="py-3 px-4 font-mono text-xs text-slate-300">
									{formatDate(event.timestamp)}
								</td>
								<td class="py-3 px-4">
									<div class="flex items-center gap-2">
										<span>{event.icon}</span>
										<span class="font-medium" style="color: {event.color}">
											{event.event_type.replace(/_/g, ' ')}
										</span>
									</div>
								</td>
								<td class="py-3 px-4 text-xs text-slate-400">
									{event.category}
								</td>
								<td class="py-3 px-4 text-xs text-slate-300">
									{truncate(event.details)}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else}
			<div class="text-center py-12 text-slate-400">
				No events match the selected filters.
			</div>
		{/if}
	</Card>
</div>
