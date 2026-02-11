<script lang="ts">
	import type { ServiceCheck } from '$lib/types/api';

	export let check: ServiceCheck;

	// Map service names to display
	const serviceNames: Record<string, string> = {
		alpha_vantage: 'Alpha Vantage',
		marketaux: 'Marketaux',
		alpaca: 'Alpaca',
		llm: 'LLM',
		finnhub: 'Finnhub'
	};

	// Status colors
	const statusColors: Record<string, string> = {
		HEALTHY: 'bg-green-100 text-green-800 border-green-300',
		DEGRADED: 'bg-yellow-100 text-yellow-800 border-yellow-300',
		UNHEALTHY: 'bg-red-100 text-red-800 border-red-300',
		SKIPPED: 'bg-gray-100 text-gray-600 border-gray-300'
	};

	// Status icons
	const statusIcons: Record<string, string> = {
		HEALTHY: '✅',
		DEGRADED: '⚠️',
		UNHEALTHY: '❌',
		SKIPPED: '⏸️'
	};

	$: displayName = serviceNames[check.service] || check.service;
	$: colorClass = statusColors[check.status] || statusColors.SKIPPED;
	$: icon = statusIcons[check.status] || '';
</script>

<div
	class="flex items-center gap-2 px-3 py-2 rounded-lg border {colorClass} transition-all"
	title={check.message}
>
	<span class="text-sm">{icon}</span>
	<div class="flex flex-col">
		<span class="text-xs font-medium">{displayName}</span>
		<span class="text-[10px] opacity-75">{check.duration_ms.toFixed(0)}ms</span>
	</div>
</div>
