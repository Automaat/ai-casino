<script lang="ts">
	import '../app.css';
	import { page } from '$app/stores';
	import Tabs from '$lib/components/ui/Tabs.svelte';
	import { health } from '$lib/stores/dashboard';

	let { children } = $props();

	const tabs = [
		{ id: 'overview', label: 'Overview', href: '/' },
		{ id: 'portfolio', label: 'Portfolio', href: '/portfolio' },
		{ id: 'rebalancing', label: 'Rebalancing', href: '/rebalancing' },
		{ id: 'journal', label: 'Journal', href: '/journal' },
		{ id: 'signals', label: 'Signals', href: '/signals' },
		{ id: 'discovery', label: 'Discovery', href: '/discovery' },
		{ id: 'risk', label: 'Risk', href: '/risk' },
		{ id: 'validation', label: 'Validation', href: '/validation' },
		{ id: 'events', label: 'Events', href: '/events' },
		{ id: 'queue', label: 'Queue', href: '/queue' },
		{ id: 'workflow', label: 'Workflow', href: '/workflow' },
		{ id: 'execution', label: 'Execution', href: '/execution' },
		{ id: 'supervisor', label: 'Supervisor', href: '/supervisor' },
		{ id: 'cost-analytics', label: 'Costs', href: '/analytics/cost' },
		{ id: 'worker-analytics', label: 'Workers', href: '/analytics/workers' },
		{ id: 'signal-analytics', label: 'Signal Accuracy', href: '/analytics/signals' },
		{ id: 'health', label: 'Health', href: '/health' },
		{ id: 'config', label: 'Config', href: '/config' }
	];

	const currentPath = $derived($page.url.pathname);
	const activeTab = $derived(
		currentPath === '/' ? 'overview'
		: currentPath.startsWith('/rebalancing') ? 'rebalancing'
		: currentPath.startsWith('/portfolio') ? 'portfolio'
		: currentPath.startsWith('/journal') ? 'journal'
		: currentPath.startsWith('/signals') ? 'signals'
		: currentPath.startsWith('/discovery') ? 'discovery'
		: currentPath.startsWith('/risk') ? 'risk'
		: currentPath.startsWith('/validation') ? 'validation'
		: currentPath.startsWith('/events') ? 'events'
		: currentPath.startsWith('/queue') ? 'queue'
		: currentPath.startsWith('/workflow') ? 'workflow'
		: currentPath.startsWith('/execution') ? 'execution'
		: currentPath.startsWith('/supervisor') ? 'supervisor'
		: currentPath.startsWith('/analytics/cost') ? 'cost-analytics'
		: currentPath.startsWith('/analytics/workers') ? 'worker-analytics'
		: currentPath.startsWith('/analytics/signals') ? 'signal-analytics'
		: currentPath.startsWith('/health') ? 'health'
		: currentPath.startsWith('/config') ? 'config'
		: 'overview'
	);

	const daemonStatus = $derived($health?.daemon_running ? 'Running' : 'Stopped');
	const statusColor = $derived($health?.daemon_running ? 'text-green-600' : 'text-red-600');
</script>

<div class="min-h-screen bg-white">
	<!-- Header -->
	<header class="bg-gray-50 border-b border-gray-300">
		<div class="container mx-auto px-6 py-4">
			<div class="flex items-center justify-between">
				<div>
					<h1 class="text-2xl font-bold text-black">AI Casino</h1>
					<p class="text-sm text-gray-600">Multi-Agent Trading System</p>
				</div>
				<div class="flex items-center gap-4">
					<div class="text-sm">
						<span class="text-gray-600">Daemon:</span>
						<span class="ml-2 font-medium {statusColor}">{daemonStatus}</span>
					</div>
					{#if $health?.uptime_seconds}
						<div class="text-sm text-gray-600">
							Uptime: {Math.floor($health.uptime_seconds / 3600)}h {Math.floor(($health.uptime_seconds % 3600) / 60)}m
						</div>
					{/if}
				</div>
			</div>
		</div>
	</header>

	<!-- Navigation -->
	<div class="bg-gray-50 border-b border-gray-300">
		<div class="container mx-auto px-6">
			<Tabs {tabs} {activeTab} />
		</div>
	</div>

	<!-- Main Content -->
	<main class="container mx-auto px-6 py-8">
		{@render children()}
	</main>

	<!-- Footer -->
	<footer class="bg-gray-50 border-t border-gray-300 mt-12">
		<div class="container mx-auto px-6 py-4 text-center text-sm text-gray-600">
			SvelteKit + TypeScript + ECharts + Lightweight Charts
		</div>
	</footer>
</div>
