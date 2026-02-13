<script lang="ts">
	import '../app.css';
	import { page } from '$app/stores';
	import Tabs from '$lib/components/ui/Tabs.svelte';
	import { health } from '$lib/stores/dashboard';

	const tabs = [
		{ id: 'overview', label: 'Overview', href: '/' },
		{ id: 'portfolio', label: 'Portfolio', href: '/portfolio' },
		{ id: 'signals', label: 'Signals', href: '/signals' },
		{ id: 'risk', label: 'Risk', href: '/risk' },
		{ id: 'events', label: 'Events', href: '/events' },
		{ id: 'workflow', label: 'Workflow', href: '/workflow' },
		{ id: 'execution', label: 'Execution', href: '/execution' },
		{ id: 'config', label: 'Config', href: '/config' }
	];

	const currentPath = $derived($page.url.pathname);
	const activeTab = $derived(
		currentPath === '/' ? 'overview'
		: currentPath.startsWith('/portfolio') ? 'portfolio'
		: currentPath.startsWith('/signals') ? 'signals'
		: currentPath.startsWith('/risk') ? 'risk'
		: currentPath.startsWith('/events') ? 'events'
		: currentPath.startsWith('/workflow') ? 'workflow'
		: currentPath.startsWith('/execution') ? 'execution'
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
		<slot />
	</main>

	<!-- Footer -->
	<footer class="bg-gray-50 border-t border-gray-300 mt-12">
		<div class="container mx-auto px-6 py-4 text-center text-sm text-gray-600">
			SvelteKit + TypeScript + ECharts + Lightweight Charts
		</div>
	</footer>
</div>
