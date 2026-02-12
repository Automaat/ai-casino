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
		{ id: 'config', label: 'Config', href: '/config' }
	];

	$: currentPath = $page.url.pathname;
	$: activeTab = currentPath === '/' ? 'overview'
		: currentPath.startsWith('/portfolio') ? 'portfolio'
		: currentPath.startsWith('/signals') ? 'signals'
		: currentPath.startsWith('/risk') ? 'risk'
		: currentPath.startsWith('/events') ? 'events'
		: currentPath.startsWith('/workflow') ? 'workflow'
		: currentPath.startsWith('/config') ? 'config'
		: 'overview';

	$: daemonStatus = $health?.daemon_running ? 'Running' : 'Stopped';
	$: statusColor = $health?.daemon_running ? 'text-green-400' : 'text-red-400';
</script>

<div class="min-h-screen bg-slate-900">
	<!-- Header -->
	<header class="bg-slate-800 border-b border-slate-700">
		<div class="container mx-auto px-6 py-4">
			<div class="flex items-center justify-between">
				<div>
					<h1 class="text-2xl font-bold text-slate-100">AI Casino</h1>
					<p class="text-sm text-slate-400">Multi-Agent Trading System</p>
				</div>
				<div class="flex items-center gap-4">
					<div class="text-sm">
						<span class="text-slate-400">Daemon:</span>
						<span class="ml-2 font-medium {statusColor}">{daemonStatus}</span>
					</div>
					{#if $health?.uptime_seconds}
						<div class="text-sm text-slate-400">
							Uptime: {Math.floor($health.uptime_seconds / 3600)}h {Math.floor(($health.uptime_seconds % 3600) / 60)}m
						</div>
					{/if}
				</div>
			</div>
		</div>
	</header>

	<!-- Navigation -->
	<div class="bg-slate-800 border-b border-slate-700">
		<div class="container mx-auto px-6">
			<Tabs {tabs} {activeTab} />
		</div>
	</div>

	<!-- Main Content -->
	<main class="container mx-auto px-6 py-8">
		<slot />
	</main>

	<!-- Footer -->
	<footer class="bg-slate-800 border-t border-slate-700 mt-12">
		<div class="container mx-auto px-6 py-4 text-center text-sm text-slate-400">
			SvelteKit + TypeScript + ECharts + Lightweight Charts
		</div>
	</footer>
</div>
