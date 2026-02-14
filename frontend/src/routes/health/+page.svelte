<script lang="ts">
	import MetricCard from '$lib/components/ui/MetricCard.svelte';
	import Card from '$lib/components/ui/Card.svelte';
	import DataTable from '$lib/components/ui/DataTable.svelte';
	import { serviceHealth } from '$lib/stores/dashboard';
	import { uptimeMetrics, degradationTimeline, serviceNames } from '$lib/stores/health';
	import { formatDate } from '$lib/utils/format';

	const currentHealth = $derived($serviceHealth);
	const metrics = $derived($uptimeMetrics);
	const timeline = $derived($degradationTimeline);

	// Service details table
	const serviceColumns = [
		{ key: 'name' as const, label: 'Service', class: 'font-medium' },
		{ key: 'status' as const, label: 'Status' },
		{ key: 'duration_ms' as const, label: 'Duration', format: (v: number) => `${v.toFixed(0)}ms` },
		{ key: 'uptime_percent' as const, label: 'Uptime', format: (v: number) => `${v.toFixed(2)}%` },
		{ key: 'checked_at' as const, label: 'Last Checked', format: (v: string) => formatDate(v) },
		{ key: 'message' as const, label: 'Message' }
	];

	const serviceData = $derived.by(() => {
		if (!metrics?.services) return [];
		return Object.values(metrics.services);
	});

	// Timeline table
	const timelineColumns = [
		{ key: 'timestamp' as const, label: 'Time', format: (v: string) => formatDate(v) },
		{ key: 'service' as const, label: 'Service', class: 'font-medium' },
		{ key: 'from_status' as const, label: 'From' },
		{ key: 'to_status' as const, label: 'To' }
	];

	// Status badge helper
	function getStatusBadgeClass(status: string): string {
		const classes: Record<string, string> = {
			HEALTHY: 'bg-green-100 text-green-800 border-green-300',
			DEGRADED: 'bg-yellow-100 text-yellow-800 border-yellow-300',
			UNHEALTHY: 'bg-red-100 text-red-800 border-red-300',
			SKIPPED: 'bg-gray-100 text-gray-600 border-gray-300'
		};
		return classes[status] || classes.SKIPPED;
	}

	function getStatusIcon(status: string): string {
		const icons: Record<string, string> = {
			HEALTHY: '✅',
			DEGRADED: '⚠️',
			UNHEALTHY: '❌',
			SKIPPED: '⏸️'
		};
		return icons[status] || '❓';
	}
</script>

<svelte:head>
	<title>Service Health - AI Casino</title>
</svelte:head>

<div class="space-y-8">
	<!-- Header -->
	<div class="flex justify-between items-center">
		<div>
			<h1 class="text-3xl font-bold text-gray-900">Service Health Dashboard</h1>
			<p class="text-gray-600 mt-1">Real-time monitoring of external services and circuit breakers</p>
		</div>
		<div class="flex items-center gap-2">
			<div
				class={`w-3 h-3 rounded-full ${
					metrics?.overall_status === 'HEALTHY'
						? 'bg-green-500'
						: metrics?.overall_status === 'UNHEALTHY'
							? 'bg-red-500'
							: 'bg-yellow-500'
				}`}
			></div>
			<span class="text-sm text-gray-600">{metrics?.overall_status || 'Unknown'}</span>
		</div>
	</div>

	<!-- Aggregate Metrics -->
	<div class="grid grid-cols-1 md:grid-cols-4 gap-6">
		<MetricCard
			title="Overall Status"
			value={metrics?.overall_status || 'Unknown'}
			icon={getStatusIcon(metrics?.overall_status || 'SKIPPED')}
		/>
		<MetricCard
			title="Services Online"
			value={`${metrics?.healthy_services || 0}/${metrics?.total_services || 0}`}
			icon="🌐"
		/>
		<MetricCard
			title="Avg Check Duration"
			value={`${Number.isFinite(metrics?.avg_duration) ? metrics?.avg_duration.toFixed(0) : 0}ms`}
			icon="⏱️"
		/>
		<MetricCard
			title="Overall Uptime"
			value={`${Number.isFinite(metrics?.overall_uptime) ? metrics?.overall_uptime.toFixed(2) : 100}%`}
			icon="📈"
		/>
	</div>

	<!-- Service Status Cards Grid -->
	<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
		{#if currentHealth?.service_checks}
			{#each currentHealth.service_checks as check}
				<Card>
					<div class="space-y-3">
						<div class="flex items-start justify-between">
							<div>
								<h3 class="text-lg font-semibold text-gray-900">
									{serviceNames[check.service] || check.service}
								</h3>
								<p class="text-xs text-gray-600 mt-1">
									Last checked: {new Date(check.checked_at).toLocaleTimeString()}
								</p>
							</div>
							<span class="text-2xl">{getStatusIcon(check.status)}</span>
						</div>

						<div class={`px-3 py-2 rounded-lg border text-sm font-medium ${getStatusBadgeClass(check.status)}`}>
							{check.status}
						</div>

						<div class="grid grid-cols-2 gap-2 text-sm">
							<div>
								<span class="text-gray-600">Duration:</span>
								<span class="ml-1 font-medium">{check.duration_ms.toFixed(0)}ms</span>
							</div>
							<div>
								<span class="text-gray-600">Uptime:</span>
								<span class="ml-1 font-medium">
									{metrics?.services[check.service]?.uptime_percent?.toFixed(1) || 100}%
								</span>
							</div>
						</div>

						{#if check.message}
							<p class="text-sm text-gray-600 border-t border-gray-200 pt-2">
								{check.message}
							</p>
						{/if}
					</div>
				</Card>
			{/each}
		{:else}
			<div class="col-span-3 text-center py-12 text-gray-600">No service health data available</div>
		{/if}
	</div>

	<!-- Service Details Table -->
	<Card title="Service Details">
		{#if serviceData.length > 0}
			<DataTable data={serviceData} columns={serviceColumns} />
		{:else}
			<div class="text-center py-12 text-gray-600">No service data available</div>
		{/if}
	</Card>

	<!-- Recovery Timeline -->
	<Card title="Status Change Timeline (Last 50 Events)">
		{#if timeline && timeline.length > 0}
			<DataTable data={timeline} columns={timelineColumns} />
		{:else}
			<div class="text-center py-12 text-gray-600">
				No status changes detected. All services stable.
			</div>
		{/if}
	</Card>

	<!-- System Information -->
	<Card title="Health Check Information">
		<div class="space-y-4 text-sm text-gray-600">
			<div>
				<span class="font-medium text-gray-900">Check Frequency:</span>
				<span class="ml-2">Every 5 seconds</span>
			</div>
			<div>
				<span class="font-medium text-gray-900">History Tracking:</span>
				<span class="ml-2">Last 100 data points (approx. 8 minutes)</span>
			</div>
			<div>
				<span class="font-medium text-gray-900">Services Monitored:</span>
				<span class="ml-2">{metrics?.total_services || 0} external APIs + Circuit Breakers</span>
			</div>
			<div>
				<span class="font-medium text-gray-900">Status Definitions:</span>
				<ul class="ml-6 mt-2 space-y-1 list-disc">
					<li><span class="font-medium">HEALTHY:</span> Service operational, responding within acceptable limits</li>
					<li><span class="font-medium">DEGRADED:</span> Service operational but circuit breaker activated or slow response</li>
					<li><span class="font-medium">UNHEALTHY:</span> Service unreachable or failing health checks</li>
					<li><span class="font-medium">SKIPPED:</span> Service check disabled (no API key configured)</li>
				</ul>
			</div>
		</div>
	</Card>
</div>
