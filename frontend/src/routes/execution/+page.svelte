<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import ForceDirectedGraph from '$lib/components/execution/ForceDirectedGraph.svelte';
	import { execution, selectedGraph, activeWorkflowIds, hasActiveWorkflows } from '$lib/stores/execution';
	import { api } from '$lib/api/client';
	import type { ExecutionGraph } from '$lib/types/api';

	let loading = $state(true);
	let error = $state<string | null>(null);
	let viewMode = $state<'active' | 'history'>('active');
	let historyGraphs = $state<ExecutionGraph[]>([]);
	let selectedHistoryIndex = $state(0);

	onMount(async () => {
		// Connect to WebSocket
		execution.connect();

		// Fetch initial active graphs and history
		try {
			await execution.fetchActive();
			await fetchHistory();
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load execution graphs';
		} finally {
			loading = false;
		}
	});

	onDestroy(() => {
		// Disconnect WebSocket on page unmount
		execution.disconnect();
	});

	async function fetchHistory() {
		try {
			const response = await api.getExecutionGraphHistory({ limit: 20 });
			historyGraphs = response.graphs;

			// Auto-switch to history if no active workflows
			if (!$hasActiveWorkflows && historyGraphs.length > 0) {
				viewMode = 'history';
			}
		} catch (err) {
			console.error('Failed to fetch history:', err);
		}
	}

	function formatTimestamp(timestamp: string): string {
		const date = new Date(timestamp);
		return date.toLocaleString();
	}

	function getNodeCounts(graph: ExecutionGraph | typeof $selectedGraph | null) {
		if (!graph) return { total: 0, running: 0, completed: 0, failed: 0 };

		const nodes = Object.values(graph.nodes);
		return {
			total: nodes.length,
			running: nodes.filter(n => n.status === 'RUNNING').length,
			completed: nodes.filter(n => n.status === 'COMPLETED').length,
			failed: nodes.filter(n => n.status === 'FAILED').length
		};
	}

	const displayGraph = $derived(
		viewMode === 'active' ? $selectedGraph : historyGraphs[selectedHistoryIndex]
	);
</script>

<div class="space-y-6">
	<div class="flex items-center justify-between">
		<h2 class="text-2xl font-bold text-black">Execution Visualization</h2>

		<!-- View Mode Toggle -->
		{#if !loading && !error}
			<div class="flex gap-2 border border-gray-300 rounded-md p-1">
				<button
					class="px-4 py-1.5 text-sm font-medium rounded transition-colors {viewMode === 'active'
						? 'bg-blue-600 text-white'
						: 'text-gray-700 hover:bg-gray-100'}"
					onclick={() => viewMode = 'active'}
				>
					Active {$hasActiveWorkflows ? `(${$activeWorkflowIds.length})` : '(0)'}
				</button>
				<button
					class="px-4 py-1.5 text-sm font-medium rounded transition-colors {viewMode === 'history'
						? 'bg-blue-600 text-white'
						: 'text-gray-700 hover:bg-gray-100'}"
					onclick={() => viewMode = 'history'}
				>
					History ({historyGraphs.length})
				</button>
			</div>
		{/if}
	</div>

	{#if loading}
		<div class="flex items-center justify-center h-96">
			<div class="text-gray-600">Loading execution graphs...</div>
		</div>
	{:else if error}
		<div class="bg-red-50 border border-red-200 rounded-lg p-4">
			<div class="font-medium text-red-800">Error</div>
			<div class="text-red-600 text-sm">{error}</div>
		</div>
	{:else if viewMode === 'active' && !$hasActiveWorkflows}
		<div class="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center">
			<div class="text-gray-800 font-medium mb-2">No Active Workflows</div>
			<div class="text-gray-600 text-sm mb-4">
				Execution graphs will appear here in real-time when workflows are running.
			</div>
			<div class="text-xs text-gray-500">
				WebSocket: {$execution.wsConnected ? '🟢 Connected' : '🔴 Disconnected'}
			</div>
			{#if historyGraphs.length > 0}
				<button
					class="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm"
					onclick={() => viewMode = 'history'}
				>
					View History ({historyGraphs.length} workflows)
				</button>
			{/if}
		</div>
	{:else if viewMode === 'history' && historyGraphs.length === 0}
		<div class="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center">
			<div class="text-gray-800 font-medium mb-2">No Historical Workflows</div>
			<div class="text-gray-600 text-sm">
				Completed workflow executions will appear here.
			</div>
		</div>
	{:else}
		<!-- Workflow Selector -->
		{#if viewMode === 'active' && $activeWorkflowIds.length > 1}
			<div class="flex items-center gap-4">
				<label for="workflow-select" class="text-sm font-medium text-gray-700">
					Select Workflow:
				</label>
				<select
					id="workflow-select"
					class="px-3 py-2 border border-gray-300 rounded-md bg-white text-gray-900 text-sm"
					value={$execution.selectedWorkflowId}
					onchange={(e) => execution.selectWorkflow((e.target as HTMLSelectElement).value)}
				>
					{#each $activeWorkflowIds as workflowId}
						{@const graph = $execution.activeGraphs[workflowId]}
						<option value={workflowId}>
							{graph.symbol || 'Unknown'} - {workflowId.substring(0, 8)}
						</option>
					{/each}
				</select>
			</div>
		{:else if viewMode === 'history' && historyGraphs.length > 1}
			<div class="flex items-center gap-4">
				<label for="history-select" class="text-sm font-medium text-gray-700">
					Select Workflow:
				</label>
				<select
					id="history-select"
					class="px-3 py-2 border border-gray-300 rounded-md bg-white text-gray-900 text-sm"
					bind:value={selectedHistoryIndex}
				>
					{#each historyGraphs as graph, index}
						<option value={index}>
							{graph.symbol || 'Unknown'} - {formatTimestamp(graph.created_at)}
						</option>
					{/each}
				</select>
			</div>
		{/if}

		<!-- Graph Metadata Header -->
		{#if displayGraph}
			{@const counts = getNodeCounts(displayGraph)}
			<div class="bg-white border border-gray-200 rounded-lg p-4">
				<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
					<div>
						<div class="text-xs text-gray-600">Symbol</div>
						<div class="font-medium text-gray-900">{displayGraph.symbol || 'N/A'}</div>
					</div>
					<div>
						<div class="text-xs text-gray-600">Workflow ID</div>
						<div class="font-medium text-gray-900 font-mono text-sm">
							{displayGraph.workflow_id.toString().substring(0, 12)}...
						</div>
					</div>
					<div>
						<div class="text-xs text-gray-600">Started</div>
						<div class="font-medium text-gray-900 text-sm">
							{formatTimestamp(displayGraph.created_at)}
						</div>
					</div>
					<div>
						<div class="text-xs text-gray-600">Last Updated</div>
						<div class="font-medium text-gray-900 text-sm">
							{formatTimestamp(displayGraph.updated_at)}
						</div>
					</div>
				</div>

				<!-- Node Status Summary -->
				<div class="mt-4 flex gap-6 text-sm">
					<div class="flex items-center gap-2">
						<div class="w-3 h-3 rounded-full bg-gray-400"></div>
						<span class="text-gray-700">Total: {counts.total}</span>
					</div>
					<div class="flex items-center gap-2">
						<div class="w-3 h-3 rounded-full bg-blue-500"></div>
						<span class="text-gray-700">Running: {counts.running}</span>
					</div>
					<div class="flex items-center gap-2">
						<div class="w-3 h-3 rounded-full bg-green-500"></div>
						<span class="text-gray-700">Completed: {counts.completed}</span>
					</div>
					<div class="flex items-center gap-2">
						<div class="w-3 h-3 rounded-full bg-red-500"></div>
						<span class="text-gray-700">Failed: {counts.failed}</span>
					</div>
				</div>
			</div>

			<!-- Graph Visualization -->
			<div class="bg-white rounded-lg p-4 border border-gray-200">
				<div class="mb-4 flex items-center justify-between">
					<h3 class="text-lg font-semibold text-gray-900">
						Execution Graph {viewMode === 'history' ? '(Historical)' : '(Live)'}
					</h3>
					<div class="text-xs text-gray-500">
						{#if viewMode === 'active'}
							WebSocket: {$execution.wsConnected ? '🟢 Connected' : '🔴 Disconnected'}
						{:else}
							{counts.total} nodes • {counts.completed} completed • {counts.failed} failed
						{/if}
					</div>
				</div>
				<ForceDirectedGraph graph={displayGraph} />
				<div class="mt-4 text-xs text-gray-500 text-center">
					💡 Drag nodes to reposition • Scroll to zoom • Click and drag background to pan
				</div>
			</div>
		{/if}
	{/if}
</div>
