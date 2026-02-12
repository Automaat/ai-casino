<script lang="ts">
	import { onMount } from 'svelte';
	import { executionMetrics, executionMetricDetail } from '$lib/stores/dashboard';
	import Card from '$lib/components/ui/Card.svelte';
	import GanttChart from '$lib/components/workflow/GanttChart.svelte';
	import WaterfallChart from '$lib/components/workflow/WaterfallChart.svelte';
	import AgentBreakdownChart from '$lib/components/workflow/AgentBreakdownChart.svelte';
	import LLMCallsTable from '$lib/components/workflow/LLMCallsTable.svelte';

	let selectedWorkflowId = $state<string>('');
	let loading = $state<boolean>(false);

	async function handleWorkflowSelection(workflowId: string) {
		if (!workflowId) return;
		selectedWorkflowId = workflowId;
		loading = true;
		await executionMetricDetail.fetch(workflowId);
		loading = false;
	}

	function formatTimestamp(timestamp: string): string {
		return new Date(timestamp).toLocaleString();
	}

	function formatDuration(durationSeconds: number): string {
		return durationSeconds.toFixed(2);
	}

	onMount(async () => {
		await executionMetrics.fetch();
	});

	const metricsList = $derived($executionMetrics?.metrics || []);
	const selectedMetrics = $derived($executionMetricDetail);
	const totalTime = $derived(selectedMetrics ? (selectedMetrics.total_latency_ms / 1000).toFixed(2) : '0');
	const llmCalls = $derived(selectedMetrics?.llm_calls.length || 0);
	const totalTokens = $derived(selectedMetrics ? ((selectedMetrics.total_input_tokens + selectedMetrics.total_output_tokens) / 1000).toFixed(1) : '0');
	const totalCost = $derived(selectedMetrics?.total_estimated_cost_usd.toFixed(4) || '0.0000');
</script>

<div class="space-y-6">
	<!-- Workflow Selector -->
	<Card title="Select Workflow Execution">
		<label for="workflow-selector" class="sr-only">Select workflow execution</label>
		<select
			id="workflow-selector"
			class="w-full bg-gray-100 text-black border border-gray-300 rounded px-4 py-2"
			bind:value={selectedWorkflowId}
			onchange={() => handleWorkflowSelection(selectedWorkflowId)}
		>
			<option value="">-- Select a workflow execution --</option>
			{#each metricsList as metric}
				<option value={metric.workflow_id}>
					{metric.symbol} @ {formatTimestamp(metric.timestamp)} ({formatDuration(metric.total_latency_ms / 1000)}s)
				</option>
			{/each}
		</select>
		{#if metricsList.length === 0}
			<p class="text-gray-600 text-sm mt-2">No workflow executions found</p>
		{/if}
	</Card>

	<!-- Metrics Cards -->
	{#if selectedMetrics && !loading}
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
			<Card title="Total Time">
				<p class="text-3xl font-bold text-blue-600">{totalTime}s</p>
			</Card>
			<Card title="LLM Calls">
				<p class="text-3xl font-bold text-green-600">{llmCalls}</p>
			</Card>
			<Card title="Tokens">
				<p class="text-3xl font-bold text-purple-400">{totalTokens}k</p>
			</Card>
			<Card title="Cost">
				<p class="text-3xl font-bold text-yellow-600">${totalCost}</p>
			</Card>
		</div>

		<!-- Visualizations -->
		<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
			<Card title="Pipeline Stages (Gantt)">
				<GanttChart data={selectedMetrics.pipeline_stages} />
			</Card>
			<Card title="Agent Latencies (Waterfall)">
				<WaterfallChart data={selectedMetrics.agent_timings} />
			</Card>
		</div>

		<Card title="Agent Breakdown">
			<AgentBreakdownChart data={selectedMetrics.agent_timings} />
		</Card>

		<Card title="LLM Call Details">
			<LLMCallsTable data={selectedMetrics.llm_calls} />
		</Card>
	{:else if loading}
		<Card>
			<p class="text-gray-600 text-center py-8">Loading metrics...</p>
		</Card>
	{:else}
		<Card>
			<p class="text-gray-600 text-center py-8">Select a workflow execution to view details</p>
		</Card>
	{/if}
</div>
