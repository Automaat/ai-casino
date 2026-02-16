<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import MetricCard from '$lib/components/ui/MetricCard.svelte';
	import Card from '$lib/components/ui/Card.svelte';
	import BarChart from '$lib/components/charts/BarChart.svelte';
	import PieChart from '$lib/components/charts/PieChart.svelte';
	import DataTable from '$lib/components/ui/DataTable.svelte';
	import { screening } from '$lib/stores/screening';

	let refreshInterval: number | null = null;
	let ws: WebSocket | null = null;
	let wsConnected = $state(false);
	let reconnectTimeout: number | null = null;
	let reconnectAttempts = 0;
	const MAX_RECONNECT_ATTEMPTS = 5;
	const RECONNECT_DELAY = 3000;

	let screeningState = $derived($screening);
	let history = $derived(screeningState.history);
	let insights = $derived(screeningState.insights);
	let loading = $derived(screeningState.loading);
	let error = $derived(screeningState.error);

	// Latest screening record
	let latestRecord = $derived.by(() => {
		if (!history?.records.length) return null;
		return history.records[0];
	});

	// Criteria filter for history table
	let criteriaFilter: string | null = $state("");

	// Filtered history records
	let filteredHistory = $derived.by(() => {
		if (!history?.records) return [];
		if (!criteriaFilter) return history.records;
		return history.records.filter((r) => r.criteria === criteriaFilter);
	});

	// Criteria breakdown chart data
	let criteriaChartData = $derived.by(() => {
		if (!insights?.criteria_breakdown) return null;
		const entries = Object.entries(insights.criteria_breakdown);
		if (entries.length === 0) return null;
		return entries.map(([criteria, count]) => ({
			label: criteria,
			value: count,
			color: getCriteriaColor(criteria)
		}));
	});

	// Sector distribution chart data
	let sectorChartData = $derived.by(() => {
		if (!insights?.sector_distribution) return null;
		const entries = Object.entries(insights.sector_distribution);
		if (entries.length === 0) return null;
		return entries.map(([sector, count]) => ({
			label: sector,
			value: count
		}));
	});

	// Signal breakdown chart data
	let signalChartData = $derived.by(() => {
		if (!insights?.top_signals) return null;
		const entries = Object.entries(insights.top_signals);
		if (entries.length === 0) return null;
		return entries.map(([signal, count]) => ({
			label: signal,
			value: count,
			color: signal === 'BUY' ? '#10b981' : signal === 'SELL' ? '#ef4444' : '#6b7280'
		}));
	});

	// Define table row types
	type CandidateRow = {
		symbol: string;
		name: string;
		sector: string;
		score: string;
		signal: string;
		rsi: string;
		macd: string;
		_raw: import('$lib/types/api').ScreeningCandidate;
	};

	type HistoryRow = {
		screened_at: string;
		criteria: string;
		universe: string;
		candidate_count: number;
		top_symbols: string;
		_raw: import('$lib/types/api').ScreeningRecord;
	};

	// Latest candidates table columns
	const candidateColumns = [
		{ key: 'symbol' as keyof CandidateRow, label: 'Symbol' },
		{ key: 'name' as keyof CandidateRow, label: 'Name' },
		{ key: 'sector' as keyof CandidateRow, label: 'Sector' },
		{ key: 'score' as keyof CandidateRow, label: 'Score' },
		{ key: 'signal' as keyof CandidateRow, label: 'Signal' },
		{ key: 'rsi' as keyof CandidateRow, label: 'RSI' },
		{ key: 'macd' as keyof CandidateRow, label: 'MACD' }
	];

	// History table columns
	const historyColumns = [
		{ key: 'screened_at' as keyof HistoryRow, label: 'Date' },
		{ key: 'criteria' as keyof HistoryRow, label: 'Criteria' },
		{ key: 'universe' as keyof HistoryRow, label: 'Universe' },
		{ key: 'candidate_count' as keyof HistoryRow, label: 'Candidates' },
		{ key: 'top_symbols' as keyof HistoryRow, label: 'Top 5 Symbols' }
	];

	// Transform candidates for table
	let candidatesTableData = $derived.by((): CandidateRow[] => {
		if (!latestRecord?.candidates) return [];
		return latestRecord.candidates.map((c) => ({
			symbol: c.symbol,
			name: c.name,
			sector: c.sector,
			score: c.score.toFixed(2),
			signal: c.signal,
			rsi: c.metrics.rsi?.toFixed(2) || 'N/A',
			macd: c.metrics.macd_histogram?.toFixed(4) || 'N/A',
			_raw: c
		}));
	});

	// Transform history for table
	let historyTableData = $derived.by((): HistoryRow[] => {
		return filteredHistory.map((r) => ({
			screened_at: new Date(r.screened_at).toLocaleString(),
			criteria: r.criteria,
			universe: r.universe,
			candidate_count: r.candidate_count,
			top_symbols: r.top_symbols.slice(0, 5).join(', '),
			_raw: r
		}));
	});

	function getCriteriaColor(criteria: string): string {
		const colors: Record<string, string> = {
			momentum: '#3b82f6',
			value: '#10b981',
			breakout: '#f59e0b',
			growth: '#8b5cf6'
		};
		return colors[criteria.toLowerCase()] || '#6b7280';
	}

	function formatDate(dateStr: string | null): string {
		if (!dateStr) return 'Never';
		return new Date(dateStr).toLocaleString();
	}

	const WS_URL =
		import.meta.env.VITE_WS_URL ||
		(typeof window !== 'undefined'
			? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`
			: 'ws://localhost:8484');

	function connectWebSocket() {
		if (ws?.readyState === WebSocket.OPEN) return;

		try {
			ws = new WebSocket(`${WS_URL}/ws`);

			ws.onopen = () => {
				wsConnected = true;
				reconnectAttempts = 0;
				console.log('[Screening] WebSocket connected');
			};

			ws.onmessage = (event) => {
				try {
					const message = JSON.parse(event.data);
					if (message.event_type === 'CYCLE_COMPLETE') {
						console.log('[Screening] Cycle complete, refreshing data...');
						screening.fetchAll();
					}
				} catch (err) {
					console.error('[Screening] Failed to parse WebSocket message:', err);
				}
			};

			ws.onerror = (error) => {
				console.error('[Screening] WebSocket error:', error);
				wsConnected = false;
			};

			ws.onclose = () => {
				wsConnected = false;
				console.log('[Screening] WebSocket disconnected');

				if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
					reconnectAttempts++;
					console.log(
						`[Screening] Reconnecting in ${RECONNECT_DELAY}ms (attempt ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...`
					);
					reconnectTimeout = window.setTimeout(connectWebSocket, RECONNECT_DELAY);
				}
			};
		} catch (err) {
			console.error('[Screening] Failed to connect WebSocket:', err);
			wsConnected = false;
		}
	}

	function disconnectWebSocket() {
		if (reconnectTimeout) {
			clearTimeout(reconnectTimeout);
			reconnectTimeout = null;
		}
		if (ws) {
			ws.close();
			ws = null;
		}
		wsConnected = false;
	}

	onMount(() => {
		screening.fetchAll();
		connectWebSocket();

		refreshInterval = window.setInterval(() => {
			screening.fetchAll();
		}, 30000);

		return () => {
			if (refreshInterval) clearInterval(refreshInterval);
			disconnectWebSocket();
			screening.reset();
		};
	});
</script>

<div class="screening-page">
	<div class="header">
		<h1>Screening Results</h1>
		<div class="header-status">
			<span class="ws-indicator" class:connected={wsConnected} class:disconnected={!wsConnected}>
				{wsConnected ? '● Live' : '○ Disconnected'}
			</span>
		</div>
	</div>

	{#if loading && !history}
		<div class="loading">Loading screening data...</div>
	{:else if error}
		<div class="error">Error: {error}</div>
	{:else if !history?.records.length && !insights}
		<Card title="No Screening Data">
			<p>No screening results available yet. Screening runs after market hours.</p>
		</Card>
	{:else}
		<!-- Metric Cards -->
		<div class="metrics-grid">
			<MetricCard
				title="Total Screenings"
				value={insights?.total_screenings?.toString() || '0'}
				icon="📊"
			/>
			<MetricCard
				title="Latest Candidates"
				value={latestRecord?.candidate_count?.toString() || '0'}
				icon="🎯"
			/>
			<MetricCard
				title="Avg Score"
				value={insights?.avg_score?.toFixed(2) || '0.00'}
				icon="⭐"
			/>
			<MetricCard
				title="Last Screened"
				value={formatDate(insights?.latest_screening_date || null)}
				icon="📅"
			/>
		</div>

		<!-- Charts Row 1: Criteria & Sectors -->
		<div class="charts-row">
			<Card title="Criteria Usage">
				{#if criteriaChartData}
					<BarChart data={criteriaChartData} height={300} yAxisLabel="Count" />
				{:else}
					<div class="empty-state">No criteria data available</div>
				{/if}
			</Card>

			<Card title="Top Sectors">
				{#if sectorChartData}
					<PieChart data={sectorChartData} height={300} />
				{:else}
					<div class="empty-state">No sector data available</div>
				{/if}
			</Card>
		</div>

		<!-- Charts Row 2: Signal Breakdown -->
		<div class="charts-row">
			<Card title="Signal Distribution">
				{#if signalChartData}
					<BarChart data={signalChartData} height={300} yAxisLabel="Count" />
				{:else}
					<div class="empty-state">No signal data available</div>
				{/if}
			</Card>
		</div>

		<!-- Latest Candidates Table -->
		{#if latestRecord}
			<Card title="Latest Screening Candidates">
				<DataTable data={candidatesTableData} columns={candidateColumns} />
			</Card>
		{/if}

		<!-- Screening History Table -->
		<Card title="Screening History">
			<div class="table-controls">
				<label for="criteria-filter">Filter by Criteria:</label>
				<select id="criteria-filter" bind:value={criteriaFilter}>
					<option value="">All</option>
					{#if insights?.criteria_breakdown}
						{#each Object.keys(insights.criteria_breakdown) as criteria}
							<option value={criteria}>{criteria}</option>
						{/each}
					{/if}
				</select>
			</div>
			<DataTable data={historyTableData} columns={historyColumns} />
		</Card>
	{/if}
</div>

<style>
	.screening-page {
		padding: 1.5rem;
		max-width: 1600px;
		margin: 0 auto;
	}

	.header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 1.5rem;
	}

	.header h1 {
		font-size: 1.875rem;
		font-weight: 700;
		color: #1f2937;
	}

	.header-status {
		display: flex;
		gap: 1rem;
		align-items: center;
	}

	.ws-indicator {
		font-size: 0.875rem;
		padding: 0.25rem 0.75rem;
		border-radius: 0.375rem;
		font-weight: 500;
	}

	.ws-indicator.connected {
		background-color: #d1fae5;
		color: #065f46;
	}

	.ws-indicator.disconnected {
		background-color: #fee2e2;
		color: #991b1b;
	}

	.metrics-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
		gap: 1.5rem;
		margin-bottom: 1.5rem;
	}

	.charts-row {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
		gap: 1.5rem;
		margin-bottom: 1.5rem;
	}

	.empty-state {
		display: flex;
		align-items: center;
		justify-content: center;
		height: 300px;
		color: #6b7280;
		font-style: italic;
	}

	.table-controls {
		display: flex;
		gap: 0.75rem;
		align-items: center;
		margin-bottom: 1rem;
	}

	.table-controls label {
		font-weight: 500;
		color: #374151;
	}

	.table-controls select {
		padding: 0.5rem;
		border: 1px solid #d1d5db;
		border-radius: 0.375rem;
		background-color: white;
		color: #374151;
		font-size: 0.875rem;
	}

	.loading,
	.error {
		text-align: center;
		padding: 2rem;
		font-size: 1.125rem;
	}

	.error {
		color: #ef4444;
	}
</style>
