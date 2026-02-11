/**
 * Dashboard state management with Svelte stores
 */

import { writable, derived, readable } from 'svelte/store';
import { api, APIError } from '$lib/api/client';
import type * as T from '$lib/types/api';

// Auto-refresh interval (5 seconds)
const REFRESH_INTERVAL = 5000;

// Health store
function createHealthStore() {
	const { subscribe, set } = writable<T.HealthResponse | null>(null);

	async function fetch() {
		try {
			const data = await api.getHealth();
			set(data);
		} catch (error) {
			console.error('Failed to fetch health:', error);
			set(null);
		}
	}

	// Auto-refresh
	if (typeof window !== 'undefined') {
		setInterval(fetch, REFRESH_INTERVAL);
		fetch(); // Initial fetch
	}

	return { subscribe, fetch };
}

// State summary store
function createStateSummaryStore() {
	const { subscribe, set } = writable<T.StateSummaryResponse | null>(null);
	let loading = false;

	async function fetch() {
		if (loading) return;
		loading = true;
		try {
			const data = await api.getStateSummary();
			set(data);
		} catch (error) {
			console.error('Failed to fetch state summary:', error);
			set(null);
		} finally {
			loading = false;
		}
	}

	// Auto-refresh
	if (typeof window !== 'undefined') {
		setInterval(fetch, REFRESH_INTERVAL);
		fetch();
	}

	return { subscribe, fetch };
}

// Analyses store
function createAnalysesStore() {
	const { subscribe, set } = writable<T.AnalysisRecordResponse[]>([]);

	async function fetch(params?: { limit?: number; symbol?: string }) {
		try {
			const data = await api.getAnalyses(params);
			set(data.analyses);
			return data.analyses;
		} catch (error) {
			console.error('Failed to fetch analyses:', error);
			return [];
		}
	}

	return { subscribe, fetch };
}

// Positions store
function createPositionsStore() {
	const { subscribe, set } = writable<T.PositionsResponse | null>(null);

	async function fetch() {
		try {
			const data = await api.getPositions();
			set(data);
		} catch (error) {
			console.error('Failed to fetch positions:', error);
			set(null);
		}
	}

	// Auto-refresh
	if (typeof window !== 'undefined') {
		setInterval(fetch, REFRESH_INTERVAL);
		fetch();
	}

	return { subscribe, fetch };
}

// Risk store
function createRiskStore() {
	const { subscribe, set } = writable<T.RiskReportResponse | null>(null);

	async function fetch() {
		try {
			const data = await api.getRisk();
			set(data);
		} catch (error) {
			console.error('Failed to fetch risk:', error);
			set(null);
		}
	}

	return { subscribe, fetch };
}

// Correlation store
function createCorrelationStore() {
	const { subscribe, set } = writable<T.CorrelationMatrixResponse | null>(null);

	async function fetch() {
		try {
			const data = await api.getCorrelation();
			set(data);
		} catch (error) {
			console.error('Failed to fetch correlation:', error);
			set(null);
		}
	}

	return { subscribe, fetch };
}

// Export stores
export const health = createHealthStore();
export const stateSummary = createStateSummaryStore();
export const analyses = createAnalysesStore();
export const positions = createPositionsStore();
export const risk = createRiskStore();
export const correlation = createCorrelationStore();

// Derived stores
export const isDaemonRunning = derived(health, ($health) => $health?.daemon_running ?? false);
export const hasPositions = derived(positions, ($positions) => ($positions?.positions?.length ?? 0) > 0);
