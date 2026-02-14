/**
 * Supervisor metrics state management
 */

import { writable } from 'svelte/store';
import { api } from '$lib/api/client';
import type * as T from '$lib/types/api';

// Recent metrics store
function createRecentMetricsStore() {
	const { subscribe, set } = writable<T.SupervisorMetricsRecent | null>(null);

	async function fetch(params?: { limit?: number }) {
		try {
			const data = await api.getSupervisorMetricsRecent(params);
			set(data);
			return data;
		} catch (error) {
			console.error('Failed to fetch supervisor recent metrics:', error);
			set(null);
			return null;
		}
	}

	return { subscribe, fetch };
}

// Summary store
function createSummaryStore() {
	const { subscribe, set } = writable<T.SupervisorMetricsSummary | null>(null);

	async function fetch(params?: { hours?: number }) {
		try {
			const data = await api.getSupervisorMetricsSummary(params);
			set(data);
			return data;
		} catch (error) {
			console.error('Failed to fetch supervisor summary:', error);
			set(null);
			return null;
		}
	}

	return { subscribe, fetch };
}

// Worker stats store
function createWorkerStatsStore() {
	const { subscribe, set } = writable<T.SupervisorMetricsWorkers | null>(null);

	async function fetch(params?: { hours?: number }) {
		try {
			const data = await api.getSupervisorMetricsWorkers(params);
			set(data);
			return data;
		} catch (error) {
			console.error('Failed to fetch worker stats:', error);
			set(null);
			return null;
		}
	}

	return { subscribe, fetch };
}

// Errors store
function createErrorsStore() {
	const { subscribe, set } = writable<T.SupervisorMetricsErrors | null>(null);

	async function fetch(params?: { hours?: number }) {
		try {
			const data = await api.getSupervisorMetricsErrors(params);
			set(data);
			return data;
		} catch (error) {
			console.error('Failed to fetch supervisor errors:', error);
			set(null);
			return null;
		}
	}

	return { subscribe, fetch };
}

// Export stores
export const recentMetrics = createRecentMetricsStore();
export const summary = createSummaryStore();
export const workerStats = createWorkerStatsStore();
export const errors = createErrorsStore();

// Fetch all supervisor metrics
export async function fetchAllSupervisorMetrics(hours = 24, limit = 20) {
	await Promise.all([
		summary.fetch({ hours }),
		recentMetrics.fetch({ limit }),
		workerStats.fetch({ hours }),
		errors.fetch({ hours })
	]);
}
