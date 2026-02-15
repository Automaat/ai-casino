/**
 * Worker performance state management
 */

import { writable } from 'svelte/store';
import { api } from '$lib/api/client';
import type * as T from '$lib/types/api';

interface WorkerPerformanceState {
	workers_24h: T.SupervisorMetricsWorkers | null;
	workers_7d: T.SupervisorMetricsWorkers | null;
	workers_30d: T.SupervisorMetricsWorkers | null;
	errors_24h: T.SupervisorMetricsErrors | null;
	errors_7d: T.SupervisorMetricsErrors | null;
	errors_30d: T.SupervisorMetricsErrors | null;
	validation: T.PaperTradingValidation | null;
	loading: boolean;
	error: string | null;
}

function createWorkerPerformanceStore() {
	const { subscribe, set, update } = writable<WorkerPerformanceState>({
		workers_24h: null,
		workers_7d: null,
		workers_30d: null,
		errors_24h: null,
		errors_7d: null,
		errors_30d: null,
		validation: null,
		loading: false,
		error: null
	});

	async function fetchAll() {
		update(state => ({ ...state, loading: true, error: null }));

		try {
			const [workers_24h, workers_7d, workers_30d, errors_24h, errors_7d, errors_30d, validation] =
				await Promise.all([
					api.getSupervisorMetricsWorkers({ hours: 24 }),
					api.getSupervisorMetricsWorkers({ hours: 168 }), // 7 days
					api.getSupervisorMetricsWorkers({ hours: 720 }), // 30 days
					api.getSupervisorMetricsErrors({ hours: 24 }),
					api.getSupervisorMetricsErrors({ hours: 168 }),
					api.getSupervisorMetricsErrors({ hours: 720 }),
					api.getPaperTradingValidation().catch(() => null) // Optional, don't fail if unavailable
				]);

			set({
				workers_24h,
				workers_7d,
				workers_30d,
				errors_24h,
				errors_7d,
				errors_30d,
				validation,
				loading: false,
				error: null
			});
		} catch (error) {
			console.error('Failed to fetch worker performance:', error);
			update(state => ({
				...state,
				loading: false,
				error: error instanceof Error ? error.message : 'Failed to fetch worker performance'
			}));
		}
	}

	function reset() {
		set({
			workers_24h: null,
			workers_7d: null,
			workers_30d: null,
			errors_24h: null,
			errors_7d: null,
			errors_30d: null,
			validation: null,
			loading: false,
			error: null
		});
	}

	return {
		subscribe,
		fetchAll,
		reset
	};
}

export const workerPerformance = createWorkerPerformanceStore();
