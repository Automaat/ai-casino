/**
 * Screening state management
 */

import { writable } from 'svelte/store';
import { api } from '$lib/api/client';
import type * as T from '$lib/types/api';

interface ScreeningState {
	history: T.ScreeningHistoryResponse | null;
	insights: T.ScreeningInsightsResponse | null;
	loading: boolean;
	error: string | null;
}

function createScreeningStore() {
	const { subscribe, set, update } = writable<ScreeningState>({
		history: null,
		insights: null,
		loading: false,
		error: null
	});

	async function fetchAll(limit: number = 30) {
		update((state) => ({ ...state, loading: true, error: null }));

		try {
			const [history, insights] = await Promise.all([
				api.getScreeningHistory(limit),
				api.getScreeningInsights()
			]);

			set({
				history,
				insights,
				loading: false,
				error: null
			});
		} catch (error) {
			console.error('Failed to fetch screening data:', error);
			update((state) => ({
				...state,
				loading: false,
				error: error instanceof Error ? error.message : 'Failed to fetch screening data'
			}));
		}
	}

	function reset() {
		set({
			history: null,
			insights: null,
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

export const screening = createScreeningStore();
