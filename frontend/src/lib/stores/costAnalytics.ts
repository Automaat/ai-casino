/**
 * Cost analytics state management
 */

import { writable } from 'svelte/store';
import { api } from '$lib/api/client';
import type * as T from '$lib/types/api';

interface CostAnalyticsState {
	summary: T.CostAnalyticsSummaryResponse | null;
	trends: T.CostTrendsResponse | null;
	bySymbol: T.CostByDimensionListResponse | null;
	byAgent: T.CostByDimensionListResponse | null;
	byModel: T.CostByDimensionListResponse | null;
	loading: boolean;
	error: string | null;
}

function createCostAnalyticsStore() {
	const { subscribe, set, update } = writable<CostAnalyticsState>({
		summary: null,
		trends: null,
		bySymbol: null,
		byAgent: null,
		byModel: null,
		loading: false,
		error: null
	});

	async function fetchAll(
		startDate: string,
		endDate: string,
		period: 'daily' | 'weekly' = 'daily'
	) {
		update(state => ({ ...state, loading: true, error: null }));

		try {
			const [summary, trends, bySymbol, byAgent, byModel] = await Promise.all([
				api.getCostSummary(startDate, endDate),
				api.getCostTrends(period, startDate, endDate),
				api.getCostBySymbol(startDate, endDate),
				api.getCostByAgent(startDate, endDate),
				api.getCostByModel(startDate, endDate)
			]);

			set({
				summary,
				trends,
				bySymbol,
				byAgent,
				byModel,
				loading: false,
				error: null
			});
		} catch (error) {
			console.error('Failed to fetch cost analytics:', error);
			update(state => ({
				...state,
				loading: false,
				error: error instanceof Error ? error.message : 'Failed to fetch cost analytics'
			}));
		}
	}

	function reset() {
		set({
			summary: null,
			trends: null,
			bySymbol: null,
			byAgent: null,
			byModel: null,
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

export const costAnalytics = createCostAnalyticsStore();
