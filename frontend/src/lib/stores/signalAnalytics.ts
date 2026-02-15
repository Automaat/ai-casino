/**
 * Signal analytics state management
 */

import { writable } from 'svelte/store';
import { api } from '$lib/api/client';
import type * as T from '$lib/types/api';

interface SignalAnalyticsState {
	summary: T.SignalFlowSummaryResponse | null;
	sankeyData: T.SankeyFlowResponse | null;
	accuracyByType: T.AccuracyByTypeListResponse | null;
	calibration: T.CalibrationCurveResponse | null;
	timing: T.TimingAnalysisResponse | null;
	executionRate: T.ExecutionRateListResponse | null;
	loading: boolean;
	error: string | null;
}

function createSignalAnalyticsStore() {
	const { subscribe, set, update } = writable<SignalAnalyticsState>({
		summary: null,
		sankeyData: null,
		accuracyByType: null,
		calibration: null,
		timing: null,
		executionRate: null,
		loading: false,
		error: null
	});

	async function fetchAll(startDate: string, endDate: string, horizon: '1d' | '5d' | '20d' = '5d') {
		update((state) => ({ ...state, loading: true, error: null }));

		try {
			const [summary, sankeyData, accuracyByType, calibration, timing, executionRate] =
				await Promise.all([
					api.getSignalSummary(startDate, endDate),
					api.getSignalSankey(startDate, endDate),
					api.getSignalAccuracyByType(startDate, endDate, horizon),
					api.getSignalCalibration(startDate, endDate, horizon),
					api.getSignalTiming(startDate, endDate),
					api.getSignalExecutionRate(startDate, endDate)
				]);

			set({
				summary,
				sankeyData,
				accuracyByType,
				calibration,
				timing,
				executionRate,
				loading: false,
				error: null
			});
		} catch (error) {
			console.error('Failed to fetch signal analytics:', error);
			update((state) => ({
				...state,
				loading: false,
				error: error instanceof Error ? error.message : 'Failed to fetch signal analytics'
			}));
		}
	}

	function reset() {
		set({
			summary: null,
			sankeyData: null,
			accuracyByType: null,
			calibration: null,
			timing: null,
			executionRate: null,
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

export const signalAnalytics = createSignalAnalyticsStore();
