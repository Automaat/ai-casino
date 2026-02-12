/**
 * Dashboard state management with Svelte stores
 */

import { writable, derived, readable } from 'svelte/store';
import { api } from '$lib/api/client';
import type * as T from '$lib/types/api';

// Auto-refresh interval (5 seconds)
const REFRESH_INTERVAL = 5000;

// Health store
function createHealthStore() {
	return readable<T.HealthResponse | null>(null, (set) => {
		async function fetch() {
			try {
				const data = await api.getHealth();
				set(data);
			} catch (error) {
				console.error('Failed to fetch health:', error);
				set(null);
			}
		}

		// Initial fetch
		fetch();

		// Auto-refresh
		const interval = setInterval(fetch, REFRESH_INTERVAL);

		// Cleanup function - clears interval when no subscribers
		return () => clearInterval(interval);
	});
}

// State summary store
function createStateSummaryStore() {
	return readable<T.StateSummaryResponse | null>(null, (set) => {
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

		// Initial fetch
		fetch();

		// Auto-refresh
		const interval = setInterval(fetch, REFRESH_INTERVAL);

		// Cleanup function
		return () => clearInterval(interval);
	});
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
	return readable<T.PositionsResponse | null>(null, (set) => {
		async function fetch() {
			try {
				const data = await api.getPositions();
				set(data);
			} catch (error) {
				console.error('Failed to fetch positions:', error);
				set(null);
			}
		}

		// Initial fetch
		fetch();

		// Auto-refresh
		const interval = setInterval(fetch, REFRESH_INTERVAL);

		// Cleanup function
		return () => clearInterval(interval);
	});
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

// Sector rotation store
function createSectorRotationStore() {
	const { subscribe, set } = writable<T.SectorRotationResponse | null>(null);

	async function fetch() {
		try {
			const data = await api.getSectorRotation();
			set(data);
		} catch (error) {
			console.error('Failed to fetch sector rotation:', error);
			set(null);
		}
	}

	return { subscribe, fetch };
}

// Config store
function createConfigStore() {
	return readable<T.FullConfigResponse | null>(null, (set) => {
		async function fetch() {
			try {
				const data = await api.getFullConfig();
				set(data);
			} catch (error) {
				console.error('Failed to fetch config:', error);
				set(null);
			}
		}

		// Initial fetch
		fetch();

		// Auto-refresh
		const interval = setInterval(fetch, REFRESH_INTERVAL);

		// Cleanup function
		return () => clearInterval(interval);
	});
}

// Service health store
function createServiceHealthStore() {
	return readable<T.ServiceHealthResponse | null>(null, (set) => {
		async function fetch() {
			try {
				const data = await api.getServiceHealth();
				set(data);
			} catch (error) {
				console.error('Failed to fetch service health:', error);
				set(null);
			}
		}

		// Initial fetch
		fetch();

		// Auto-refresh
		const interval = setInterval(fetch, REFRESH_INTERVAL);

		// Cleanup function
		return () => clearInterval(interval);
	});
}

// Game plan store
function createGamePlanStore() {
	return readable<T.GamePlanResponse | null>(null, (set) => {
		async function fetch() {
			try {
				const data = await api.getGamePlan();
				set(data);
			} catch (error) {
				console.error('Failed to fetch game plan:', error);
				set(null);
			}
		}

		// Initial fetch
		fetch();

		// Auto-refresh
		const interval = setInterval(fetch, REFRESH_INTERVAL);

		// Cleanup function
		return () => clearInterval(interval);
	});
}

// Watchlist store
function createWatchlistStore() {
	return readable<T.WatchlistResponse | null>(null, (set) => {
		async function fetch() {
			try {
				const data = await api.getWatchlist();
				set(data);
			} catch (error) {
				console.error('Failed to fetch watchlist:', error);
				set(null);
			}
		}

		// Initial fetch
		fetch();

		// Auto-refresh
		const interval = setInterval(fetch, REFRESH_INTERVAL);

		// Cleanup function
		return () => clearInterval(interval);
	});
}

// Degradation store
function createDegradationStore() {
	return readable<T.DegradationResponse | null>(null, (set) => {
		async function fetch() {
			try {
				const data = await api.getDegradation();
				set(data);
			} catch (error) {
				console.error('Failed to fetch degradation:', error);
				set(null);
			}
		}

		// Initial fetch
		fetch();

		// Auto-refresh
		const interval = setInterval(fetch, REFRESH_INTERVAL);

		// Cleanup function
		return () => clearInterval(interval);
	});
}

// Events store
function createEventsStore() {
	const { subscribe, set } = writable<T.EventResponse | null>(null);

	async function fetch(params?: { limit?: number }) {
		try {
			const data = await api.getEvents(params?.limit);
			set(data);
			return data;
		} catch (error) {
			console.error('Failed to fetch events:', error);
			set(null);
			return null;
		}
	}

	return { subscribe, fetch };
}

// Market events store
function createMarketEventsStore() {
	const { subscribe, set } = writable<T.MarketEventsResponse | null>(null);

	async function fetch(params?: { limit?: number }) {
		try {
			const data = await api.getMarketEvents(params?.limit);
			set(data);
			return data;
		} catch (error) {
			console.error('Failed to fetch market events:', error);
			set(null);
			return null;
		}
	}

	return { subscribe, fetch };
}

// Degradation history store
function createDegradationHistoryStore() {
	const { subscribe, set } = writable<T.DegradationHistoryResponse | null>(null);

	async function fetch(params?: { limit?: number }) {
		try {
			const data = await api.getDegradationHistory(params?.limit);
			set(data);
			return data;
		} catch (error) {
			console.error('Failed to fetch degradation history:', error);
			set(null);
			return null;
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
export const sectorRotation = createSectorRotationStore();
export const config = createConfigStore();
export const serviceHealth = createServiceHealthStore();
export const gamePlan = createGamePlanStore();
export const watchlist = createWatchlistStore();
export const degradation = createDegradationStore();
export const events = createEventsStore();
export const marketEvents = createMarketEventsStore();
export const degradationHistory = createDegradationHistoryStore();

// Derived stores
export const isDaemonRunning = derived(health, ($health) => $health?.daemon_running ?? false);
export const hasPositions = derived(positions, ($positions) => ($positions?.positions?.length ?? 0) > 0);
