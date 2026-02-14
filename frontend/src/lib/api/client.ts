/**
 * Type-safe API client for FastAPI daemon
 */

import type * as T from '$lib/types/api';

// Use Docker service name for SSR (server-side), localhost for browser (client-side)
const API_BASE_URL = import.meta.env.VITE_API_URL ||
	(typeof window === 'undefined' ? 'http://daemon:8484' : 'http://localhost:8484');

class APIError extends Error {
	constructor(
		message: string,
		public status: number,
		public statusText: string
	) {
		super(message);
		this.name = 'APIError';
	}
}

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
	const url = `${API_BASE_URL}${endpoint}`;
	
	try {
		const response = await fetch(url, {
			...options,
			headers: {
				'Content-Type': 'application/json',
				...options?.headers
			}
		});

		if (!response.ok) {
			throw new APIError(
				`API error: ${response.statusText}`,
				response.status,
				response.statusText
			);
		}

		return await response.json();
	} catch (error) {
		if (error instanceof APIError) {
			throw error;
		}
		throw new APIError(
			`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`,
			0,
			'Network Error'
		);
	}
}

export const api = {
	// Health & State
	async getHealth(): Promise<T.HealthResponse> {
		return fetchAPI<T.HealthResponse>('/health');
	},

	async getServiceHealth(): Promise<T.ServiceHealthResponse> {
		return fetchAPI<T.ServiceHealthResponse>('/health/services');
	},

	async getStateSummary(): Promise<T.StateSummaryResponse> {
		return fetchAPI<T.StateSummaryResponse>('/state/summary');
	},

	async getWatchlist(): Promise<T.WatchlistResponse> {
		return fetchAPI<T.WatchlistResponse>('/watchlist');
	},

	// Analyses
	async getAnalyses(params?: {
		limit?: number;
		symbol?: string;
	}): Promise<T.AnalysesResponse> {
		const query = new URLSearchParams();
		if (params?.limit) query.set('limit', params.limit.toString());
		if (params?.symbol) query.set('symbol', params.symbol);
		const queryString = query.toString();
		return fetchAPI<T.AnalysesResponse>(`/analyses${queryString ? `?${queryString}` : ''}`);
	},

	// Portfolio
	async getPositions(): Promise<T.PositionsResponse> {
		return fetchAPI<T.PositionsResponse>('/positions');
	},

	async getSnapshots(limit?: number): Promise<T.SnapshotsResponse> {
		const query = limit ? `?limit=${limit}` : '';
		return fetchAPI<T.SnapshotsResponse>(`/portfolio/snapshots${query}`);
	},

	async getRebalance(): Promise<T.RebalanceResponse> {
		return fetchAPI<T.RebalanceResponse>('/portfolio/rebalance');
	},

	// Risk
	async getRisk(): Promise<T.RiskReportResponse> {
		return fetchAPI<T.RiskReportResponse>('/risk');
	},

	async getRiskHistory(limit?: number): Promise<T.RiskHistoryResponse> {
		const query = limit ? `?limit=${limit}` : '';
		return fetchAPI<T.RiskHistoryResponse>(`/risk/history${query}`);
	},

	async getCorrelation(): Promise<T.CorrelationMatrixResponse> {
		return fetchAPI<T.CorrelationMatrixResponse>('/correlation/latest');
	},

	// Market Analysis
	async getSectorRotation(): Promise<T.SectorRotationResponse> {
		return fetchAPI<T.SectorRotationResponse>('/sector-rotation/latest');
	},

	async getDegradation(): Promise<T.DegradationResponse> {
		return fetchAPI<T.DegradationResponse>('/degradation');
	},

	// Events
	async getEvents(limit?: number): Promise<T.EventResponse> {
		const query = limit ? `?limit=${limit}` : '';
		return fetchAPI<T.EventResponse>(`/events${query}`);
	},

	async getMarketEvents(limit?: number): Promise<T.MarketEventsResponse> {
		const query = limit ? `?limit=${limit}` : '';
		return fetchAPI<T.MarketEventsResponse>(`/events/market${query}`);
	},

	async getDegradationHistory(limit?: number): Promise<T.DegradationHistoryResponse> {
		const query = limit ? `?limit=${limit}` : '';
		return fetchAPI<T.DegradationHistoryResponse>(`/events/degradation-history${query}`);
	},

	// Config
	async getConfig(): Promise<T.ConfigResponse> {
		return fetchAPI<T.ConfigResponse>('/config');
	},

	// Game Plan
	async getGamePlan(): Promise<T.GamePlanResponse> {
		return fetchAPI<T.GamePlanResponse>('/game-plan');
	},

	// Execution Metrics
	async getExecutionMetrics(): Promise<T.ExecutionMetricsListResponse> {
		return fetchAPI<T.ExecutionMetricsListResponse>('/api/execution-metrics');
	},

	async getExecutionMetricDetail(workflowId: string): Promise<T.WorkflowExecutionMetrics> {
		return fetchAPI<T.WorkflowExecutionMetrics>(
			`/api/execution-metrics/${encodeURIComponent(workflowId)}`
		);
	},

	// Full Config
	async getFullConfig(): Promise<T.FullConfigResponse> {
		return fetchAPI<T.FullConfigResponse>('/config/full');
	},

	// Execution Graphs
	async getActiveExecutionGraphs(): Promise<T.ActiveExecutionGraphsResponse> {
		return fetchAPI<T.ActiveExecutionGraphsResponse>('/api/execution/active');
	},

	async getExecutionGraphHistory(params?: {
		limit?: number;
		symbol?: string;
		days?: number;
	}): Promise<T.ExecutionGraphHistoryResponse> {
		const query = new URLSearchParams();
		if (params?.limit) query.set('limit', params.limit.toString());
		if (params?.symbol) query.set('symbol', params.symbol);
		if (params?.days) query.set('days', params.days.toString());
		const queryString = query.toString();
		return fetchAPI<T.ExecutionGraphHistoryResponse>(
			`/api/execution/history${queryString ? `?${queryString}` : ''}`
		);
	},

	// Supervisor Metrics
	async getSupervisorMetricsRecent(params?: { limit?: number }): Promise<T.SupervisorMetricsRecent> {
		const query = new URLSearchParams();
		if (params?.limit !== undefined) query.set('limit', params.limit.toString());
		const queryString = query.toString();
		return fetchAPI<T.SupervisorMetricsRecent>(
			`/supervisor/metrics/recent${queryString ? `?${queryString}` : ''}`
		);
	},

	async getSupervisorMetricsSummary(params?: { hours?: number }): Promise<T.SupervisorMetricsSummary> {
		const query = new URLSearchParams();
		if (params?.hours !== undefined) query.set('hours', params.hours.toString());
		const queryString = query.toString();
		return fetchAPI<T.SupervisorMetricsSummary>(
			`/supervisor/metrics/summary${queryString ? `?${queryString}` : ''}`
		);
	},

	async getSupervisorMetricsWorkers(params?: { hours?: number }): Promise<T.SupervisorMetricsWorkers> {
		const query = new URLSearchParams();
		if (params?.hours !== undefined) query.set('hours', params.hours.toString());
		const queryString = query.toString();
		return fetchAPI<T.SupervisorMetricsWorkers>(
			`/supervisor/metrics/workers${queryString ? `?${queryString}` : ''}`
		);
	},

	async getSupervisorMetricsErrors(params?: { hours?: number }): Promise<T.SupervisorMetricsErrors> {
		const query = new URLSearchParams();
		if (params?.hours !== undefined) query.set('hours', params.hours.toString());
		const queryString = query.toString();
		return fetchAPI<T.SupervisorMetricsErrors>(
			`/supervisor/metrics/errors${queryString ? `?${queryString}` : ''}`
		);
	},

	// Validation
	async getPaperTradingValidation(): Promise<T.PaperTradingValidation> {
		return fetchAPI<T.PaperTradingValidation>('/validation/paper-trading');
	},

	// Discovery
	async getDiscoveryInsights(): Promise<T.DiscoveryInsightsResponse> {
		return fetchAPI<T.DiscoveryInsightsResponse>('/api/discovery/insights');
	}
};

export { APIError };
