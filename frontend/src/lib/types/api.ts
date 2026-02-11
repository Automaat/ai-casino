/**
 * API types matching FastAPI Pydantic models
 */

export type Signal = "BUY" | "SELL" | "HOLD";

export type TradingSession = "REGULAR" | "PRE_MARKET";

export interface HealthResponse {
	status: string;
	uptime_seconds: number;
	daemon_running: boolean;
	current_cycle?: number;
	last_cycle_time?: string;
}

export interface StateSummaryResponse {
	total_analyses: number;
	recent_analyses: AnalysisRecordResponse[];
	total_trades: number;
	positions_count: number;
	win_rate: number;
	error_count: number;
	degradation_tier: string;
}

export interface AnalysisRecordResponse {
	id: string;
	symbol: string;
	timestamp: string;
	signal: Signal;
	confidence: number;
	technical_signal: Signal;
	sentiment_signal: Signal;
	news_signal: Signal;
	final_reasoning: string;
	risk_level: string;
	trading_session: TradingSession;
}

export interface AnalysesResponse {
	analyses: AnalysisRecordResponse[];
	total: number;
}

export interface PositionResponse {
	symbol: string;
	quantity: number;
	avg_entry_price: number;
	current_price: number;
	market_value: number;
	unrealized_pnl: number;
	unrealized_pnl_percent: number;
}

export interface PositionsResponse {
	positions: PositionResponse[];
	total_value: number;
	cash: number;
	portfolio_value: number;
}

export interface SnapshotRecord {
	timestamp: string;
	portfolio_value: number;
	cash: number;
	positions_value: number;
}

export interface SnapshotsResponse {
	snapshots: SnapshotRecord[];
}

export interface RebalanceAllocation {
	symbol: string;
	target_percent: number;
	current_percent: number;
	current_value: number;
	target_value: number;
	action: string;
	shares_to_trade: number;
}

export interface RebalanceResponse {
	allocations: RebalanceAllocation[];
	timestamp: string;
	portfolio_value: number;
}

export interface RiskReportResponse {
	portfolio_volatility: number;
	sharpe_ratio: number;
	max_drawdown: number;
	var_95: number;
	timestamp: string;
}

export interface RiskHistoryResponse {
	history: RiskReportResponse[];
}

export interface CorrelationMatrixResponse {
	symbols: string[];
	matrix: number[][];
	timestamp: string;
}

export interface SectorRotationResponse {
	sector_scores: Record<string, number>;
	timestamp: string;
}

export interface DegradationResponse {
	tier: string;
	score: number;
	reasons: string[];
	timestamp: string;
}

export interface EventResponse {
	events: Array<{
		timestamp: string;
		type: string;
		message: string;
		severity: string;
	}>;
}

export interface MarketEventsResponse {
	events: Array<{
		timestamp: string;
		symbol: string;
		event_type: string;
		description: string;
	}>;
}

export interface GamePlanResponse {
	timestamp: string;
	plan: string;
	symbols: string[];
}

export interface ConfigResponse {
	watchlist: string[];
	interval_minutes: number;
	market_hours_only: boolean;
	auto_trade: boolean;
	trading_mode: string;
}

export interface ExecutionMetric {
	workflow_id: string;
	symbol: string;
	start_time: string;
	end_time: string;
	duration_seconds: number;
	success: boolean;
	error?: string;
}

export interface ExecutionMetricsListResponse {
	metrics: ExecutionMetric[];
}
