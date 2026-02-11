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
	var_99: number;
	cvar_95: number;
	cvar_99: number;
	cdar_95: number;
	risk_status: string;
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
	timestamp: string;
	leading_sectors?: string[];
	lagging_sectors?: string[];
	sector_strengths?: Record<string, number>;
	sector_momenta?: Record<string, string>;
	flagged_positions?: string[];
}

export interface DegradationResponse {
	tier: string;
	unavailable_services: string[];
	confidence_adjustment: number;
	halt_reason: string | null;
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
	date: string;
	priority_symbols: string[];
	risk_stance: string;
	sector_focus: string[];
	reasoning: string;
	confidence: number;
	generated_at: string;
}

export interface ServiceCheck {
	service: string;
	status: string;
	message: string;
	duration_ms: number;
	checked_at: string;
}

export interface ServiceHealthResponse {
	overall_status: string;
	service_checks: ServiceCheck[];
}

export interface WatchlistResponse {
	symbols: string[];
	count: number;
	sources: Record<string, number>;
}

export interface ConfigResponse {
	watchlist: string[];
	interval_minutes: number;
	market_hours_only: boolean;
	auto_trade: boolean;
	trading_mode: string;
}

export interface FullConfigResponse {
	watchlist: string[];
	interval_minutes: number;
	market_hours_only: boolean;
	auto_trade: boolean;
	max_concurrent_analyses: number;
	trading_mode: string;
	paper_trading: Record<string, any>;
	schedule: Record<string, any>;
	state: Record<string, any>;
	journal: Record<string, any>;
	health: Record<string, any>;
	optimization: Record<string, any>;
	screening: Record<string, any>;
	prefetch: Record<string, any>;
	sector_rotation: Record<string, any>;
	earnings_calendar: Record<string, any>;
	peer_analysis: Record<string, any>;
	correlation_audit: Record<string, any>;
	reporting: Record<string, any>;
	risk_limits: Record<string, any>;
	rebalancing: Record<string, any>;
	signal_tracking: Record<string, any>;
	pre_trade_backtesting: Record<string, any>;
	game_plan: Record<string, any>;
	position_management: Record<string, any>;
	monte_carlo: Record<string, any>;
	notifications: Record<string, any>;
	analysis_orchestration: Record<string, any>;
	news_watcher: Record<string, any>;
	social_watcher: Record<string, any>;
	filings_watcher: Record<string, any>;
	anomaly_watcher: Record<string, any>;
	api: Record<string, any>;
	llm: Record<string, any>;
	api_keys: Record<string, any>;
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
