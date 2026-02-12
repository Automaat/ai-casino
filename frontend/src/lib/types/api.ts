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
	entry_price: number;
	current_qty: number;
	current_stop_loss: number;
	entry_timestamp: string;
	entry_signal: string;
	entry_confidence: number;
	days_held: number;
	trailing_stop_activated: boolean;
	breakeven_activated: boolean;
	profit_targets: number[];
	current_price: number;
}

export interface PositionsResponse {
	positions: PositionResponse[];
	count: number;
}

export interface SnapshotRecord {
	timestamp: string;
	portfolio_value: number;
	cash: number;
	positions_value: number;
}

export interface SnapshotsResponse {
	snapshots: SnapshotRecord[];
	count: number;
	database_enabled: boolean;
	has_trades: boolean;
}

export interface RebalanceAllocation {
	symbol: string;
	target_weight: number;
	current_weight: number;
	delta: number;
	action: string;
}

export interface RebalanceResponse {
	enabled: boolean;
	timestamp?: string | null;
	method?: string | null;
	allocations: RebalanceAllocation[];
	expected_return?: number | null;
	expected_volatility?: number | null;
	sharpe_ratio?: number | null;
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

export interface SystemEvent {
	timestamp: string;
	event_type: string;
	data: Record<string, any>;
}

export interface EventResponse {
	events: SystemEvent[];
}

export interface MarketEvent {
	signal_timestamp: string;
	event: {
		event_type: string;
		symbol: string;
		summary: string;
		impact_score: number;
		source: string;
	};
	summary: string;
}

export interface MarketEventsResponse {
	events: MarketEvent[];
}

export interface DegradationRecord {
	timestamp: string;
	tier: string;
	unavailable_services: string[];
	confidence_adjustment: number;
	reason: string;
}

export interface DegradationHistoryResponse {
	records: DegradationRecord[];
	count: number;
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
	timestamp: string;
	total_latency_ms: number;
	success: boolean;
	error?: string;
}

export interface ExecutionMetricsListResponse {
	metrics: ExecutionMetric[];
}

export interface LLMCallMetric {
	timestamp: string;
	agent_name: string;
	method: string;
	provider: string;
	model: string;
	latency_ms: number;
	input_tokens: number | null;
	output_tokens: number | null;
	estimated_cost_usd: number | null;
	success: boolean;
	error: string | null;
}

export interface AgentTimingMetric {
	agent_name: string;
	latency_ms: number;
	llm_calls: number;
}

export interface PipelineStageMetric {
	stage: string;
	latency_ms: number;
}

export interface WorkflowExecutionMetrics {
	workflow_id: string;
	symbol: string;
	timestamp: string;
	total_latency_ms: number;
	llm_calls: LLMCallMetric[];
	agent_timings: AgentTimingMetric[];
	pipeline_stages: PipelineStageMetric[];
	total_input_tokens: number;
	total_output_tokens: number;
	total_estimated_cost_usd: number;
	provider: string;
	model: string;
}
