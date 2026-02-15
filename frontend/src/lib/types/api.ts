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
	win_rate: number | null;
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

export interface MetricsSnapshot {
	expected_return: number;
	expected_volatility: number;
	sharpe_ratio: number;
}

export interface RebalanceCalculation {
	timestamp: string;
	method: string;
	allocations: RebalanceAllocation[];
	expected_return: number;
	expected_volatility: number;
	sharpe_ratio: number;
}

export interface RebalanceHistoryEntry {
	timestamp: string;
	method: string;
	avg_deviation_pct: number;
	max_deviation_pct: number;
	metrics: MetricsSnapshot;
}

export interface RebalancingHistoryResponse {
	enabled: boolean;
	current_portfolio_value: number;
	rebalance_threshold: number;
	current_metrics: MetricsSnapshot | null;
	latest: RebalanceCalculation | null;
	history: RebalanceHistoryEntry[];
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

export type ExecutionNodeType = "TOOL" | "AGENT" | "WORKFLOW_STAGE";

export type ExecutionStatus = "RUNNING" | "COMPLETED" | "FAILED";

export interface ExecutionNode {
	node_id: string;
	node_type: ExecutionNodeType;
	name: string;
	parent_id: string | null;
	status: ExecutionStatus;
	start_time: string;
	end_time: string | null;
	duration_ms: number | null;
	error: string | null;
	metadata: Record<string, any>;
}

export interface ExecutionGraph {
	workflow_id: string;
	symbol: string | null;
	root_node_id: string | null;
	nodes: Record<string, ExecutionNode>;
	created_at: string;
	updated_at: string;
}

export interface ActiveExecutionGraphsResponse {
	graphs: ExecutionGraph[];
	count: number;
}

export interface ExecutionGraphDetailResponse {
	workflow_id: string;
	graph: ExecutionGraph;
	source: string;
}

export interface ExecutionGraphHistoryResponse {
	graphs: ExecutionGraph[];
	count: number;
	database_enabled: boolean;
}

export type EventType =
	| "CYCLE_START"
	| "CYCLE_COMPLETE"
	| "ANALYSIS_START"
	| "ANALYSIS_COMPLETE"
	| "ANALYSIS_ERROR"
	| "TRADE_EXECUTED"
	| "HEALTH_CHECK"
	| "DEGRADATION"
	| "SCHEDULED_TASK"
	| "STATE_UPDATE"
	| "EXECUTION_NODE_START"
	| "EXECUTION_NODE_COMPLETE";

export interface DashboardEvent {
	event_id: string;
	event_type: EventType;
	timestamp: string;
	data: Record<string, any>;
}

// Supervisor Metrics
export interface SupervisorMetricRecord {
	// Backend fields (SupervisorMetricResponse)
	id: string;
	created_at: string;
	workflow_id: string;
	symbol: string;
	timestamp: string;
	required_analyses: string[];
	optional_analyses: string[];
	skip_analyses: Record<string, string>;
	routing_reasoning: string;
	total_workers: number;
	required_workers: number;
	optional_workers: number;
	successful_workers: number;
	failed_workers: number;
	routing_decision_ms: number;
	group1_execution_ms: number;
	research_execution_ms: number;
	total_supervisor_overhead_ms: number;
	worker_timings: Record<string, number>;
	worker_errors: Record<string, string>;
	total_llm_calls: number;
	total_cost_usd: number;
	planning_fallback_used: boolean;
	synthesis_fallback_used: boolean;
	confidence_adjustment: number;
	synthesis_reasoning: string;
	parallel_efficiency_percent: number;
	timeout_triggered: boolean;
}

export interface SupervisorMetricsRecent {
	metrics: SupervisorMetricRecord[];
	count: number;
}

export interface SupervisorMetricsSummary {
	avg_efficiency_percent: number;
	avg_routing_ms: number;
	avg_group1_ms: number;
	avg_research_ms: number;
	avg_total_ms: number;
	timeout_rate_percent: number;
	sample_size: number;
	symbol: string | null;
}

export interface WorkerStats {
	total_executions: number;
	successful_executions: number;
	failed_executions: number;
	success_rate: number;
	avg_duration_ms: number;
}

export interface SupervisorMetricsWorkers {
	worker_stats: Record<string, WorkerStats>;
	total_workers: number;
	sample_size: number;
}

export interface SupervisorMetricsErrors {
	error_counts: Record<string, number>;
	total_errors: number;
}

export interface ValidationCriterion {
	name: string;
	passed: boolean;
	current_value: number;
	threshold: number;
	message: string;
}

export interface PaperTradingValidation {
	ready_for_live: boolean;
	assessment_date: string;
	paper_trading_duration_days: number;
	total_paper_trades: number;
	criteria: ValidationCriterion[];
	recommendations: string[];
}

export interface DiscoverySourceBreakdown {
	source: string;
	count: number;
	percentage: number;
}

export interface DiscoveryRecord {
	symbol: string;
	discovered_at: string;
	composite_score: number;
	sources: string[];
	added_to_watchlist: boolean;
	first_signal: string | null;
	first_signal_date: string | null;
	outcome_7d: number | null;
	outcome_30d: number | null;
}

export interface DiscoverySuccessMetrics {
	total_discovered: number;
	added_to_watchlist: number;
	received_signal: number;
	signal_rate: number;
}

export interface DiscoveryInsightsResponse {
	source_breakdown: DiscoverySourceBreakdown[];
	success_metrics: DiscoverySuccessMetrics;
	recent_discoveries: DiscoveryRecord[];
	avg_composite_score: number;
	total_discoveries: number;
}

export interface PositionManagementActionResponse {
	action_type: string;
	timestamp: string;
	old_stop_loss: number | null;
	new_stop_loss: number | null;
	qty_sold: number | null;
	price: number;
	reason: string;
	executed: boolean;
	order_id: string | null;
}

export interface PositionTimelineResponse {
	symbol: string;
	entry_price: number;
	current_price: number;
	current_qty: number;
	entry_timestamp: string;
	days_held: number;
	actions: PositionManagementActionResponse[];
	count: number;
	database_enabled: boolean;
}

export type TradeAction = "BUY" | "SELL";

export type TradeStatus = "OPEN" | "CLOSED" | "REJECTED";

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";

export interface TradeResponse {
	id: string;
	timestamp: string;
	symbol: string;
	action: TradeAction;
	entry_price: number;
	exit_price: number | null;
	shares: number;
	confidence: number;
	risk_level: RiskLevel;
	status: TradeStatus;
	pnl: number | null;
	pnl_percent: number | null;
	strategy_name: string | null;
	is_paper_trade: boolean;
	closed_at: string | null;
}

export interface TradesResponse {
	trades: TradeResponse[];
	total_count: number;
	returned_count: number;
	database_enabled: boolean;
}

export interface EnrichedTradeResponse {
	trade: TradeResponse;
	analysis: AnalysisRecordResponse | null;
}

// Cost Analytics
export interface CostAnalyticsSummaryResponse {
	total_cost_usd: number;
	total_tokens: number;
	total_executions: number;
	avg_cost_per_execution: number;
	avg_cost_per_signal: number;
	forecast_30d_usd: number;
	date_range: [string, string];
}

export interface CostTrendPointResponse {
	timestamp: string;
	cost_usd: number;
	tokens: number;
	execution_count: number;
}

export interface CostByDimensionResponse {
	dimension_value: string;
	cost_usd: number;
	tokens: number;
	execution_count: number;
	percentage: number;
}

export interface CostTrendsResponse {
	trends: CostTrendPointResponse[];
	count: number;
}

export interface CostByDimensionListResponse {
	data: CostByDimensionResponse[];
	count: number;
}
