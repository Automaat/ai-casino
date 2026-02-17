"""Configuration endpoints."""

from fastapi import APIRouter, Request

from src.daemon.api.models import ConfigResponse, FullConfigResponse
from src.daemon.api.routers.shared import get_components, mask_sensitive_field

router = APIRouter(tags=["config"])


@router.get("/config", response_model=ConfigResponse)
async def config(request: Request) -> ConfigResponse:
    """Get daemon configuration (no secrets)."""
    components = get_components(request)

    trading_mode = await components.state.get_current_trading_mode()
    return ConfigResponse(
        watchlist=components.config.watchlist,
        interval_minutes=components.config.interval_minutes,
        market_hours_only=components.config.market_hours_only,
        auto_trade=components.config.auto_trade,
        trading_mode=trading_mode,
        pre_market_enabled=components.config.schedule.enable_pre_market,
    )


@router.get("/config/full", response_model=FullConfigResponse)
async def config_full(request: Request) -> FullConfigResponse:
    """Get full daemon configuration with masked sensitive fields."""
    components = get_components(request)
    cfg = components.config

    # Mask API keys
    masked_api_keys = {
        "alpha_vantage_api_key": mask_sensitive_field(cfg.api_keys.alpha_vantage_api_key),
        "marketaux_api_key": mask_sensitive_field(cfg.api_keys.marketaux_api_key),
        "finnhub_api_key": mask_sensitive_field(cfg.api_keys.finnhub_api_key),
        "alpaca_api_key": mask_sensitive_field(cfg.api_keys.alpaca_api_key),
        "alpaca_secret_key": mask_sensitive_field(cfg.api_keys.alpaca_secret_key),
        "alpaca_paper_api_key": mask_sensitive_field(cfg.api_keys.alpaca_paper_api_key),
        "alpaca_paper_secret_key": mask_sensitive_field(cfg.api_keys.alpaca_paper_secret_key),
        "reddit_client_id": mask_sensitive_field(cfg.api_keys.reddit_client_id),
        "reddit_client_secret": mask_sensitive_field(cfg.api_keys.reddit_client_secret),
        "reddit_user_agent": mask_sensitive_field(cfg.api_keys.reddit_user_agent),
        "anthropic_api_key": mask_sensitive_field(cfg.api_keys.anthropic_api_key),
        "openai_api_key": mask_sensitive_field(cfg.api_keys.openai_api_key),
        "openai_api_base": mask_sensitive_field(cfg.api_keys.openai_api_base),
    }

    # Mask telegram secrets
    notifications_dict = cfg.notifications.model_dump()
    notifications_dict["telegram"]["bot_token"] = mask_sensitive_field(cfg.notifications.telegram.bot_token)
    notifications_dict["telegram"]["chat_id"] = mask_sensitive_field(cfg.notifications.telegram.chat_id)

    return FullConfigResponse(
        watchlist=cfg.watchlist,
        interval_minutes=cfg.interval_minutes,
        market_hours_only=cfg.market_hours_only,
        auto_trade=cfg.auto_trade,
        max_concurrent_analyses=cfg.max_concurrent_analyses,
        trading_mode=await components.state.get_current_trading_mode(),
        paper_trading=cfg.paper_trading.model_dump(),
        schedule=cfg.schedule.model_dump(),
        state=cfg.state.model_dump(),
        journal=cfg.journal.model_dump(),
        health=cfg.health.model_dump(),
        optimization=cfg.optimization.model_dump(),
        prefetch=cfg.prefetch.model_dump(),
        sector_rotation=cfg.sector_rotation.model_dump(),
        earnings_calendar=cfg.earnings_calendar.model_dump(),
        peer_analysis=cfg.peer_analysis.model_dump(),
        correlation_audit=cfg.correlation_audit.model_dump(),
        reporting=cfg.reporting.model_dump(),
        risk_limits=cfg.risk_limits.model_dump(),
        rebalancing=cfg.rebalancing.model_dump(),
        signal_tracking=cfg.signal_tracking.model_dump(),
        pre_trade_backtesting=cfg.pre_trade_backtesting.model_dump(),
        game_plan=cfg.game_plan.model_dump(),
        position_management=cfg.position_management.model_dump(),
        monte_carlo=cfg.monte_carlo.model_dump(),
        notifications=notifications_dict,
        analysis_orchestration=cfg.analysis_orchestration.model_dump(),
        news_watcher=cfg.news_watcher.model_dump(),
        social_watcher=cfg.social_watcher.model_dump(),
        filings_watcher=cfg.filings_watcher.model_dump(),
        anomaly_watcher=cfg.anomaly_watcher.model_dump(),
        api=cfg.api.model_dump(),
        llm=cfg.llm.model_dump(),
        api_keys=masked_api_keys,
    )
