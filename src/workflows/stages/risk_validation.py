"""Stage 5.5: Pre-decision risk validation."""

from src.validators.risk import AnalysisContext, RiskValidator
from src.workflows.models.risk_validation import RiskValidationInput, RiskValidationOutput


def validate_analyses_stage(
    validation_input: RiskValidationInput,
    risk_validator: RiskValidator,
) -> RiskValidationOutput:
    """Stage 5.5: Validate analyses before decision.

    Args:
        validation_input: Risk validation input with analyses and context
        risk_validator: Risk validator instance

    Returns:
        RiskValidationOutput with validation result
    """
    ctx = AnalysisContext(
        symbol=validation_input.symbol,
        trading_session=validation_input.trading_session,
        technical=validation_input.technical_analysis,
        sentiment=validation_input.sentiment_analysis,
        news=validation_input.news_analysis,
        fundamental=validation_input.fundamental_analysis,
        bullish=validation_input.bullish_research,
        bearish=validation_input.bearish_research,
        market_data=validation_input.market_data,
        degradation_context=validation_input.degradation_context,
    )

    result = risk_validator.validate(ctx)

    return RiskValidationOutput(validation_result=result)
