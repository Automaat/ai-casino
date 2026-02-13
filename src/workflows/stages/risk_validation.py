"""Stage 5.5: Pre-decision risk validation."""

from src.validators.risk import RiskValidator
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
    result = risk_validator.validate(
        validation_input.symbol,
        validation_input.trading_session,
        validation_input.technical_analysis,
        validation_input.sentiment_analysis,
        validation_input.news_analysis,
        validation_input.fundamental_analysis,
        validation_input.bullish_research,
        validation_input.bearish_research,
        validation_input.market_data,
        validation_input.degradation_context,
    )

    return RiskValidationOutput(validation_result=result)
