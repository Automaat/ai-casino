# Finnhub API - Free Tier Endpoints

Research findings on Finnhub API endpoint availability (free vs premium tier).

## ✅ Free Tier (60 req/min)

### Market Data
- `/quote` - real-time quotes
- `/stock/candle` - OHLCV candlestick data
- `/forex/candle`, `/crypto/candle` - forex/crypto prices

### Company Info
- `/stock/profile2` - company profile
- `/company-news` - company-specific news (currently used)
- `/news` - general market news
- `/company-basic-financials` - fundamental metrics

### Economic
- `/calendar/earnings` - earnings calendar
- `/calendar/economic` - economic events

## ❌ Premium Only (403 errors)

### Alternative Data
- `/stock/social-sentiment` - Reddit/Twitter sentiment ⚠️ currently used
- `/news-sentiment` - news sentiment indicator ⚠️ currently used
- `/stock/dividends` - dividend data
- `/stock/insider-transactions` - insider trading
- `/stock/pattern` - pattern recognition
- `/stock/support-resistance` - S/R levels

### Other Premium
- Congressional trading, lobbying data, USA spending
- Filing sentiment analysis
- Earnings call transcripts

## Current Issues

Our code uses **premium-only endpoints**:
- `src/data/finnhub.py:151` - `/stock/social-sentiment`
- `src/data/finnhub.py:234` - `/news-sentiment`

Both return 403 on free tier.

## Recommendation

Switch to free endpoints:
- Keep `/company-news` (already in `finnhub_news.py`)
- Remove `FinnhubFetcher` social/news sentiment methods
- Use FinBERT for sentiment instead (already have)

## Rate Limits

- Free: 60 API calls/minute
- Premium: varies by plan (500k/day mentioned)
- All plans: 30 calls/second hard limit

## Sources

- [Finnhub API Docs](https://finnhub.io/docs/api)
- [GitHub Issue #271 - Free Endpoints now Premium](https://github.com/finnhubio/Finnhub-API/issues/271)
- [GitHub Issue #534 - Free Tier Access](https://github.com/finnhubio/Finnhub-API/issues/534)
- [IBKR Campus Guide](https://www.interactivebrokers.com/campus/ibkr-quant-news/exploring-the-finnhub-io-api/)
- [Robot Wealth - Finnhub API](https://robotwealth.com/finnhub-api/)

## Date

Research conducted: 2026-02-16
