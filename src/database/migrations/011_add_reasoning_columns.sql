-- Add reasoning columns to signal_outcomes for detailed agent analysis

ALTER TABLE signal_outcomes
    ADD COLUMN technical_reasoning TEXT,
    ADD COLUMN sentiment_reasoning TEXT,
    ADD COLUMN news_reasoning TEXT;

-- Add comment explaining the columns
COMMENT ON COLUMN signal_outcomes.technical_reasoning IS 'Technical analysis interpretation from TechnicalAnalyst agent';
COMMENT ON COLUMN signal_outcomes.sentiment_reasoning IS 'Sentiment analysis summary from SentimentAnalyst agent';
COMMENT ON COLUMN signal_outcomes.news_reasoning IS 'News impact assessment from NewsAnalyst agent';
