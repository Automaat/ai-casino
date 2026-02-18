-- Add unique constraint to reddit_ticker_sentiment for ON CONFLICT deduplication

CREATE UNIQUE INDEX IF NOT EXISTS idx_reddit_ticker_sentiment_unique
    ON reddit_ticker_sentiment(symbol, subreddit, window_start);
