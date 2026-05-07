# Out of domain dataset

This dataset is a **derived** version of the [review datasets for sentiment analysis](https://github.com/kinit-sk/slovakbert-auxiliary/tree/main/sentiment_reviews) from Pikuliak et. al. It includes real and AI-Generated reviews across 5 domains. The `combined/` folder is a combined version of all of them. Each entry in the datasets has the following columns:

- `content:` The review content itself.
- `label:` 0 for original review, 1 for AI-Generated
- `category:` One of `(books, cars, games, mobiles, movies)`