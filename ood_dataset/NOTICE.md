# Out of domain dataset

This dataset is a **derived** version of the [review datasets for sentiment analysis](https://github.com/kinit-sk/slovakbert-auxiliary/tree/main/sentiment_reviews) from Pikuliak et al. The original work is available under the [Apache 2.0 License](./LICENSE), and so is this derived work. It includes real and AI-Generated reviews across 5 domains. The `combined/` folder is a combined version of all of them. Each entry in the datasets has the following columns:

- `content:` The review content itself.
- `label:` 0 for original review (from the original dataset), 1 for AI-Generated (by the [mistral-sk-7b](https://huggingface.co/slovak-nlp/mistral-sk-7b)) model.
- `category:` One of `(books, cars, games, mobiles, movies)`

The modifications to the original work are:

- Removing the original second column
- Adding AI-Generated reviews and removing some of the original ones (where the model created duplicates)
- Adding the `label` column and naming the first column `content`