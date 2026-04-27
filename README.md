# Detection of AI-Generated Reviews on online platforms

This is the repository for my bachelor's thesis, which is about the implementation of a classifier for AI-Generated reviews. The thesis concerns itself with reviews specifically in the Slovak language. This repository contains the thesis itself, and all the Python source code. The used dataset itself is not published, due to [Booking.com's Terms of Service](https://www.booking.com/content/terms.html). This README describes the structure of the overall repository, to see the descriptions of the main implementation parts, you can navigate to them into their own repositories. They are they own Git repositories, and are included here as Git submodules. These include:

- `review-classifier`: The implementation of the classification library/models. Also contains the framework for conducting experiments.
- `review-scraper`: Implementation of a web scraper for [Booking.com](https://www.booking.com) Slovak reviews.
- `review-synthesis`: Implementation of several models/templates for generating synthetic hotel reviews in Slovak from examples.


The structure of this thesis is:

