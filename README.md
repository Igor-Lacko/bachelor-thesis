# Detection of AI-Generated Reviews on online platforms

This is the repository for my bachelor's thesis, which is about the implementation of a classifier for AI-Generated reviews. The thesis concerns itself with reviews specifically in the Slovak language. This repository contains the thesis itself, and all the Python source code.This README describes the structure of the overall repository, to see the descriptions of the main implementation parts, you can navigate to them into their own repositories. They are they own Git repositories, and are included here as Git submodules. These include:

- `review-classifier`: The implementation of the classification library/models. Also contains the framework for conducting experiments.
- `review-scraper`: Implementation of a web scraper for hotel reviews.
- `review-synthesis`: Implementation of several models/templates for generating synthetic reviews in Slovak from examples.
- `review-utils`: Other utilities (scripts for plotting graphs and creating LaTeX tables, data processing...)


The structure of this repository is:

```
├── presentations                           # Semestral project and state final exam presentation PDF
│   └── figures
├── README.md                               # This README
├── review-classifier                       # The classification library and experiment framework
├── review-scraper                          # Scraper for reviews
├── review-synthesis                        # Scripts for synthesis of AI generated reviews and their quality evaluation
├── review-utils                            # Other utilities (visualizations, data processing...)
└── thesis                                  # Thesis text
    ├── BP.pdf
    └── figures
```