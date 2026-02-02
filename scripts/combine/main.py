"""
main.py
Main script to combine the real and synthetic datasets into one, and also combine the various
synthetic dataset csvs into one csv file.
Author: Igor Lacko
"""

from utils import *
from combine import combine

def main():
    """Script main function."""
    validate_args()
    combine()

if __name__ == "__main__": 
    main()