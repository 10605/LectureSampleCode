tr -sc '[:alpha:]' '\n' < data/brown_nolines.txt | sort | uniq -c | sort -nr
