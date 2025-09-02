tr -sc '[:alpha:]' '\n' < data/bluecorpus.txt | sort | uniq -c | sort -nr
