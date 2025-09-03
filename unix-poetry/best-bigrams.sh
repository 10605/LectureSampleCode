# words
tr -sc '[:alpha:]' '\n' < data/brown_nolines.txt > /tmp/brownwords.txt

# word frequency
sort /tmp/brownwords.txt | uniq -c > /tmp/brownword_count.txt

# bigrams and bigram frequency
tail -n +2  /tmp/brownwords.txt  > /tmp/brownnextwords.txt
paste /tmp/brownwords.txt /tmp/brownnextwords.txt \
   | sort | uniq -c > /tmp/brownbigram_count.txt


# score with log(freq[xy] / freq[x]*freq[y])
cat /tmp/brownword_count.txt /tmp/brownbigram_count.txt |  awk \
    'NF == 2 { freq[$2] = $1}
     NF == 3 { print log( $1  / (freq[$2] * freq[$3]) ),  $2, $3, $1, freq[$2], freq[$3]}' \
  > /tmp/scored-phrases.txt

# filter out infrequent phrases
cat /tmp/scored-phrases.txt | awk '$4 > 5 {print}' | sort -nr

