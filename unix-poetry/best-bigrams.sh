# word frequency
tr -sc '[:alpha:]' '\n' < data/bluecorpus.txt | sort | uniq -c > /tmp/bluewc.txt

# all bigrams
tail -n +2  /tmp/bluewords.txt  > /tmp/bluenextwords.txt
paste /tmp/bluewords.txt /tmp/bluenextwords.txt | sort | uniq -c > /tmp/bluebigram_count.txt

# score with log(freq[xy] / freq[x]*freq[y])

cat /tmp/bluewc.txt /tmp/bluebigram_count.txt |  awk \
    'NF == 2 { freq[$2] = $1}
     NF == 3 { print log( $1  / (freq[$2] * freq[$3]) ),  $2, $3, $1, freq[$2], freq[$3]}' > /tmp/scored-phrases.txt

# filter out infrequent phrases
cat /tmp/scored-phrases.txt | awk '$4 > 5 {print}' | sort -nr
