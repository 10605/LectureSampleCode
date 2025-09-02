# word frequency
tr -sc '[:alpha:]' '\n' < data/bluecorpus.txt | sort | uniq -c > /tmp/bluehist.txt

# all bigrams
tail -n +2  /tmp/bluewords.txt  > /tmp/bluenextwords.txt
paste /tmp/bluewords.txt /tmp/bluenextwords.txt | sort | uniq -c > /tmp/bluebigrams.txt

# score them by log (N & freq[xy] / freq[x]*freq[y])
N="`wc -l < /tmp/bluewords.txt`"

cat /tmp/bluehist.txt /tmp/bluebigrams.txt |  awk \
    'NF == 2 { freq[$2] = $1}
     NF == 3 { print log( N * $1  / (freq[$2] * freq[$3]) ),  $2, $3, $1, freq[$2], freq[$3]}' N=$$N > /tmp/scored-phrases.txt

# filter out infrequent phrases
cat /tmp/scored-phrases.txt | awk '$4 > 5 {print}' | sort -nr
