 <$1 tr -cd "[:alpha:][:space:]-'" | tr ' [:upper:]' '\n[:lower:]' | tr -s '\n' | sed "s/^['-]*//;s/['-]$//" | sort | uniq -c > $1.words
awk '{print $2}' $1.words >$1.vocab