#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget \
#--quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
#'https://docs.google.com/uc?export=download&id=1-tGpU4MxEDx1EL3ZbFYwqYNc971neINf' -O- \
#| sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-tGpU4MxEDx1EL3ZbFYwqYNc971neINf" \
#-O wiki-en.txt && rm -rf /tmp/cookies.txt

#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget \
#--quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
#'https://docs.google.com/uc?export=download&id=1DoHbi6rQ_qPGP9fSEupuDj3TQjlLKEmt' -O- \
#| sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1DoHbi6rQ_qPGP9fSEupuDj3TQjlLKEmt" \
#-O extened_bert.zip && rm -rf /tmp/cookies.txt

#unzip extened_bert.zip

#rm -rf extened_bert.zip

mkdir saved_models

python bert_mlm_nsp.py


