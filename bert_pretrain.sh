wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget \
--quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1-tGpU4MxEDx1EL3ZbFYwqYNc971neINf' -O- \
| sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-tGpU4MxEDx1EL3ZbFYwqYNc971neINf" \
-O wiki-ne.txt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget \
--quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1gRAbXRBUhjwteJSn7dboRyPRVSS07LRK' -O- \
| sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gRAbXRBUhjwteJSn7dboRyPRVSS07LRK" \
-O extened_bert.zip && rm -rf /tmp/cookies.txt

unzip extened_bert.zip

rm -rf extened_bert.zip

mkdir saved_models

cd bert_pretrain

python bert_mlm_nsp.py \
--wiki_file ../wiki-ne.txt \
--mlm_output_file_path mlm_file_ne.txt \
--nsp_output_file_path nsp_file_ne.txt \
--log_file log_mlm_nsp.txt \
--model_path ../extened_bert/ \
--iterations 2400000 \
--save_after 200000 \
--backprop_after 8 \
--batch_size 2 \
--learning_rate 0.0001 \
--warmup_steps 40000 \
--model_save_path ../saved_models/ \
--device cuda

