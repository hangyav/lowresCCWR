if [ -f "wiki-ne.txt" ]; then
    echo "wiki-ne.txt exists! Skipping download."
else
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget \
    --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
    'https://docs.google.com/uc?export=download&id=1-tGpU4MxEDx1EL3ZbFYwqYNc971neINf' -O- \
    | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-tGpU4MxEDx1EL3ZbFYwqYNc971neINf" \
    -O wiki-ne.txt && rm -rf /tmp/cookies.txt
fi

if [ -d "extened_bert" ]; then
    echo "extened_bert directory exists! Skipping download."
else
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget \
    --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
    'https://docs.google.com/uc?export=download&id=1gRAbXRBUhjwteJSn7dboRyPRVSS07LRK' -O- \
    | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gRAbXRBUhjwteJSn7dboRyPRVSS07LRK" \
    -O extened_bert.zip && rm -rf /tmp/cookies.txt

    unzip extened_bert.zip

    rm -rf extened_bert.zip
fi

mkdir -p saved_models

pushd bert_pretrain

backprop_after=8
if [ $# -ge 1 ]; then
    backprop_after=$1
fi

python bert_mlm_nsp.py \
--wiki_file ../wiki-ne.txt \
--mlm_output_file_path mlm_file_ne.txt \
--nsp_output_file_path nsp_file_ne.txt \
--log_file log_mlm_nsp.txt \
--model_path ../extened_bert/ \
--iterations 2400000 \
--save_after 200000 \
--backprop_after $backprop_after \
--batch_size 2 \
--learning_rate 0.0001 \
--warmup_steps 40000 \
--model_save_path ../saved_models/ \
--device cuda

popd
