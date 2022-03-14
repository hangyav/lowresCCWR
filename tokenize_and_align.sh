
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < en-sw/en-sw.en > en-sw/en-sw.en.token
mosesdecoder/scripts/tokenizer/tokenizer.perl -q --no-escape -l sw < en-sw/en-sw.sw > en-sw/en-sw.sw.token
python fastalign_data.py --src_file en-sw/en-sw.sw.token --tgt_file en-sw/en-sw.en.token --output_file en-sw/sw-en.tokens
sed 's/( .. ) //g' en-sw/sw-en.tokens | grep -F -v "\\" | grep -P -v "[\xAD]" | sed "s/&apos;/'/g" | sed 's/&quot;/"/g' | sed 's/&#91;/[/g' | sed 's/&#93;/]/g' | sed 's/&amp;/\&/g' | grep -v "=" | grep -v "#" | grep -v "http" | grep -v "&gt;" | grep -v "&lt;" | sed '/^ |||/ d' | sed '/||| $/d' | sed '/^|||/ d' | sed '/|||$/d' > en-sw/sw-en.tokens.clean

fast_align/build/fast_align -i en-sw/sw-en.tokens.clean -d -o -v > en-sw/sw-en.align
fast_align/build/fast_align -i en-sw/sw-en.tokens.clean -d -o -v -r > en-sw/sw-en.reverse.align
fast_align/build/atools -i en-sw/sw-en.align -j en-sw/sw-en.reverse.align -c intersect > en-sw/sw-en.alignment

mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < en-sd/en-sd.en > en-sd/en-sd.en.token
mosesdecoder/scripts/tokenizer/tokenizer.perl -q --no-escape -l sd < en-sd/en-sd.sd > en-sd/en-sd.sd.token
python fastalign_data.py --src_file en-sd/en-sd.sd.token --tgt_file en-sd/en-sd.en.token --output_file en-sd/sd-en.tokens
sed 's/( .. ) //g' en-sd/sd-en.tokens | grep -F -v "\\" | grep -P -v "[\xAD]" | sed "s/&apos;/'/g" | sed 's/&quot;/"/g' | sed 's/&#91;/[/g' | sed 's/&#93;/]/g' | sed 's/&amp;/\&/g' | grep -v "=" | grep -v "#" | grep -v "http" | grep -v "&gt;" | grep -v "&lt;" | sed '/^ |||/ d' | sed '/||| $/d' | sed '/^|||/ d' | sed '/|||$/d' > en-sd/sd-en.tokens.clean

fast_align/build/fast_align -i en-sd/sd-en.tokens.clean -d -o -v > en-sd/sd-en.align
fast_align/build/fast_align -i en-sd/sd-en.tokens.clean -d -o -v -r > en-sd/sd-en.reverse.align
fast_align/build/atools -i en-sd/sd-en.align -j en-sd/sd-en.reverse.align -c intersect > en-sd/sd-en.alignment

mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < en-mi/en-mi.en > en-mi/en-mi.en.token
mosesdecoder/scripts/tokenizer/tokenizer.perl -q --no-escape -l mi < en-mi/en-mi.mi > en-mi/en-mi.mi.token
python fastalign_data.py --src_file en-mi/en-mi.mi.token --tgt_file en-mi/en-mi.en.token --output_file en-mi/mi-en.tokens
sed 's/( .. ) //g' en-mi/mi-en.tokens | grep -F -v "\\" | grep -P -v "[\xAD]" | sed "s/&apos;/'/g" | sed 's/&quot;/"/g' | sed 's/&#91;/[/g' | sed 's/&#93;/]/g' | sed 's/&amp;/\&/g' | grep -v "=" | grep -v "#" | grep -v "http" | grep -v "&gt;" | grep -v "&lt;" | sed '/^ |||/ d' | sed '/||| $/d' | sed '/^|||/ d' | sed '/|||$/d' > en-mi/mi-en.tokens.clean

fast_align/build/fast_align -i en-mi/mi-en.tokens.clean -d -o -v > en-mi/mi-en.align
fast_align/build/fast_align -i en-mi/mi-en.tokens.clean -d -o -v -r > en-mi/mi-en.reverse.align
fast_align/build/atools -i en-mi/mi-en.align -j en-mi/mi-en.reverse.align -c intersect > en-mi/mi-en.alignment

mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < en-ml/en-ml.en > en-ml/en-ml.en.token
python tokenize_data.py --input_file en-ml/en-ml.ml --output_file en-ml/en-ml.ml.token --language ml
python fastalign_data.py --src_file en-ml/en-ml.ml.token --tgt_file en-ml/en-ml.en.token --output_file en-ml/ml-en.tokens
sed 's/( .. ) //g' en-ml/ml-en.tokens | grep -F -v "\\" | grep -P -v "[\xAD]" | sed "s/&apos;/'/g" | sed 's/&quot;/"/g' | sed 's/&#91;/[/g' | sed 's/&#93;/]/g' | sed 's/&amp;/\&/g' | grep -v "=" | grep -v "#" | grep -v "http" | grep -v "&gt;" | grep -v "&lt;" | sed '/^ |||/ d' | sed '/||| $/d' | sed '/^|||/ d' | sed '/|||$/d' > en-ml/ml-en.tokens.clean

fast_align/build/fast_align -i en-ml/ml-en.tokens.clean -d -o -v > en-ml/ml-en.align
fast_align/build/fast_align -i en-ml/ml-en.tokens.clean -d -o -v -r > en-ml/ml-en.reverse.align
fast_align/build/atools -i en-ml/ml-en.align -j en-ml/ml-en.reverse.align -c intersect > en-ml/ml-en.alignment

mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < en-am/en-am.en > en-am/en-am.en.token
python tokenize_data.py --input_file en-am/en-am.am --output_file en-am/en-am.am.token --language am
python fastalign_data.py --src_file en-am/en-am.am.token --tgt_file en-am/en-am.en.token --output_file en-am/am-en.tokens
sed 's/( .. ) //g' en-am/am-en.tokens | grep -F -v "\\" | grep -P -v "[\xAD]" | sed "s/&apos;/'/g" | sed 's/&quot;/"/g' | sed 's/&#91;/[/g' | sed 's/&#93;/]/g' | sed 's/&amp;/\&/g' | grep -v "=" | grep -v "#" | grep -v "http" | grep -v "&gt;" | grep -v "&lt;" | sed '/^ |||/ d' | sed '/||| $/d' | sed '/^|||/ d' | sed '/|||$/d' > en-am/am-en.tokens.clean

fast_align/build/fast_align -i en-am/am-en.tokens.clean -d -o -v > en-am/am-en.align
fast_align/build/fast_align -i en-am/am-en.tokens.clean -d -o -v -r > en-am/am-en.reverse.align
fast_align/build/atools -i en-am/am-en.align -j en-am/am-en.reverse.align -c intersect > en-am/am-en.alignment

mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < en-si/en-si.en > en-si/en-si.en.token
python tokenize_data.py --input_file en-si/en-si.si --output_file en-si/en-si.si.token --language si
python fastalign_data.py --src_file en-si/en-si.si.token --tgt_file en-si/en-si.en.token --output_file en-si/si-en.tokens
sed 's/( .. ) //g' en-si/si-en.tokens | grep -F -v "\\" | grep -P -v "[\xAD]" | sed "s/&apos;/'/g" | sed 's/&quot;/"/g' | sed 's/&#91;/[/g' | sed 's/&#93;/]/g' | sed 's/&amp;/\&/g' | grep -v "=" | grep -v "#" | grep -v "http" | grep -v "&gt;" | grep -v "&lt;" | sed '/^ |||/ d' | sed '/||| $/d' | sed '/^|||/ d' | sed '/|||$/d' > en-si/si-en.tokens.clean

fast_align/build/fast_align -i en-si/si-en.tokens.clean -d -o -v > en-si/si-en.align
fast_align/build/fast_align -i en-si/si-en.tokens.clean -d -o -v -r > en-si/si-en.reverse.align
fast_align/build/atools -i en-si/si-en.align -j en-si/si-en.reverse.align -c intersect > en-si/si-en.alignment


mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < en-af/en-af.en > en-af/en-af.en.token
mosesdecoder/scripts/tokenizer/tokenizer.perl -q --no-escape -l af < en-af/en-af.af > en-af/en-af.af.token
python fastalign_data.py --src_file en-af/en-af.af.token --tgt_file en-af/en-af.en.token --output_file en-af/af-en.tokens
sed 's/( .. ) //g' en-af/af-en.tokens | grep -F -v "\\" | grep -P -v "[\xAD]" | sed "s/&apos;/'/g" | sed 's/&quot;/"/g' | sed 's/&#91;/[/g' | sed 's/&#93;/]/g' | sed 's/&amp;/\&/g' | grep -v "=" | grep -v "#" | grep -v "http" | grep -v "&gt;" | grep -v "&lt;" | sed '/^ |||/ d' | sed '/||| $/d' | sed '/^|||/ d' | sed '/|||$/d' > en-af/af-en.tokens.clean

fast_align/build/fast_align -i en-af/af-en.tokens.clean -d -o -v > en-af/af-en.align
fast_align/build/fast_align -i en-af/af-en.tokens.clean -d -o -v -r > en-af/af-en.reverse.align
fast_align/build/atools -i en-af/af-en.align -j en-af/af-en.reverse.align -c intersect > en-af/af-en.alignment


mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < en-mk/en-mk.en > en-mk/en-mk.en.token
mosesdecoder/scripts/tokenizer/tokenizer.perl -q --no-escape -l mk < en-mk/en-mk.mk > en-mk/en-mk.mk.token
python fastalign_data.py --src_file en-mk/en-mk.mk.token --tgt_file en-mk/en-mk.en.token --output_file en-mk/mk-en.tokens
sed 's/( .. ) //g' en-mk/mk-en.tokens | grep -F -v "\\" | grep -P -v "[\xAD]" | sed "s/&apos;/'/g" | sed 's/&quot;/"/g' | sed 's/&#91;/[/g' | sed 's/&#93;/]/g' | sed 's/&amp;/\&/g' | grep -v "=" | grep -v "#" | grep -v "http" | grep -v "&gt;" | grep -v "&lt;" | sed '/^ |||/ d' | sed '/||| $/d' | sed '/^|||/ d' | sed '/|||$/d' > en-mk/mk-en.tokens.clean

fast_align/build/fast_align -i en-mk/mk-en.tokens.clean -d -o -v > en-mk/mk-en.align
fast_align/build/fast_align -i en-mk/mk-en.tokens.clean -d -o -v -r > en-mk/mk-en.reverse.align
fast_align/build/atools -i en-mk/mk-en.align -j en-mk/mk-en.reverse.align -c intersect > en-mk/mk-en.alignment

mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < en-eu/en-eu.en > en-eu/en-eu.en.token
mosesdecoder/scripts/tokenizer/tokenizer.perl -q --no-escape -l eu < en-eu/en-eu.eu > en-eu/en-eu.eu.token
python fastalign_data.py --src_file en-eu/en-eu.eu.token --tgt_file en-eu/en-eu.en.token --output_file en-eu/eu-en.tokens
sed 's/( .. ) //g' en-eu/eu-en.tokens | grep -F -v "\\" | grep -P -v "[\xAD]" | sed "s/&apos;/'/g" | sed 's/&quot;/"/g' | sed 's/&#91;/[/g' | sed 's/&#93;/]/g' | sed 's/&amp;/\&/g' | grep -v "=" | grep -v "#" | grep -v "http" | grep -v "&gt;" | grep -v "&lt;" | sed '/^ |||/ d' | sed '/||| $/d' | sed '/^|||/ d' | sed '/|||$/d' > en-eu/eu-en.tokens.clean

fast_align/build/fast_align -i en-eu/eu-en.tokens.clean -d -o -v > en-eu/eu-en.align
fast_align/build/fast_align -i en-eu/eu-en.tokens.clean -d -o -v -r > en-eu/eu-en.reverse.align
fast_align/build/atools -i en-eu/eu-en.align -j en-eu/eu-en.reverse.align -c intersect > en-eu/eu-en.alignment

mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < en-bg/en-bg.en > en-bg/en-bg.en.token
mosesdecoder/scripts/tokenizer/tokenizer.perl -q --no-escape -l bg < en-bg/en-bg.bg > en-bg/en-bg.bg.token
python fastalign_data.py --src_file en-bg/en-bg.bg.token --tgt_file en-bg/en-bg.en.token --output_file en-bg/bg-en.tokens
sed 's/( .. ) //g' en-bg/bg-en.tokens | grep -F -v "\\" | grep -P -v "[\xAD]" | sed "s/&apos;/'/g" | sed 's/&quot;/"/g' | sed 's/&#91;/[/g' | sed 's/&#93;/]/g' | sed 's/&amp;/\&/g' | grep -v "=" | grep -v "#" | grep -v "http" | grep -v "&gt;" | grep -v "&lt;" | sed '/^ |||/ d' | sed '/||| $/d' | sed '/^|||/ d' | sed '/|||$/d' > en-bg/bg-en.tokens.clean

fast_align/build/fast_align -i en-bg/bg-en.tokens.clean -d -o -v > en-bg/bg-en.align
fast_align/build/fast_align -i en-bg/bg-en.tokens.clean -d -o -v -r > en-bg/bg-en.reverse.align
fast_align/build/atools -i en-bg/bg-en.align -j en-bg/bg-en.reverse.align -c intersect > en-bg/bg-en.alignment

mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < en-gu/en-gu.en > en-gu/en-gu.en.token
python tokenize_data.py --input_file en-gu/en-gu.gu --output_file en-gu/en-gu.gu.token --language gu
python fastalign_data.py --src_file en-gu/en-gu.gu.token --tgt_file en-gu/en-gu.en.token --output_file en-gu/gu-en.tokens
sed 's/( .. ) //g' en-gu/gu-en.tokens | grep -F -v "\\" | grep -P -v "[\xAD]" | sed "s/&apos;/'/g" | sed 's/&quot;/"/g' | sed 's/&#91;/[/g' | sed 's/&#93;/]/g' | sed 's/&amp;/\&/g' | grep -v "=" | grep -v "#" | grep -v "http" | grep -v "&gt;" | grep -v "&lt;" | sed '/^ |||/ d' | sed '/||| $/d' | sed '/^|||/ d' | sed '/|||$/d' > en-gu/gu-en.tokens.clean

fast_align/build/fast_align -i en-gu/gu-en.tokens.clean -d -o -v > en-gu/gu-en.align
fast_align/build/fast_align -i en-gu/gu-en.tokens.clean -d -o -v -r > en-gu/gu-en.reverse.align
fast_align/build/atools -i en-gu/gu-en.align -j en-gu/gu-en.reverse.align -c intersect > en-gu/gu-en.alignment

mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < en-kn/en-kn.en > en-kn/en-kn.en.token
python tokenize_data.py --input_file en-kn/en-kn.kn --output_file en-kn/en-kn.kn.token --language kn
python fastalign_data.py --src_file en-kn/en-kn.kn.token --tgt_file en-kn/en-kn.en.token --output_file en-kn/kn-en.tokens
sed 's/( .. ) //g' en-kn/kn-en.tokens | grep -F -v "\\" | grep -P -v "[\xAD]" | sed "s/&apos;/'/g" | sed 's/&quot;/"/g' | sed 's/&#91;/[/g' | sed 's/&#93;/]/g' | sed 's/&amp;/\&/g' | grep -v "=" | grep -v "#" | grep -v "http" | grep -v "&gt;" | grep -v "&lt;" | sed '/^ |||/ d' | sed '/||| $/d' | sed '/^|||/ d' | sed '/|||$/d' > en-kn/kn-en.tokens.clean

fast_align/build/fast_align -i en-kn/kn-en.tokens.clean -d -o -v > en-kn/kn-en.align
fast_align/build/fast_align -i en-kn/kn-en.tokens.clean -d -o -v -r > en-kn/kn-en.reverse.align
fast_align/build/atools -i en-kn/kn-en.align -j en-kn/kn-en.reverse.align -c intersect > en-kn/kn-en.alignment

#python train_test_val.py --tokens_file en-sw/sw-en.tokens.clean --alignment_file en-sw/sw-en.alignment --language sw
#python train_test_val.py --tokens_file en-sd/sd-en.tokens.clean --alignment_file en-sd/sd-en.alignment --language sd
#python train_test_val.py --tokens_file en-si/si-en.tokens.clean --alignment_file en-si/si-en.alignment --language si
#python train_test_val.py --tokens_file en-mi/mi-en.tokens.clean --alignment_file en-mi/mi-en.alignment --language mi
#python train_test_val.py --tokens_file en-ml/ml-en.tokens.clean --alignment_file en-ml/ml-en.alignment --language ml
#python train_test_val.py --tokens_file en-am/am-en.tokens.clean --alignment_file en-am/am-en.alignment --language am
