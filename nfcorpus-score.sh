rm -rf output.txt

export RUN="cargo run --bin warp-cli --release --features accelerate,t5-quantized,progress"

rm -f mydb.sqlite*
$RUN readcsv datasets/nfcorpus.tsv
$RUN embed
$RUN index

$RUN querycsv $HOME/src/xtr-warp/beir/nfcorpus/questions.test.tsv output.txt &&\

python score.py output.txt $HOME/src/xtr-warp/beir/nfcorpus/collection_map.json $HOME/src/xtr-warp/beir/nfcorpus/qrels.test.json
