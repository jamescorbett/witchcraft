rm -rf output.txt

cargo build --release --features accelerate
time cargo run --release --features accelerate querycsv $HOME/src/xtr-warp/beir/nfcorpus/questions.test.tsv output.txt &&\

python score.py output.txt $HOME/src/xtr-warp/beir/nfcorpus/collection_map.json $HOME/src/xtr-warp/beir/nfcorpus/qrels.test.json
