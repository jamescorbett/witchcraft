env/bin/activate:
	python3 -m venv env

env/bin/transformers-cli: env/bin/activate
	(source env/bin/activate; pip install -r requirements.txt)

assets/config.json.zst assets/tokenizer.json.zst xtr.safetensors assets/xtr.safetensors.zst: env/bin/transformers-cli
	(source env/bin/activate; python downloadweights.py)

xtr.gguf: xtr.safetensors
	cargo run --release --bin quantize-tool xtr.safetensors xtr.gguf

assets/xtr.gguf.zst: xtr.gguf
	zstd -19 -f xtr.gguf -o assets/xtr.gguf.zst

download: assets/config.json.zst assets/tokenizer.json.zst assets/xtr.safetensors.zst assets/xtr.gguf.zst

build: download
	cargo build --release --features accelerate
	ln -vf target/release/libwarp.dylib target/release/warp.node

buildemb: download
	cargo build --release --features accelerate,embed-assets
	ln -vf target/release/libwarp.dylib target/release/warp.node

win: download
	cargo xwin build --release --target x86_64-pc-windows-msvc --features embed-assets

run: build
	node index.js

mcp: buildemb
	yarn build
	cmcp "yarn start" tools/call name=search 'arguments:={"q": "teenagers and acne" }'
