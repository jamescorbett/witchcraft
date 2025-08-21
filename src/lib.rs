use napi_derive::napi;

use std::{
    sync::{mpsc, OnceLock},
    thread::{self, JoinHandle},
};

type Job = (String, String, String);

#[derive(Debug)]
pub struct Indexer {
    tx: mpsc::Sender<Job>,
    db_name: String,
    _handle: JoinHandle<()>,
}

static INDEXER: OnceLock<Indexer> = OnceLock::new();

impl Indexer {
    pub fn new(db_name: String) -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let db = warp::DB::new(&db_name);
        let handle = thread::spawn(move || {
            let device = warp::make_device();
            while let Ok(job) = rx.recv() {
                let (command, arg1, arg2) = job;
                if command == "add" {
                    warp::add_doc_from_string(&db, &arg1, &arg2).unwrap();
                } else if command == "index" {
                    let count = warp::count_unindexed_chunks(&db).unwrap();
                    println!("count {}", count);
                    if count >= 2048 {
                        warp::index_chunks(&db, &device).unwrap();
                    }
                } else if command == "embed" {
                    warp::embed_chunks(&db, &device).unwrap();
                }
            }
        });
        Indexer {
            tx,
            db_name: db_name.to_string(),
            _handle: handle,
        }
    }
    pub fn init_global(db_name: String) -> &'static Self {
        INDEXER.get_or_init(|| Indexer::new(db_name))
    }

    pub fn global() -> &'static Self {
        INDEXER.get().expect("Indexer not initialized. Call `Indexer::init_global()` first.")
    }
}

mod warp;

#[napi(js_name = "Warp")]
pub struct Warp {
    db: warp::DB,
    embedder: warp::Embedder,
    cache: warp::EmbeddingsCache,
}

#[napi]
impl Warp {
    #[napi(constructor)]
    pub fn new(db_name: String) -> Self {
        let indexer = Indexer::init_global(db_name);
        let db = warp::DB::new_reader(&indexer.db_name);
        let device = warp::make_device();
        let embedder = warp::Embedder::new(&device);
        let cache = warp::EmbeddingsCache::new(16);

        Self {
            db: db,
            embedder: embedder,
            cache: cache,
        }
    }
    #[napi]
    pub fn search(&mut self, q: String, threshold: f64, top_k: u32, sql_filter: String) -> Vec<(String, String)> {

        let filter = if !sql_filter.is_empty() {
            Some(sql_filter.as_str())
        } else {
            None
        };
        match warp::search(&self.db, &self.embedder, &mut self.cache, &q, threshold as f32, top_k as usize, true, filter) {
            Ok(v) => v,
            Err(e) => {
                println!("error {} querying for {}", e, &q);
                [].to_vec()
            }
        }
    }

    #[napi]
    pub fn score(&mut self, q: String, sentences: Vec<String>) -> Vec<f32> {
        warp::score_query_sentences(&self.embedder, &mut self.cache, &q, &sentences).unwrap()
    }

    #[napi]
    pub fn add(&self, metadata: String, body: String) {
        Indexer::global().tx.send(("add".to_string(), metadata, body)).unwrap();
    }

    #[napi]
    pub fn embed(&self) {
        Indexer::global().tx.send(("embed".to_string(), "".to_string(), "".to_string())).unwrap();
    }

    #[napi]
    pub fn index(&self) {
        Indexer::global().tx.send(("index".to_string(), "".to_string(), "".to_string())).unwrap();
    }
}
