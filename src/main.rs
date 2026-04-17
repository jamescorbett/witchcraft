use anyhow::Result;
use log::info;
use std::path::PathBuf;
use witchcraft::*;

struct SimpleLogger;

impl log::Log for SimpleLogger {
    fn enabled(&self, _metadata: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        println!("{}: {}", record.level(), record.args());
    }

    fn flush(&self) {}
}

static LOGGER: SimpleLogger = SimpleLogger;

fn get_assets_path() -> PathBuf {
    std::env::var("WARP_ASSETS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./assets"))
}

#[tokio::main]
async fn main() -> Result<()> {
    log::set_logger(&LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Info);

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: warp-cli <command> [args...]");
        eprintln!("Commands: embed, index, query <text>, hybrid <text>, clear, score <query> <sentences...>");
        std::process::exit(1);
    }

    let db_path = PathBuf::from("mydb.lance");
    let assets = get_assets_path();
    let schema = MetadataSchema::new();

    match args[1].as_str() {
        "readcsv" => {
            if args.len() < 3 {
                eprintln!("Usage: warp-cli readcsv <file.tsv>");
                std::process::exit(1);
            }
            let csvname = PathBuf::from(&args[2]);
            let mut wc = Witchcraft::new(&db_path, &assets, schema).await?;
            read_csv(&mut wc, csvname).await?;
        }
        "index" => {
            let mut wc = Witchcraft::new(&db_path, &assets, schema).await?;
            wc.build_index().await?;
        }
        "query" | "hybrid" => {
            if args.len() < 3 {
                eprintln!("Usage: warp-cli {} <text>", args[1]);
                std::process::exit(1);
            }
            let use_fulltext = args[1] == "hybrid";
            let query_text = args[2..].join(" ");
            let mut wc = Witchcraft::new(&db_path, &assets, schema).await?;
            let results = wc.search(&query_text, 0.7, 10, use_fulltext, None).await?;

            for (i, r) in results.iter().enumerate() {
                println!(
                    "#{} score={:.4} date={} sub_idx={}",
                    i + 1,
                    r.score,
                    r.date,
                    r.matched_sub_idx
                );
                let idx = r.matched_sub_idx as usize;
                if idx < r.bodies.len() {
                    let preview: String = r.bodies[idx].chars().take(200).collect();
                    println!("  {}", preview);
                }
                println!();
            }
        }
        "score" => {
            if args.len() < 4 {
                eprintln!("Usage: warp-cli score <query> <sentence1> [sentence2] ...");
                std::process::exit(1);
            }
            let device = make_device();
            let embedder = Embedder::new(&device, &assets)?;
            let mut cache = EmbeddingsCache::new(1);
            let q = args[2].clone();
            let sentences: Vec<String> = args[3..].iter().cloned().collect();
            let scores =
                score_query_sentences(&embedder, &mut cache, &q, &sentences)?;
            for (s, score) in sentences.iter().zip(scores.iter()) {
                println!("{:.4} {}", score, s);
            }
        }
        "clear" => {
            let mut wc = Witchcraft::new(&db_path, &assets, schema).await?;
            wc.clear().await?;
            info!("Database cleared.");
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Read a TSV file and add each record as a document.
async fn read_csv(wc: &mut Witchcraft, csvname: PathBuf) -> Result<()> {
    use csv::ReaderBuilder;
    use std::collections::HashMap;
    use uuid::Uuid;

    #[derive(serde::Deserialize)]
    #[allow(dead_code)]
    struct CSVRecord {
        name: String,
        body: String,
    }

    let namespace = Uuid::from_bytes([
        0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4,
        0x30, 0xc8,
    ]);

    let mut reader = ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(&csvname)?;

    let mut count = 0;
    for result in reader.deserialize() {
        let record: CSVRecord = result?;
        let chunks: Vec<String> = text_splitter::TextSplitter::new(300)
            .chunks(&record.body)
            .map(|s| s.to_string())
            .collect();
        let body = chunks.join("");
        let lens: Vec<usize> = chunks.iter().map(|c| c.chars().count()).collect();
        let uuid = Uuid::new_v5(&namespace, body.as_bytes());

        wc.add_document(&uuid, None, HashMap::new(), &body, Some(lens))
            .await?;
        count += 1;
    }
    info!("read {} records from {:?}", count, csvname);
    Ok(())
}
