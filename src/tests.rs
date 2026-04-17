use crate::*;
use std::collections::HashMap;
use uuid::Uuid;

const NAMESPACE: Uuid = Uuid::from_bytes([
    0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30,
    0xc8,
]);

const FACTS: &[&str] = &[
    "The shortest war in history lasted 38 minutes between Britain and Zanzibar.",
    "Honey never spoils; archaeologists have found 3000-year-old honey in Egyptian tombs that was still edible.",
    "Octopuses have three hearts and blue blood.",
    "The Eiffel Tower can grow up to 6 inches taller during the summer due to thermal expansion.",
    "Bananas are berries, but strawberries are not.",
    "A group of flamingos is called a 'flamboyance'.",
    "The longest hiccuping spree lasted 68 years.",
    "Venus is the only planet that spins clockwise.",
    "Cleopatra lived closer in time to the Moon landing than to the construction of the Great Pyramid.",
    "A teaspoonful of neutron star material would weigh about 6 billion tons.",
    "There are more possible iterations of a game of chess than there are atoms in the known universe.",
    "The shortest complete sentence in English is 'I am.'",
    "The total weight of all ants on Earth is roughly equal to the total weight of all humans.",
    "Scotland's national animal is the unicorn.",
    "A day on Venus is longer than a year on Venus.",
    "Cows have best friends and get stressed when separated.",
    "The human nose can detect over 1 trillion different scents.",
    "The inventor of the Pringles can is buried in one.",
    "There are more stars in the universe than grains of sand on all of Earth's beaches.",
    "A bolt of lightning is five times hotter than the surface of the sun.",
    "The fingerprints of a koala are so similar to humans that they can taint crime scenes.",
    "An octopus has a donut-shaped brain.",
    "The longest wedding veil was longer than 63 football fields.",
    "A jiffy is an actual unit of time: 1/100th of a second.",
    "The moon has moonquakes.",
    "A shrimp's heart is in its head.",
    "It takes a photon about 40,000 years to travel from the core of the sun to its surface, but only 8 minutes to travel from the sun's surface to Earth.",
    "The DNA of humans and bananas is about 60% identical.",
    "Wombat poop is cube-shaped.",
    "The Great Wall of China is not visible from space with the naked eye.",
    "A single strand of spaghetti is called a 'spaghetto.'",
    "Polar bears' skin is actually black under their white fur.",
    "The first computer programmer was a woman named Ada Lovelace.",
];

const THRESHOLD: f32 = 0.7;

fn get_assets_path() -> std::path::PathBuf {
    std::env::var("WARP_ASSETS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("./assets"))
}

fn assets_available(assets: &std::path::Path) -> bool {
    // Check for the actual model files, not just the directory
    assets.join("config.json").exists()
}

#[tokio::test]
async fn test_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("test_db");
    let assets = get_assets_path();
    if !assets_available(&assets) {
        eprintln!("Skipping test: assets not found at {:?}", assets);
        return;
    }

    let schema = MetadataSchema::new();
    let mut wc = Witchcraft::new(&db_path, &assets, schema).await.unwrap();

    // Insert all facts
    for (_i, fact) in FACTS.iter().enumerate() {
        let uuid = Uuid::new_v5(&NAMESPACE, fact.as_bytes());
        wc.add_document(&uuid, Some("2024-01-01"), HashMap::new(), fact, None)
            .await
            .unwrap();
    }

    // Build indexes
    wc.build_index().await.unwrap();

    // Search
    let results = wc
        .search("what was the shortest war ever?", THRESHOLD, 3, true, None)
        .await
        .unwrap();

    assert!(!results.is_empty(), "expected at least one search result");
    // The top result should mention Zanzibar
    let top_body = results[0].bodies.join(" ");
    assert!(
        top_body.contains("Zanzibar") || top_body.contains("shortest war"),
        "expected top result to be about the shortest war, got: {}",
        top_body
    );
}

#[tokio::test]
async fn test_scoring() {
    let assets = get_assets_path();
    if !assets_available(&assets) {
        eprintln!("Skipping test: assets not found at {:?}", assets);
        return;
    }

    let device = make_device();
    let embedder = Embedder::new(&device, &assets).unwrap();
    let mut cache = EmbeddingsCache::new(1);

    let sentences: Vec<String> = vec![
        "The Eiffel Tower is in Paris.".into(),
        "Bananas are a popular fruit.".into(),
        "Octopuses have three hearts.".into(),
        "The shortest war in history lasted 38 minutes.".into(),
        "Honey never spoils.".into(),
        "Venus spins clockwise.".into(),
        "Cows have best friends.".into(),
    ];

    let scores = score_query_sentences(
        &embedder,
        &mut cache,
        &"what was the shortest war ever?".to_string(),
        &sentences,
    )
    .unwrap();

    assert_eq!(scores.len(), 7);
    // The highest score should be for sentence index 3
    let max_idx = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(max_idx, 3, "expected sentence 3 to score highest");

    // Run again to test cache
    let scores2 = score_query_sentences(
        &embedder,
        &mut cache,
        &"what was the shortest war ever?".to_string(),
        &sentences,
    )
    .unwrap();
    assert_eq!(scores.len(), scores2.len());
}

#[tokio::test]
async fn test_remove_document() {
    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("test_remove");
    let assets = get_assets_path();
    if !assets_available(&assets) {
        eprintln!("Skipping test: assets not found at {:?}", assets);
        return;
    }

    let schema = MetadataSchema::new();
    let mut wc = Witchcraft::new(&db_path, &assets, schema).await.unwrap();

    let uuid = Uuid::new_v5(&NAMESPACE, b"test doc");
    wc.add_document(
        &uuid,
        Some("2024-01-01"),
        HashMap::new(),
        "The sky is blue and the grass is green.",
        None,
    )
    .await
    .unwrap();

    // Remove it
    wc.remove_document(&uuid).await.unwrap();

    // Verify it's gone by searching
    let results = wc
        .search("sky is blue", 0.5, 10, false, None)
        .await
        .unwrap();
    assert!(results.is_empty(), "document should have been removed");
}

#[tokio::test]
async fn test_metadata_filter() {
    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("test_filter");
    let assets = get_assets_path();
    if !assets_available(&assets) {
        eprintln!("Skipping test: assets not found at {:?}", assets);
        return;
    }

    let schema = MetadataSchema::new()
        .add_string("source", true);
    let mut wc = Witchcraft::new(&db_path, &assets, schema).await.unwrap();

    // Add two docs with different sources
    let uuid1 = Uuid::new_v5(&NAMESPACE, b"doc1");
    let mut meta1 = HashMap::new();
    meta1.insert("source".to_string(), MetadataValue::String("alpha".into()));
    wc.add_document(&uuid1, Some("2024-01-01"), meta1, "Cats are wonderful pets.", None)
        .await
        .unwrap();

    let uuid2 = Uuid::new_v5(&NAMESPACE, b"doc2");
    let mut meta2 = HashMap::new();
    meta2.insert("source".to_string(), MetadataValue::String("beta".into()));
    wc.add_document(&uuid2, Some("2024-01-01"), meta2, "Dogs are loyal companions.", None)
        .await
        .unwrap();

    // Search with filter for source = 'alpha'
    let filter = Filter::eq("source", MetadataValue::String("alpha".into()));
    let results = wc
        .search("pets", 0.3, 10, false, Some(&filter))
        .await
        .unwrap();

    // Should only get the cats doc
    for r in &results {
        let top = &r.bodies[0];
        assert!(
            top.contains("Cats"),
            "filtered results should only contain alpha source"
        );
    }
}
