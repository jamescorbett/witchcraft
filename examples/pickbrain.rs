use anyhow::Result;
use log::{Level, LevelFilter, Metadata, Record};
use std::env;
use std::io::Write;
use std::path::PathBuf;

use warp::{DB, Embedder};

struct SimpleLogger;
impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Warn
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: SimpleLogger = SimpleLogger;

fn db_path() -> PathBuf {
    let home = env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".claude/pickbrain.db")
}

fn assets_path() -> PathBuf {
    PathBuf::from(env::var("WARP_ASSETS").unwrap_or_else(|_| "assets".into()))
}

fn ingest(db_name: &PathBuf) -> Result<bool> {
    let mut db = DB::new(db_name.clone()).unwrap();
    let (sessions, memories, authored) = warp::claude_code::ingest_claude_code(&mut db)?;
    if sessions + memories + authored == 0 {
        return Ok(false);
    }
    eprintln!("ingested {sessions} sessions, {memories} memory files, {authored} authored files");
    Ok(true)
}

fn embed_and_index(db: &DB, embedder: &Embedder, device: &candle_core::Device) -> Result<()> {
    let embedded = warp::embed_chunks(db, embedder, None)?;
    if embedded > 0 {
        eprintln!("embedded {embedded} chunks");
        warp::index_chunks(db, device)?;
        eprintln!("index rebuilt");
    }
    Ok(())
}

// --- Search result data ---

struct SearchResult {
    timestamp: String,
    project: String,
    session_id: String,
    turn: u64,
    path: String,
    bodies: Vec<String>,
    match_idx: usize,
}

fn run_search(
    db_name: &PathBuf,
    assets: &PathBuf,
    q: &str,
    session: Option<&str>,
) -> Result<(Vec<SearchResult>, u128)> {
    use warp::types::*;
    let device = warp::make_device();
    let embedder = warp::Embedder::new(&device, assets)?;

    {
        let db_rw = DB::new(db_name.clone()).unwrap();
        embed_and_index(&db_rw, &embedder, &device)?;
    }

    let mut cache = warp::EmbeddingsCache::new(1);
    let db = DB::new_reader(db_name.clone()).unwrap();
    let sql_filter = session.map(|id| SqlStatementInternal {
        statement_type: SqlStatementType::Condition,
        condition: Some(SqlConditionInternal {
            key: "$.session_id".to_string(),
            operator: SqlOperator::Equals,
            value: Some(SqlValue::String(id.to_string())),
        }),
        logic: None,
        statements: None,
    });
    let now = std::time::Instant::now();
    let results = warp::search(
        &db,
        &embedder,
        &mut cache,
        q,
        0.5,
        10,
        true,
        sql_filter.as_ref(),
    )?;
    let search_ms = now.elapsed().as_millis();

    let out: Vec<SearchResult> = results
        .into_iter()
        .map(|(_score, metadata, bodies, sub_idx, date)| {
            let meta: serde_json::Value = serde_json::from_str(&metadata).unwrap_or_default();
            let idx = (sub_idx as usize).min(bodies.len().saturating_sub(1));
            SearchResult {
                timestamp: format_date(&date),
                project: meta["project"].as_str().unwrap_or("").to_string(),
                session_id: meta["session_id"].as_str().unwrap_or("").to_string(),
                turn: meta["turn"].as_u64().unwrap_or(0),
                path: meta["path"].as_str().unwrap_or("").to_string(),
                bodies,
                match_idx: idx,
            }
        })
        .collect();
    Ok((out, search_ms))
}

// --- TUI ---

enum View {
    List,
    Detail(usize),
}

fn search_tui(
    db_name: &PathBuf,
    assets: &PathBuf,
    q: &str,
    session: Option<&str>,
) -> Result<()> {
    let (results, search_ms) = run_search(db_name, assets, q, session)?;
    if results.is_empty() {
        eprintln!("no results");
        return Ok(());
    }

    use crossterm::event::{self, Event, KeyCode, KeyModifiers};
    use crossterm::terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
    };
    use ratatui::backend::CrosstermBackend;
    use ratatui::layout::{Constraint, Direction, Layout};
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};
    use ratatui::widgets::{ListState, Paragraph, Wrap};
    use ratatui::Terminal;

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut view = View::List;
    let mut selected: usize = 0;
    let mut list_state = ListState::default();
    list_state.select(Some(0));
    let mut scroll_offset: u16 = 0;

    loop {
        terminal.draw(|f| {
            let area = f.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(2), Constraint::Min(0)])
                .split(area);

            // Header
            let header = Paragraph::new(Line::from(vec![
                Span::styled(
                    format!("[[ {q} ]]"),
                    Style::default()
                        .fg(Color::Rgb(0, 255, 0))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  {search_ms} ms  "),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    match view {
                        View::List => "↑↓ navigate  ⏎ open  q quit".to_string(),
                        View::Detail(_) => "↑↓ scroll  esc back  q quit".to_string(),
                    },
                    Style::default().fg(Color::DarkGray),
                ),
            ]));
            f.render_widget(header, chunks[0]);

            match view {
                View::List => {
                    let width = chunks[1].width as usize;
                    let items: Vec<ratatui::widgets::ListItem> = results
                        .iter()
                        .map(|r| {
                            let preview = first_line(&r.bodies[r.match_idx]);
                            let mut meta_spans = vec![
                                Span::styled(
                                    format!("{} ", r.timestamp),
                                    Style::default().fg(Color::Green),
                                ),
                                Span::styled(&r.project, Style::default().fg(Color::Cyan)),
                            ];
                            if r.path.ends_with(".md") {
                                meta_spans.push(Span::styled(
                                    format!("  {}", r.path),
                                    Style::default().fg(Color::Yellow),
                                ));
                            }
                            if !r.session_id.is_empty() {
                                meta_spans.push(Span::styled(
                                    format!("  turn {}", r.turn),
                                    Style::default().fg(Color::DarkGray),
                                ));
                            }
                            ratatui::widgets::ListItem::new(vec![
                                Line::from(meta_spans),
                                Line::styled(
                                    format!("  {}", truncate(&preview, width.saturating_sub(4))),
                                    Style::default(),
                                ),
                                Line::from(""),
                            ])
                        })
                        .collect();

                    let list = ratatui::widgets::List::new(items).highlight_style(
                        Style::default()
                            .bg(Color::DarkGray)
                            .add_modifier(Modifier::BOLD),
                    );
                    f.render_stateful_widget(list, chunks[1], &mut list_state);
                }
                View::Detail(idx) => {
                    let r = &results[idx];
                    let mut lines: Vec<Line> = Vec::new();

                    // Session header
                    lines.push(Line::from(vec![
                        Span::styled(
                            format!("{} ", r.timestamp),
                            Style::default().fg(Color::Green),
                        ),
                        Span::styled(&r.project, Style::default().fg(Color::Cyan)),
                    ]));
                    if !r.session_id.is_empty() {
                        lines.push(Line::from(vec![
                            Span::styled(&r.session_id, Style::default().fg(Color::Magenta)),
                            Span::styled(
                                format!("  turn {}", r.turn),
                                Style::default().fg(Color::DarkGray),
                            ),
                        ]));
                    }
                    lines.push(Line::from(""));

                    for (i, chunk) in r.bodies.iter().enumerate() {
                        let style = if i == r.match_idx {
                            Style::default().add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(Color::DarkGray)
                        };
                        for line in chunk.lines().filter(|l| !l.is_empty()) {
                            lines.push(Line::styled(format!("  {line}"), style));
                        }
                        lines.push(Line::from(""));
                    }

                    let detail = Paragraph::new(lines)
                        .wrap(Wrap { trim: false })
                        .scroll((scroll_offset, 0));
                    f.render_widget(detail, chunks[1]);
                }
            }
        })?;

        if let Event::Key(key) = event::read()? {
            match (&view, key.code, key.modifiers) {
                (_, KeyCode::Char('q'), _) => break,
                (_, KeyCode::Char('c'), KeyModifiers::CONTROL) => break,

                // List view
                (View::List, KeyCode::Down | KeyCode::Char('j'), _) => {
                    if selected + 1 < results.len() {
                        selected += 1;
                        list_state.select(Some(selected));
                    }
                }
                (View::List, KeyCode::Up | KeyCode::Char('k'), _) => {
                    if selected > 0 {
                        selected -= 1;
                        list_state.select(Some(selected));
                    }
                }
                (View::List, KeyCode::Enter, _) => {
                    scroll_offset = 0;
                    view = View::Detail(selected);
                }

                // Detail view
                (View::Detail(_), KeyCode::Esc, _) => {
                    view = View::List;
                }
                (View::Detail(_), KeyCode::Down | KeyCode::Char('j'), _) => {
                    scroll_offset = scroll_offset.saturating_add(1);
                }
                (View::Detail(_), KeyCode::Up | KeyCode::Char('k'), _) => {
                    scroll_offset = scroll_offset.saturating_sub(1);
                }
                (View::Detail(_), KeyCode::PageDown | KeyCode::Char(' '), _) => {
                    scroll_offset = scroll_offset.saturating_add(20);
                }
                (View::Detail(_), KeyCode::PageUp, _) => {
                    scroll_offset = scroll_offset.saturating_sub(20);
                }
                _ => {}
            }
        }
    }

    disable_raw_mode()?;
    crossterm::execute!(std::io::stdout(), LeaveAlternateScreen)?;
    Ok(())
}

fn first_line(text: &str) -> String {
    text.lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .to_string()
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max.saturating_sub(3)])
    }
}

// --- Plain text fallback (piped output) ---

fn search_plain(
    db_name: &PathBuf,
    assets: &PathBuf,
    q: &str,
    session: Option<&str>,
) -> Result<()> {
    let (results, search_ms) = run_search(db_name, assets, q, session)?;

    let mut buf = Vec::new();
    writeln!(buf, "\n[[ {q} ]]")?;
    writeln!(buf, "search completed in {search_ms} ms\n")?;
    for r in &results {
        writeln!(buf, "---")?;
        let filename = if r.path.ends_with(".md") {
            format!("  {}", r.path)
        } else {
            String::new()
        };
        writeln!(buf, "{}  {}{filename}", r.timestamp, r.project)?;
        if !r.session_id.is_empty() {
            writeln!(buf, "  {} turn {}", r.session_id, r.turn)?;
        }
        let idx = r.match_idx;
        if idx > 0 {
            for line in r.bodies[idx - 1].lines().filter(|l| !l.is_empty()) {
                writeln!(buf, "  {line}")?;
            }
        }
        for line in r.bodies[idx].lines().filter(|l| !l.is_empty()) {
            writeln!(buf, "  {line}")?;
        }
        if idx + 1 < r.bodies.len() {
            for line in r.bodies[idx + 1].lines().filter(|l| !l.is_empty()) {
                writeln!(buf, "  {line}")?;
            }
        }
    }
    if results.is_empty() {
        writeln!(buf, "no results")?;
    }
    std::io::stdout().write_all(&buf)?;
    Ok(())
}

fn format_date(iso: &str) -> String {
    let month = match iso.get(5..7) {
        Some("01") => "Jan",
        Some("02") => "Feb",
        Some("03") => "Mar",
        Some("04") => "Apr",
        Some("05") => "May",
        Some("06") => "Jun",
        Some("07") => "Jul",
        Some("08") => "Aug",
        Some("09") => "Sep",
        Some("10") => "Oct",
        Some("11") => "Nov",
        Some("12") => "Dec",
        _ => "???",
    };
    let day = iso.get(8..10).unwrap_or("??");
    let time = iso.get(11..16).unwrap_or("??:??");
    format!("{month} {day} {time}")
}

fn parse_range(s: &str) -> (usize, usize) {
    if let Some((a, b)) = s.split_once('-') {
        let start = a.parse().unwrap_or(0);
        let end = b.parse().unwrap_or(usize::MAX);
        (start, end)
    } else {
        let n = s.parse().unwrap_or(0);
        (n, n)
    }
}

fn dump(db_name: &PathBuf, session_id: &str, turns_range: Option<&str>) -> Result<()> {
    let db = DB::new_reader(db_name.clone()).unwrap();

    let (turn_start, turn_end) = turns_range.map(parse_range).unwrap_or((0, usize::MAX));

    let mut stmt = db.query(
        "SELECT date, body, json_extract(metadata, '$.turn') as turn
         FROM document
         WHERE json_extract(metadata, '$.session_id') = ?1
         ORDER BY turn",
    )?;
    let rows: Vec<(String, String, i64)> = stmt
        .query_map((session_id,), |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    if rows.is_empty() {
        eprintln!("No session found for {session_id}");
        std::process::exit(1);
    }

    let mut buf = Vec::new();
    for (date, body, turn) in &rows {
        let t = *turn as usize;
        if t < turn_start || t > turn_end {
            continue;
        }
        writeln!(buf, "---")?;
        writeln!(buf, "turn {t}  {}", format_date(date))?;
        for line in body.lines().skip_while(|l| {
            l.starts_with('[') && !l.starts_with("[User]") && !l.starts_with("[Claude]")
        }) {
            writeln!(buf, "{line}")?;
        }
    }
    if !buf.is_empty() {
        writeln!(buf, "---")?;
    }

    use std::io::IsTerminal;
    let output = String::from_utf8(buf)?;
    if std::io::stdout().is_terminal() {
        use std::process::{Command, Stdio};
        let mut pager = Command::new("less")
            .args(["-RFX"])
            .stdin(Stdio::piped())
            .spawn()?;
        pager.stdin.take().unwrap().write_all(output.as_bytes())?;
        let _ = pager.wait();
    } else {
        print!("{output}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Warn));

    let args: Vec<String> = env::args().skip(1).collect();
    let do_update = args.iter().any(|a| a == "--update");
    let mut session_filter: Option<String> = None;
    let mut dump_session: Option<String> = None;
    let mut turns_range: Option<String> = None;
    let mut query_args: Vec<&str> = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--update" => {}
            "--session" => {
                session_filter = iter.next().cloned();
            }
            "--dump" => {
                dump_session = iter.next().cloned();
            }
            "--turns" => {
                turns_range = iter.next().cloned();
            }
            _ => {
                query_args.push(arg);
            }
        }
    }

    let db_name = db_path();
    let assets = assets_path();

    // Always ingest first — cheap filesystem walk with mtime watermarks
    let has_query = !query_args.is_empty();
    let has_dump = dump_session.is_some();
    if has_query || has_dump || do_update {
        match ingest(&db_name) {
            Ok(false) => {
                if do_update && !has_query && !has_dump {
                    println!("up to date");
                }
            }
            Ok(true) => {
                if do_update && !has_query {
                    let device = warp::make_device();
                    let embedder = Embedder::new(&device, &assets)?;
                    let db_rw = DB::new(db_name.clone()).unwrap();
                    embed_and_index(&db_rw, &embedder, &device)?;
                }
            }
            Err(e) => {
                if !db_name.exists() {
                    eprintln!("No database found. Run: pickbrain --update");
                    std::process::exit(1);
                }
                eprintln!("warning: ingest failed: {e}");
            }
        }
    }

    if let Some(ref sid) = dump_session {
        dump(&db_name, sid, turns_range.as_deref())?;
    } else if has_query {
        let q = query_args.join(" ");
        use std::io::IsTerminal;
        if std::io::stdout().is_terminal() {
            search_tui(&db_name, &assets, &q, session_filter.as_deref())?;
        } else {
            search_plain(&db_name, &assets, &q, session_filter.as_deref())?;
        }
    } else if !do_update {
        eprintln!("Usage: pickbrain [--update] [--session UUID] <query>");
        eprintln!("       pickbrain --dump <UUID> [--turns N-M]");
    }
    Ok(())
}
