use anyhow::Result;
use log::{Level, LevelFilter, Metadata, Record};
use std::env;
use std::io::Write;
use std::path::PathBuf;

mod claude_code;
mod codex;
mod watermark;

use witchcraft::{Filter, MetadataSchema, MetadataValue, Witchcraft};

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
    PathBuf::from(home).join(".claude/pickbrain.lance")
}

fn pickbrain_schema() -> MetadataSchema {
    MetadataSchema::new()
        .add_string("project", true)
        .add_string("session_id", true)
        .add_number("turn", false)
        .add_string("path", false)
        .add_string("source", true)
        .add_string("cwd", false)
}

fn assets_path() -> PathBuf {
    PathBuf::from(env::var("WARP_ASSETS").unwrap_or_else(|_| "assets".into()))
}

async fn ingest(wc: &mut Witchcraft) -> Result<bool> {
    let (sessions, memories, authored) = claude_code::ingest_claude_code(wc).await?;
    let codex_sessions = codex::ingest_codex(wc).await?;
    let total = sessions + memories + authored + codex_sessions;
    if total == 0 {
        eprintln!("No new sessions to ingest.");
        return Ok(false);
    }
    eprintln!(
        "ingested {sessions} claude sessions, {codex_sessions} codex sessions, {memories} memory files, {authored} authored files"
    );
    Ok(true)
}

// --- Search result data ---

struct TurnMeta {
    role: String,
    timestamp: String,
    byte_offset: u64,
    byte_len: u64,
}

struct SearchResult {
    timestamp: String,
    project: String,
    session_id: String,
    turn: u64,
    path: String,
    cwd: String,
    source: String,
    bodies: Vec<String>,
    match_idx: usize,
    turns: Vec<TurnMeta>,
}

// A turn from the original JSONL session file
struct SessionTurn {
    role: String,
    text: String,
    timestamp: String,
}

fn read_jsonl_line(path: &str, offset: u64, len: u64) -> Option<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).ok()?;
    f.seek(SeekFrom::Start(offset)).ok()?;
    let mut buf = vec![0u8; len as usize];
    f.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

fn read_turn_at(path: &str, source: &str, tm: &TurnMeta) -> Option<SessionTurn> {
    let line = read_jsonl_line(path, tm.byte_offset, tm.byte_len)?;
    let v: serde_json::Value = serde_json::from_str(&line).ok()?;

    let text = if source == "codex" {
        let payload = v.get("payload")?;
        let ptype = payload.get("type")?.as_str()?;
        if ptype == "message" && payload.get("role")?.as_str()? == "user" {
            let content = payload.get("content")?.as_array()?;
            let texts: Vec<&str> = content
                .iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("input_text"))
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect();
            texts.join("\n")
        } else if ptype == "agent_reasoning" {
            payload.get("text")?.as_str()?.to_string()
        } else {
            return None;
        }
    } else {
        // Claude Code
        let msg = v.get("message")?;
        match msg.get("content")? {
            c if c.is_string() => c.as_str()?.to_string(),
            c if c.is_array() => {
                let blocks = c.as_array()?;
                blocks
                    .iter()
                    .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            _ => return None,
        }
    };

    Some(SessionTurn {
        role: tm.role.clone(),
        text,
        timestamp: tm.timestamp.clone(),
    })
}


async fn run_search(
    wc: &mut Witchcraft,
    q: &str,
    session: Option<&str>,
) -> Result<(Vec<SearchResult>, u128)> {
    let filter = session.map(|id| Filter::eq("session_id", MetadataValue::String(id.to_string())));
    let now = std::time::Instant::now();
    let results = wc.search(q, 0.5, 10, true, filter.as_ref()).await?;
    let search_ms = now.elapsed().as_millis();

    let out: Vec<SearchResult> = results
        .into_iter()
        .map(|r| {
            let idx = (r.matched_sub_idx as usize).min(r.bodies.len().saturating_sub(1));
            SearchResult {
                timestamp: format_date(&r.date),
                project: match r.metadata.get("project") {
                    Some(MetadataValue::String(s)) => s.clone(),
                    _ => String::new(),
                },
                session_id: match r.metadata.get("session_id") {
                    Some(MetadataValue::String(s)) => s.clone(),
                    _ => String::new(),
                },
                turn: match r.metadata.get("turn") {
                    Some(MetadataValue::Number(n)) => *n as u64,
                    _ => 0,
                },
                path: match r.metadata.get("path") {
                    Some(MetadataValue::String(s)) => s.clone(),
                    _ => String::new(),
                },
                cwd: match r.metadata.get("cwd") {
                    Some(MetadataValue::String(s)) => s.clone(),
                    _ => String::new(),
                },
                source: match r.metadata.get("source") {
                    Some(MetadataValue::String(s)) => s.clone(),
                    _ => "claude".to_string(),
                },
                bodies: r.bodies,
                match_idx: idx,
                turns: vec![], // Turns metadata not stored as flat columns; could be reconstructed from JSONL if needed
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

async fn search_tui(
    wc: &mut Witchcraft,
    q: &str,
    session: Option<&str>,
) -> Result<Option<(String, String, String)>> {
    let (results, search_ms) = run_search(wc, q, session).await?;
    if results.is_empty() {
        eprintln!("no results");
        return Ok(None);
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
    let mut scroll_offset: usize = 0;
    let mut resume_session: Option<(String, String, String)> = None;
    let mut confirm_resume: Option<(String, String, String, String)> = None;
    struct DetailState {
        result_idx: usize,
        turns: Vec<SessionTurn>,
        highlight: usize,
    }
    let mut detail_cache: Option<DetailState> = None;

    loop {
        terminal.draw(|f| {
            let area = f.area();
            let show_footer = confirm_resume.is_some() && matches!(view, View::Detail(_));
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(if show_footer {
                    vec![Constraint::Length(2), Constraint::Min(0), Constraint::Length(1)]
                } else {
                    vec![Constraint::Length(2), Constraint::Min(0), Constraint::Length(0)]
                })
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
                        View::List => "↑↓ navigate  ⏎ open  q quit",
                        View::Detail(idx) if !results[idx].session_id.is_empty() => {
                            "↑↓ scroll  r resume session  esc back  q quit"
                        }
                        View::Detail(_) => "↑↓ scroll  esc back  q quit",
                    },
                    Style::default().fg(Color::DarkGray),
                ),
            ]));
            f.render_widget(header, chunks[0]);

            // Footer: resume confirmation
            if show_footer {
                let cwd = confirm_resume.as_ref()
                    .map(|(_, _, c, _)| c.as_str())
                    .unwrap_or("?");
                let sid = confirm_resume.as_ref()
                    .map(|(s, _, _, _)| s.as_str())
                    .unwrap_or("?");
                let src = confirm_resume.as_ref()
                    .map(|(_, _, _, s)| s.as_str())
                    .unwrap_or("claude");
                let footer = Paragraph::new(Line::from(vec![
                    Span::styled(
                        format!(" Exit pickbrain and resume {src} session {sid} in {cwd}? "),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        "(Y/n)",
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                ]));
                f.render_widget(footer, chunks[2]);
            }

            match view {
                View::List => {
                    let width = chunks[1].width as usize;
                    let items: Vec<ratatui::widgets::ListItem> = results
                        .iter()
                        .map(|r| {
                            let preview_idx = if r.match_idx == 0 && r.bodies.len() > 1 {
                                1
                            } else {
                                r.match_idx
                            };
                            let raw_preview = first_line(&r.bodies[preview_idx]);
                            let preview = strip_body_prefix(&raw_preview);
                            let matched_tm = if r.match_idx > 0 {
                                r.turns.get(r.match_idx - 1)
                            } else {
                                r.turns.first()
                            };
                            let ts = matched_tm
                                .filter(|tm| !tm.timestamp.is_empty())
                                .map(|tm| format_date(&tm.timestamp))
                                .unwrap_or_else(|| r.timestamp.clone());
                            let mut meta_spans = vec![
                                Span::styled(
                                    format!("{ts} "),
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
                                let source_label = if r.source == "codex" {
                                    "codex"
                                } else {
                                    "claude"
                                };
                                let short_sid = if r.session_id.len() > 8 {
                                    &r.session_id[..8]
                                } else {
                                    &r.session_id
                                };
                                meta_spans.push(Span::styled(
                                    format!("  {source_label} {short_sid}"),
                                    Style::default().fg(Color::Magenta),
                                ));
                                meta_spans.push(Span::styled(
                                    format!("  turn {}", r.turn),
                                    Style::default().fg(Color::DarkGray),
                                ));
                            }
                            let match_role = matched_tm.map(|tm| tm.role.as_str()).unwrap_or("");
                            let role_prefix = if match_role == "user" {
                                "[User] "
                            } else if match_role == "assistant" {
                                if r.source == "codex" { "[Codex] " } else { "[Claude] " }
                            } else {
                                ""
                            };
                            ratatui::widgets::ListItem::new(vec![
                                Line::from(meta_spans),
                                Line::from(vec![
                                    Span::styled(
                                        format!("  {role_prefix}"),
                                        Style::default().fg(if match_role == "user" {
                                            Color::Rgb(0, 255, 0)
                                        } else {
                                            Color::Cyan
                                        }),
                                    ),
                                    Span::raw(truncate(&preview, width.saturating_sub(4 + role_prefix.len()))),
                                ]),
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

                    // If we have a JSONL path and a session, show the real conversation
                    let dw = detail_cache
                        .as_ref()
                        .filter(|dw| dw.result_idx == idx);

                    if let Some(dw) = dw {
                        for (i, turn) in dw.turns.iter().enumerate() {
                            let is_highlight = i == dw.highlight;
                            let role_style = if turn.role == "user" {
                                Style::default()
                                    .fg(Color::Rgb(0, 255, 0))
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default()
                                    .fg(Color::Cyan)
                                    .add_modifier(Modifier::BOLD)
                            };
                            lines.push(Line::from(vec![
                                Span::styled(
                                    if turn.role == "user" {
                                        "[User] "
                                    } else if r.source == "codex" {
                                        "[Codex] "
                                    } else {
                                        "[Claude] "
                                    },
                                    role_style,
                                ),
                                Span::styled(
                                    format_date(&turn.timestamp),
                                    Style::default().fg(Color::DarkGray),
                                ),
                            ]));
                            let text_style = if is_highlight {
                                Style::default().fg(Color::White)
                            } else {
                                Style::default().fg(Color::DarkGray)
                            };
                            for line in turn.text.lines() {
                                lines.push(Line::styled(format!("  {line}"), text_style));
                            }
                            lines.push(Line::from(""));
                        }
                    } else {
                        // Fallback: show indexed bodies (for .md files etc.)
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
                    }

                    let detail = Paragraph::new(lines)
                        .wrap(Wrap { trim: false })
                        .scroll((scroll_offset as u16, 0));
                    f.render_widget(detail, chunks[1]);
                }
            }
        })?;

        if let Event::Key(key) = event::read()? {
            match (&view, key.code, key.modifiers) {
                (_, KeyCode::Char('q') | KeyCode::Esc, _) if confirm_resume.is_some() => {
                    confirm_resume = None;
                }
                (View::Detail(_), KeyCode::Char('q') | KeyCode::Esc, _) => {
                    view = View::List;
                }
                (View::List, KeyCode::Char('q') | KeyCode::Esc, _) => break,
                (_, KeyCode::Char('c'), KeyModifiers::CONTROL) => break,
                (_, KeyCode::Char('z'), KeyModifiers::CONTROL) => {
                    disable_raw_mode()?;
                    crossterm::execute!(
                        std::io::stdout(),
                        LeaveAlternateScreen,
                        crossterm::cursor::Show
                    )?;
                    unsafe { libc::raise(libc::SIGTSTP); }
                    // When resumed (fg), re-enter TUI
                    enable_raw_mode()?;
                    crossterm::execute!(
                        std::io::stdout(),
                        EnterAlternateScreen,
                        crossterm::cursor::Hide
                    )?;
                    // Force full redraw
                    terminal.clear()?;
                }

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
                    let r = &results[selected];
                    if !r.session_id.is_empty() && !r.path.is_empty() && !r.turns.is_empty() {
                        let mi = if r.match_idx > 0 { r.match_idx - 1 } else { 0 };
                        let mut turns = Vec::new();
                        for tm in &r.turns {
                            if let Some(turn) = read_turn_at(&r.path, &r.source, tm) {
                                turns.push(turn);
                            }
                        }
                        // Position viewport at the highlight turn
                        let mut pre_lines: usize = if r.session_id.is_empty() { 2 } else { 3 };
                        for t in &turns[..mi.min(turns.len())] {
                            pre_lines += 2 + t.text.lines().count();
                        }
                        scroll_offset = pre_lines;
                        detail_cache = Some(DetailState {
                            result_idx: selected,
                            turns,
                            highlight: mi,
                        });
                    } else {
                        scroll_offset = 0;
                        detail_cache = None;
                    }
                    view = View::Detail(selected);
                }

                // Detail view: j/k line scroll
                (View::Detail(_), KeyCode::Down | KeyCode::Char('j'), _) => {
                    scroll_offset = scroll_offset.saturating_add(1);
                }
                (View::Detail(_), KeyCode::Up | KeyCode::Char('k'), _) => {
                    scroll_offset = scroll_offset.saturating_sub(1);
                }
                (View::Detail(idx), KeyCode::Char('r'), _) => {
                    let r = &results[*idx];
                    if !r.session_id.is_empty() {
                        let cwd = if !r.cwd.is_empty() {
                            r.cwd.clone()
                        } else {
                            read_cwd_from_jsonl(&r.path, &r.source)
                                .unwrap_or_else(|| "?".to_string())
                        };
                        confirm_resume = Some((r.session_id.clone(), r.path.clone(), cwd, r.source.clone()));
                    }
                }
                (View::Detail(_), KeyCode::Char('y') | KeyCode::Enter, _) => {
                    if let Some((ref sid, ref path, _, ref source)) = confirm_resume {
                        resume_session = Some((sid.clone(), path.clone(), source.clone()));
                        break;
                    }
                }
                (View::Detail(_), KeyCode::Char('n'), _) => {
                    confirm_resume = None;
                }
                _ => {}
            }
        }
    }

    disable_raw_mode()?;
    crossterm::execute!(std::io::stdout(), LeaveAlternateScreen)?;
    Ok(resume_session)
}

fn read_cwd_from_jsonl(path: &str, source: &str) -> Option<String> {
    let raw = std::fs::read_to_string(path).ok()?;
    for line in raw.lines() {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        if source == "codex" {
            // Codex: cwd is in payload of session_meta entries
            if v.get("type").and_then(|t| t.as_str()) == Some("session_meta") {
                if let Some(cwd) = v.get("payload").and_then(|p| p.get("cwd")).and_then(|c| c.as_str()) {
                    return Some(cwd.to_string());
                }
            }
        } else {
            // Claude: cwd is a top-level field
            if let Some(cwd) = v.get("cwd").and_then(|c| c.as_str()) {
                return Some(cwd.to_string());
            }
        }
    }
    None
}

fn launch_resume(session_id: &str, jsonl_path: &str, source: &str) -> Result<()> {
    use std::os::unix::process::CommandExt;
    if let Some(cwd) = read_cwd_from_jsonl(jsonl_path, source) {
        let _ = std::env::set_current_dir(&cwd);
    }
    if source == "codex" {
        eprintln!("Resuming codex session {session_id}...");
        let err = std::process::Command::new("codex")
            .args(["resume", session_id])
            .exec();
        Err(err.into())
    } else {
        eprintln!("Resuming claude session {session_id}...");
        let err = std::process::Command::new("claude")
            .args(["--resume", session_id])
            .exec();
        Err(err.into())
    }
}


fn strip_body_prefix(s: &str) -> &str {
    s.strip_prefix("[User] ")
        .or_else(|| s.strip_prefix("[Claude] "))
        .or_else(|| s.strip_prefix("[Codex] "))
        .unwrap_or(s)
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

async fn search_plain(
    wc: &mut Witchcraft,
    q: &str,
    session: Option<&str>,
) -> Result<()> {
    let (results, search_ms) = run_search(wc, q, session).await?;

    let mut buf = Vec::new();
    writeln!(buf, "\n[[ {q} ]]")?;
    writeln!(buf, "search completed in {search_ms} ms\n")?;
    for r in &results {
        writeln!(buf, "---")?;
        // match_idx 0 = header, turns[0] = first turn → turns[match_idx - 1]
        let matched_tm = if r.match_idx > 0 {
            r.turns.get(r.match_idx - 1)
        } else {
            r.turns.first()
        };
        let ts = matched_tm
            .filter(|tm| !tm.timestamp.is_empty())
            .map(|tm| format_date(&tm.timestamp))
            .unwrap_or_else(|| r.timestamp.clone());
        let source_label = if r.source == "codex" { "codex" } else { "claude" };
        let filename = if r.path.ends_with(".md") {
            format!("  {}", r.path)
        } else {
            String::new()
        };
        let session_info = if !r.session_id.is_empty() {
            format!("  {source_label} {} turn {}", r.session_id, r.turn)
        } else {
            String::new()
        };
        writeln!(buf, "{ts}  {}{filename}{session_info}", r.project)?;
        if !r.session_id.is_empty() && !r.path.is_empty() && !r.turns.is_empty() {
            // Read only the matched turn + neighbors via byte offsets
            let mi = if r.match_idx > 0 { r.match_idx - 1 } else { 0 };
            let ctx_start = mi.saturating_sub(1);
            let ctx_end = (mi + 2).min(r.turns.len());
            for i in ctx_start..ctx_end {
                let tm = &r.turns[i];
                let label = if tm.role == "user" {
                    "[User]"
                } else if r.source == "codex" {
                    "[Codex]"
                } else {
                    "[Claude]"
                };
                let prefix = if i == mi { ">>>" } else { "  " };
                writeln!(buf, "{prefix} {label} {}", format_date(&tm.timestamp))?;
                if let Some(turn) = read_turn_at(&r.path, &r.source, tm) {
                    for line in turn.text.lines().take(10) {
                        writeln!(buf, "{prefix}   {line}")?;
                    }
                }
            }
        } else {
            // Fallback for .md files etc: use indexed bodies
            let idx = r.match_idx;
            for line in r.bodies[idx].lines().filter(|l| !l.is_empty()) {
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

async fn dump(wc: &mut Witchcraft, session_id: &str, turns_range: Option<&str>) -> Result<()> {
    use futures::TryStreamExt;
    use lancedb::query::{ExecutableQuery, QueryBase, Select};

    let (turn_start, turn_end) = turns_range.map(parse_range).unwrap_or((0, usize::MAX));

    let filter = format!("session_id = '{}'", session_id);
    let batches = wc.table_ref()
        .query()
        .only_if(&filter)
        .select(Select::Columns(vec![
            "date".to_string(),
            "body".to_string(),
            "sub_idx".to_string(),
            "turn".to_string(),
        ]))
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await?;

    // Collect (date, body, turn) and deduplicate by turn (take sub_idx 0)
    let mut rows: Vec<(String, String, i64)> = Vec::new();
    for batch in &batches {
        let date_col = batch.column_by_name("date").and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        let body_col = batch.column_by_name("body").and_then(|c| c.as_any().downcast_ref::<arrow_array::LargeStringArray>());
        let sub_idx_col = batch.column_by_name("sub_idx").and_then(|c| c.as_any().downcast_ref::<arrow_array::UInt32Array>());
        let turn_col = batch.column_by_name("turn").and_then(|c| c.as_any().downcast_ref::<arrow_array::Float64Array>());

        if let (Some(dates), Some(bodies), Some(sub_idxs), Some(turns)) = (date_col, body_col, sub_idx_col, turn_col) {
            for i in 0..batch.num_rows() {
                if sub_idxs.value(i) == 0 { // Only take first sub-chunk per turn
                    rows.push((
                        dates.value(i).to_string(),
                        bodies.value(i).to_string(),
                        turns.value(i) as i64,
                    ));
                }
            }
        }
    }
    rows.sort_by_key(|(_, _, t)| *t);

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

#[tokio::main]
async fn main() -> Result<()> {
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Warn));

    let args: Vec<String> = env::args().skip(1).collect();
    let mut session_filter: Option<String> = None;
    let mut dump_session: Option<String> = None;
    let mut turns_range: Option<String> = None;
    let mut query_args: Vec<&str> = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--nuke" => {
                let db_name = db_path();
                if db_name.exists() {
                    // Remove the lance directory
                    std::fs::remove_dir_all(&db_name).ok();
                    eprintln!("removed {}", db_name.display());
                } else {
                    eprintln!("no database to remove");
                }
                watermark::remove(&watermark::claude_path());
                watermark::remove(&watermark::codex_path());
                std::process::exit(0);
            }
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
    let schema = pickbrain_schema();

    let mut wc = Witchcraft::new(&db_name, &assets, schema).await?;

    match ingest(&mut wc).await {
        Ok(have_changes) => {
            if have_changes {
                wc.build_index().await?;
            }
        },
        Err(e) => {
            eprintln!("warning: ingest failed: {e}");
            std::process::exit(1);
        }
    }

    if let Some(ref sid) = dump_session {
        dump(&mut wc, sid, turns_range.as_deref()).await?;
    } else if !query_args.is_empty() {
        let q = query_args.join(" ");
        use std::io::IsTerminal;
        if std::io::stdout().is_terminal() {
            if let Some((sid, path, source)) = search_tui(&mut wc, &q, session_filter.as_deref()).await? {
                launch_resume(&sid, &path, &source)?;
            }
        } else {
            search_plain(&mut wc, &q, session_filter.as_deref()).await?;
        }
    } else {
        eprintln!("Usage: pickbrain [--session UUID] <query>");
        eprintln!("       pickbrain --dump <UUID> [--turns N-M]");
        eprintln!("       pickbrain --nuke");
    }
    Ok(())
}
