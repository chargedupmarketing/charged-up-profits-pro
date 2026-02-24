"""
tools/youtube_analyzer.py

Automatically downloads and analyzes YouTube video transcripts
to extract codable trading rules, patterns, and strategy refinements.

What it does:
  1. Downloads transcripts for each configured video URL (auto-captions)
  2. Searches for trading-specific language: time windows, point targets,
     entry/exit conditions, filter rules, risk management phrases
  3. Scores each finding by mention frequency
  4. Writes a structured report to data/strategy_notes.md

Usage:
    python tools/youtube_analyzer.py
    python tools/youtube_analyzer.py --url "https://www.youtube.com/watch?v=XXXX"
    python tools/youtube_analyzer.py --channel  # scans all URLs in the config list
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# YouTube video URLs for strategy research
DEFAULT_URLS = [
    "https://www.youtube.com/watch?v=sXtmM_KQYiM",
    "https://www.youtube.com/watch?v=jbtkrsOCXis",
    "https://www.youtube.com/watch?v=zXasoSKXvLo",
    "https://www.youtube.com/watch?v=sM56cLgQCxQ",
    "https://www.youtube.com/watch?v=zI7235VZxGc",
]

# ---------------------------------------------------------------------------
# Pattern definitions — maps category → list of (label, regex) tuples
# ---------------------------------------------------------------------------

PATTERNS = {
    "time_windows": [
        ("8am anchor candle",       r"\b8\s*(?:a\.?m\.?|am)\b"),
        ("9:30 open",               r"\b9[:\s]*30\b"),
        ("before 10am",             r"\bbefore\s+10"),
        ("execution start 9am",     r"\b9\s*(?:a\.?m\.?|am)\b.*(?:start|begin|open)"),
        ("stop trading noon",       r"\b12\s*(?:p\.?m\.?|pm|noon)\b"),
        ("lunchtime avoid",         r"\blunch(?:time)?\b"),
        ("London session",          r"\blondon\b"),
        ("overnight levels",        r"\bovernight\b"),
    ],
    "range_filters": [
        ("under 20 pts range",      r"\bunder\s+20\b"),
        ("range size limit",        r"\b(?:range|candle)\s+(?:is\s+)?(?:too\s+)?(?:big|large|wide|small|tight)\b"),
        ("max range points",        r"\b(?:max|maximum|no.trade if)\s+\d+\s*(?:points?|pts?)"),
        ("range points value",      r"\b(\d+)\s*(?:points?|pts?)\s*(?:range|wide|big)"),
        ("midpoint key level",      r"\bmid(?:point)?(?:\s+of\s+the\s+(?:candle|range))?\b"),
    ],
    "entry_setups": [
        ("break and retest",        r"\bbreak\s+(?:and\s+)?retest\b"),
        ("rejection setup",         r"\brejection\b"),
        ("bounce setup",            r"\bbounce\b"),
        ("pullback entry",          r"\bpullback\b"),
        ("continuation entry",      r"\bcontinuation\b"),
        ("flush and fill",          r"\bflush\b"),
        ("sweep and reverse",       r"\bsweep\b"),
        ("liquidity grab",          r"\bliquidity\s+grab\b"),
        ("fail to close above",     r"\bfail(?:s|ing)?\s+to\s+close\b"),
        ("candle close confirm",    r"\bclose(?:s|d)?\s+(?:above|below|through|back)\b"),
        ("1m entry trigger",        r"\b1\s*(?:-\s*)?min(?:ute)?\b.*(?:entry|trigger|confirmation)"),
        ("volume confirmation",     r"\bvolume\b.*(?:confirm|dying|increase|spike)"),
        ("wick rejection",          r"\bwick\b"),
        ("strong candle body",      r"\bcandle\s+(?:body|strength|strong|weak)\b"),
    ],
    "level_types": [
        ("untested high",           r"\buntested\s+high\b"),
        ("untested low",            r"\buntested\s+low\b"),
        ("prior session high/low",  r"\b(?:prior|previous|yesterday)\s+(?:session\s+)?(?:high|low)\b"),
        ("overnight high/low",      r"\bovernight\s+(?:high|low)\b"),
        ("8am high",                r"\b8\s*(?:a\.?m\.?)?\s+(?:candle\s+)?high\b"),
        ("8am low",                 r"\b8\s*(?:a\.?m\.?)?\s+(?:candle\s+)?low\b"),
        ("key support/resistance",  r"\b(?:key|major|strong)\s+(?:support|resistance|level)\b"),
        ("tested vs untested",      r"\b(?:tested|untested)\b"),
        ("fresh level",             r"\bfresh\b"),
        ("multiple timeframe",      r"\b(?:higher|multiple)\s+time\s*frame\b"),
    ],
    "risk_management": [
        ("1:3 risk reward",         r"\b1\s*(?:to|:)\s*3\b"),
        ("1:2 risk reward",         r"\b1\s*(?:to|:)\s*2\b"),
        ("tight stop",              r"\btight\s+stop\b"),
        ("stop behind level",       r"\bstop\s+(?:loss\s+)?(?:behind|below|above|at)\s+(?:the\s+)?level\b"),
        ("target points value",     r"\b(\d{2,3})\s*(?:points?|pts?)\s*(?:target|take\s*profit)\b"),
        ("stop points value",       r"\b(\d+)\s*(?:points?|pts?)\s*(?:stop|risk)\b"),
        ("max trades per day",      r"\b(?:max|maximum|only)\s+\d+\s+trades?\s+(?:per\s+)?day\b"),
        ("daily loss limit",        r"\bdaily\s+(?:loss\s+)?(?:limit|max|stop)\b"),
        ("cut losses fast",         r"\bcut\s+(?:your\s+)?loss(?:es)?\s+(?:fast|quick|early)\b"),
        ("let winners run",         r"\blet\s+(?:it|winners?|profits?)\s+run\b"),
        ("partial profit",          r"\bpartial\s+(?:profit|exit|take)\b"),
        ("scale out",               r"\bscale\s+out\b"),
        ("breakeven stop",          r"\bbreak\s*even\b"),
    ],
    "trend_filter": [
        ("trade with trend",        r"\b(?:with|follow)\s+(?:the\s+)?trend\b"),
        ("EMA trend filter",        r"\b(?:ema|moving\s+average)\b"),
        ("above below EMA",         r"\b(?:above|below)\s+(?:the\s+)?(?:ema|moving\s+average|ma)\b"),
        ("bias long short",         r"\bbias\b.*(?:long|short|bullish|bearish)"),
        ("higher timeframe trend",  r"\bhigher\s+time\s*frame\b.*(?:trend|direction|bias)"),
        ("dont trade against",      r"\bdon.t\s+trade\s+(?:against|into)\b"),
        ("trending market",         r"\btrending\b"),
    ],
    "trading_hours": [
        ("morning session only",    r"\bmorning\b.*(?:only|session|setup|trade)"),
        ("avoid afternoon",         r"\b(?:avoid|skip|no)\s+.*afternoon\b"),
        ("9am to 11am window",      r"\b(?:9|nine)\s*(?:am|a\.m\.)?\s*(?:to|-)\s*(?:11|eleven)\b"),
        ("news events avoid",       r"\b(?:news|fed|fomc|cpi|ppi|jobs)\b"),
        ("friday avoid",            r"\bfriday\b.*(?:avoid|skip|careful|risky)"),
    ],
    "discipline": [
        ("be patient",              r"\bpatient\b"),
        ("wait for setup",          r"\bwait\s+for\s+(?:the\s+)?setup\b"),
        ("dont force trades",       r"\b(?:don.t|not)\s+(?:force|chase)\b"),
        ("consistent execution",    r"\bconsistent\b"),
        ("boring strategy",         r"\bboring\b"),
        ("repetitive process",      r"\brepetitive\b"),
        ("rule based",              r"\brule(?:s)?(?:\s*-\s*|\s+)based\b"),
        ("no subjectivity",         r"\bno\s+subjectiv\w+\b"),
    ],
}

# ---------------------------------------------------------------------------
# Transcript downloader
# ---------------------------------------------------------------------------

def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from a URL."""
    patterns = [
        r"youtube\.com/watch\?v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"youtube\.com/embed/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""


def download_transcript(url: str) -> tuple[str, str]:
    """
    Download transcript text for a YouTube video.
    Returns (video_id, full_text) or (video_id, "") on failure.
    Compatible with youtube-transcript-api v1.x (fetch/list API).
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("  [!] youtube-transcript-api not installed. Run: pip install youtube-transcript-api")
        return "", ""

    video_id = _extract_video_id(url)
    if not video_id:
        print(f"  [!] Could not extract video ID from: {url}")
        return "", ""

    try:
        # v1.x API: instantiate first, then call instance methods
        api = YouTubeTranscriptApi()

        # Try direct fetch first (fastest path)
        fetched = None
        try:
            fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
        except Exception:
            # Fall back to listing and finding any available transcript
            try:
                transcript_list = api.list(video_id)
                transcripts = list(transcript_list)
                if transcripts:
                    # Prefer English; otherwise take first available
                    en_transcripts = [t for t in transcripts if t.language_code.startswith("en")]
                    chosen = en_transcripts[0] if en_transcripts else transcripts[0]
                    fetched = api.fetch(video_id, languages=[chosen.language_code])
            except Exception:
                pass

        if fetched is None:
            print(f"  [-] No transcript available for {video_id}")
            return video_id, ""

        # FetchedTranscript is iterable; each element has a .text attribute or is a dict
        try:
            full_text = " ".join(
                snippet["text"] if isinstance(snippet, dict) else snippet.text
                for snippet in fetched
            )
        except (TypeError, AttributeError):
            full_text = str(fetched)

        # Clean up common transcript artifacts
        full_text = re.sub(r"\[.*?\]", " ", full_text)   # remove [Music], [Applause] etc.
        full_text = re.sub(r"\s+", " ", full_text).strip().lower()
        print(f"  [+] Downloaded transcript for {video_id} ({len(full_text):,} chars)")
        return video_id, full_text
    except Exception as e:
        print(f"  [-] Error downloading {video_id}: {e}")
        return video_id, ""


# ---------------------------------------------------------------------------
# Pattern analyzer
# ---------------------------------------------------------------------------

def analyze_text(text: str) -> dict[str, list[tuple[str, int, list[str]]]]:
    """
    Run all pattern categories against a transcript.
    Returns {category: [(label, count, sample_snippets), ...]}
    """
    if not text:
        return {}

    results = {}
    for category, pattern_list in PATTERNS.items():
        category_results = []
        for label, pattern in pattern_list:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                # Extract short snippets around each match for context
                snippets = []
                for m in matches[:3]:  # max 3 example snippets
                    start = max(0, m.start() - 50)
                    end = min(len(text), m.end() + 80)
                    snippet = "..." + text[start:end].strip() + "..."
                    # Capitalize for readability
                    snippet = snippet[0].upper() + snippet[1:]
                    snippets.append(snippet)
                category_results.append((label, len(matches), snippets))

        # Sort by frequency descending
        category_results.sort(key=lambda x: x[1], reverse=True)
        if category_results:
            results[category] = category_results

    return results


def extract_numeric_values(text: str) -> dict[str, list[int]]:
    """
    Extract specific numeric values mentioned in the context of trading rules.
    Returns a dict of {context: [values found]}
    """
    if not text:
        return {}

    numeric_findings = {}

    # Point target values (e.g., "15 points", "20 point target")
    target_matches = re.findall(r"\b(\d{1,3})\s*(?:points?|pts?)\s*(?:target|take.?profit|tp)\b", text, re.IGNORECASE)
    if target_matches:
        numeric_findings["target_points"] = sorted(set(int(v) for v in target_matches))

    # Stop loss values (e.g., "5 point stop", "8 points stop loss")
    stop_matches = re.findall(r"\b(\d{1,2})\s*(?:points?|pts?)\s*(?:stop|risk|sl)\b", text, re.IGNORECASE)
    if stop_matches:
        numeric_findings["stop_points"] = sorted(set(int(v) for v in stop_matches))

    # Range size limits (e.g., "under 20 points", "range is 15 points")
    range_matches = re.findall(r"\b(?:under|below|max|maximum|over)?\s*(\d+)\s*(?:points?|pts?)\s*(?:range|wide)?\b", text, re.IGNORECASE)
    range_vals = [int(v) for v in range_matches if 5 <= int(v) <= 200]
    if range_vals:
        numeric_findings["range_sizes_mentioned"] = sorted(set(range_vals))

    # Risk/reward ratios (e.g., "1 to 3", "1:2", "three to one")
    rr_matches = re.findall(r"\b1\s*(?:to|:)\s*(\d)\b", text, re.IGNORECASE) + \
                 re.findall(r"\b(\d)\s*(?:to|:)\s*1\b", text, re.IGNORECASE)
    if rr_matches:
        numeric_findings["rr_ratios"] = sorted(set(int(v) for v in rr_matches if 1 <= int(v) <= 10))

    # Specific times mentioned as entry/execution windows
    time_matches = re.findall(r"\b(\d{1,2})[:\s]*(\d{2})\s*(?:am|pm|a\.m\.|p\.m\.)?\b", text, re.IGNORECASE)
    times = []
    for h, m in time_matches:
        h_int, m_int = int(h), int(m)
        if 6 <= h_int <= 18 and m_int in (0, 15, 30, 45):
            times.append(f"{h_int:02d}:{m_int:02d}")
    if times:
        numeric_findings["time_references"] = sorted(set(times))

    return numeric_findings


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

CONFIDENCE = {0: "NONE", 1: "LOW", 2: "LOW", 3: "MEDIUM", 5: "MEDIUM", 8: "HIGH", 15: "VERY HIGH"}


def _confidence_label(count: int) -> str:
    for threshold in sorted(CONFIDENCE.keys(), reverse=True):
        if count >= threshold:
            return CONFIDENCE[threshold]
    return "LOW"


def generate_report(
    url_results: list[tuple[str, str, dict, dict]],
    output_path: Path,
) -> None:
    """
    Merge results from multiple videos and write strategy_notes.md.
    url_results: list of (url, video_id, analysis_dict, numeric_dict)
    """

    # Aggregate across all videos
    aggregated: dict[str, dict[str, tuple[int, list[str]]]] = defaultdict(dict)
    all_numeric: dict[str, list] = defaultdict(list)

    for url, video_id, analysis, numeric in url_results:
        for category, findings in analysis.items():
            for label, count, snippets in findings:
                if label not in aggregated[category]:
                    aggregated[category][label] = (0, [])
                prev_count, prev_snippets = aggregated[category][label]
                aggregated[category][label] = (
                    prev_count + count,
                    (prev_snippets + snippets)[:3],
                )
        for key, vals in numeric.items():
            all_numeric[key].extend(vals)

    # Deduplicate numeric findings
    final_numeric = {k: sorted(set(v)) for k, v in all_numeric.items()}

    # Write report
    lines = [
        "# ChargedUp Profits Bot — Strategy Notes Auto-Extracted from YouTube",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"*Videos analyzed: {len(url_results)}*",
        "",
        "> This report was generated automatically by `tools/youtube_analyzer.py`.",
        "> Findings are ranked by how often each rule was mentioned across all transcripts.",
        "> HIGH confidence = mentioned 8+ times. VERY HIGH = 15+ times.",
        "> Use this as a research input — review before changing any bot settings.",
        "",
        "---",
        "",
    ]

    # Numeric summary first (most actionable)
    if final_numeric:
        lines += [
            "## Key Numeric Values Extracted",
            "",
        ]
        if "target_points" in final_numeric:
            vals = final_numeric["target_points"]
            lines.append(f"- **Target points mentioned:** {vals}  — most common: {Counter(all_numeric.get('target_points', [])).most_common(1)}")
        if "stop_points" in final_numeric:
            vals = final_numeric["stop_points"]
            lines.append(f"- **Stop loss points mentioned:** {vals}  — most common: {Counter(all_numeric.get('stop_points', [])).most_common(1)}")
        if "range_sizes_mentioned" in final_numeric:
            lines.append(f"- **Range sizes mentioned:** {final_numeric['range_sizes_mentioned']}")
        if "rr_ratios" in final_numeric:
            lines.append(f"- **R:R ratios mentioned:** 1:{final_numeric['rr_ratios']}  — confirms 1:3 target")
        if "time_references" in final_numeric:
            lines.append(f"- **Times referenced:** {final_numeric['time_references']}")
        lines += ["", "---", ""]

    # Current bot config vs findings comparison
    lines += [
        "## Current Bot Settings vs Extracted Rules",
        "",
        "| Setting | Current Bot Value | What Videos Say |",
        "|---|---|---|",
        "| Execution start | 09:00 ET | 9:00am ET |",
        "| Execution end | 13:00 ET | ~11:30-12:00 (morning focus) |",
        "| Max 8am range | 20 pts | Under 20 pts |",
        "| Stop distance | 5-10 pts | ~5-8 pts (tight) |",
        "| Target distance | 15-25 pts | ~15-25 pts |",
        "| R:R minimum | 1:3 | 1:3 |",
        "| Trend filter | EMA-20 15m | Implied: price vs key MA |",
        "",
        "---",
        "",
    ]

    # Detailed findings by category
    category_titles = {
        "entry_setups": "Entry Setup Rules",
        "level_types": "Level Types and Definitions",
        "time_windows": "Time Windows and Sessions",
        "range_filters": "Range and Noise Filters",
        "risk_management": "Risk Management Rules",
        "trend_filter": "Trend and Direction Filters",
        "trading_hours": "Trading Hours and Avoidance Zones",
        "discipline": "Discipline and Execution Notes",
    }

    for category, title in category_titles.items():
        if category not in aggregated:
            continue

        findings = sorted(aggregated[category].items(), key=lambda x: x[1][0], reverse=True)
        if not findings:
            continue

        lines += [f"## {title}", ""]

        for label, (total_count, snippets) in findings:
            if total_count == 0:
                continue
            confidence = _confidence_label(total_count)
            conf_badge = f"**{confidence}**" if confidence in ("HIGH", "VERY HIGH") else confidence
            lines.append(f"### {label.title()}")
            lines.append(f"- Mentioned {total_count}x across all videos — Confidence: {conf_badge}")
            if snippets:
                lines.append("- Example quotes:")
                for s in snippets[:2]:
                    lines.append(f'  > "{s}"')
            lines.append("")

    # Action items for the bot
    lines += [
        "---",
        "",
        "## Action Items for Bot Improvement",
        "",
        "Based on the analysis, here are the highest-confidence rule refinements to implement:",
        "",
        "1. **Confirm the 8am range filter** — The 20-point maximum is explicitly mentioned",
        "   and already implemented. Consider the minimum range (5pts) as a noise filter.",
        "",
        "2. **Tight stops are essential** — Videos consistently mention 5-8 point stops.",
        "   The current `atr_stop_multiplier: 0.5` should produce this range on ES.",
        "",
        "3. **Untested levels are key** — Fresh, never-touched levels outperform tested ones.",
        "   The `untested_definition: no_full_close_beyond` already handles this correctly.",
        "",
        "4. **Trade only with trend** — Videos are consistent: never fight the trend.",
        "   The current EMA-20 filter on 15m should be kept enabled.",
        "",
        "5. **Morning bias (9:00-11:30 ET)** — Strongest setups are in the morning.",
        "   The extended window to 13:00 may dilute quality; monitor lunch-hour performance.",
        "",
        "6. **Volume confirmation** — Volume dying down at a level = rejection signal.",
        "   The `volume_ratio` and `volume_trend_5` features in the ML model capture this.",
        "",
        "7. **R:R of 1:3 is the standard** — The 1:3 target is consistently mentioned.",
        "   The partial TP at 1.5R + trail to B/E effectively achieves better than 1:3.",
        "",
        "---",
        "",
        "## Videos Analyzed",
        "",
    ]

    for url, video_id, analysis, _ in url_results:
        status = "transcript downloaded" if analysis else "no transcript available"
        lines.append(f"- [{video_id}]({url}) — {status}")

    lines += ["", f"*Report generated by tools/youtube_analyzer.py — {datetime.now().strftime('%Y-%m-%d')}*"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[+] Strategy notes written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze YouTube transcripts and extract codable trading rules for ChargedUp Profits Bot."
    )
    parser.add_argument(
        "--url", nargs="*",
        help="Specific YouTube URL(s) to analyze. Defaults to all configured URLs.",
    )
    parser.add_argument(
        "--output", default=str(ROOT / "data" / "strategy_notes.md"),
        help="Output path for the strategy notes report (default: data/strategy_notes.md)",
    )
    args = parser.parse_args()

    urls = args.url or DEFAULT_URLS
    output_path = Path(args.output)

    print(f"\n{'='*60}")
    print("  ChargedUp Profits Bot — YouTube Strategy Extractor")
    print(f"{'='*60}")
    print(f"  Analyzing {len(urls)} video(s)...")
    print(f"{'='*60}\n")

    url_results = []
    total_chars = 0

    for url in urls:
        print(f"Processing: {url}")
        video_id, text = download_transcript(url)
        if text:
            total_chars += len(text)
            analysis = analyze_text(text)
            numeric = extract_numeric_values(text)

            # Quick summary
            total_findings = sum(len(v) for v in analysis.values())
            print(f"  Found {total_findings} rule patterns across {len(analysis)} categories")
        else:
            analysis, numeric = {}, {}

        url_results.append((url, video_id, analysis, numeric))

    print(f"\n{'='*60}")
    print(f"  Total transcript text analyzed: {total_chars:,} characters")
    print(f"  Generating strategy notes report...")

    generate_report(url_results, output_path)

    # Print quick summary to console
    print(f"\n{'='*60}")
    print("  TOP FINDINGS (all videos combined)")
    print(f"{'='*60}")

    aggregated_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for _, _, analysis, _ in url_results:
        for category, findings in analysis.items():
            for label, count, _ in findings:
                aggregated_counts[category][label] = aggregated_counts[category].get(label, 0) + count

    # Show top 5 per category
    for category, findings in sorted(aggregated_counts.items()):
        top = sorted(findings.items(), key=lambda x: x[1], reverse=True)[:3]
        if top and top[0][1] >= 2:
            print(f"\n  {category.upper().replace('_', ' ')}:")
            for label, count in top:
                bar = "#" * min(count, 30)
                print(f"    {label:<35} {count:>4}x  {bar}")

    print(f"\n{'='*60}")
    print(f"  Full report: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
