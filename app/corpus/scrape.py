"""Build the RAG corpus from hhgoa.com.

Findings from inspecting the live site (2026-08-22) that shape this module:

1.  The site is Next.js but server-rendered — all prose is present in the
    initial HTML. No headless browser needed, which keeps the image small.

2.  Layout classes are hashed Tailwind utilities that change on every rebuild.
    So we anchor on SEMANTIC ids (`#notice-board`, `#tasks`, `#timeline`) and
    heading text, never on classes.

3.  The four hero stat counters (Registrations / Projects / Hackers / Bounties)
    animate from zero in client JS. Their real values are NOT in the HTML —
    scraping them yields "0". We therefore EXCLUDE them rather than ship a
    corpus that confidently states "0 hackers attended". Queries about those
    numbers are meant to hit the retrieval guardrail and be refused. See
    EXCLUSIONS below; the reason is recorded in the manifest.

4.  `/result` is a live leaderboard reveal that changes as ranks are announced.
    Indexing it would produce answers that are stale the moment a rank drops,
    so it is excluded by default (`--include-volatile` to override).

5.  Empty-state placeholders ("Nothing pinned yet") are volatile in the same
    way, AND they poison ranking. See EMPTY_STATE_RE below for the measurement.

Run:  py -m app.corpus.scrape
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag

BASE = "https://hhgoa.com"
OUT_DIR = Path("data/corpus")

# (path, is_volatile)
PAGES: list[tuple[str, bool]] = [
    ("/", False),
    ("/terms", False),
    ("/result", True),
]

EXCLUSIONS = [
    {
        "what": "hero stat counters (Registrations 2024, Projects, Hackers, Bounties 2026)",
        "why": (
            "Values are animated from 0 by client-side JS and are absent from the "
            "server HTML. Indexing them would ground answers in a literal '0'. "
            "Excluded so these queries are refused instead of answered wrongly."
        ),
    },
    {
        "what": "/result leaderboard ranks",
        "why": "Live reveal page; contents change as ranks are announced. Stale by design.",
    },
    {
        "what": "empty-state placeholders (e.g. the notice board's 'Nothing pinned yet')",
        "why": (
            "Volatile UI state rather than content, and measurably harmful in the "
            "index: a 41-char chunk had the highest mean cosine of all 92 vectors "
            "against control queries it shares no topic with, taking rank 1 on "
            "6/48 benchmark queries. See EMPTY_STATE_RE."
        ),
    },
]

# Text that signals a stats/counter block we must drop (finding #3).
COUNTER_MARKERS = ("Registrations", "Bounties", "Inside HHG", "Past Editions")

# Empty-state placeholders (finding #5). The notice board renders "Nothing
# pinned yet" until an organiser posts something. Two independent reasons to
# drop it, either of which would be sufficient:
#
#   volatile   It is a UI state, not content -- true today, wrong the moment
#              something is pinned. Same category as /result.
#
#   attractor  MEASURED: as a 41-char chunk it had the HIGHEST mean cosine of
#              all 92 vectors against six control queries about car tyres,
#              cookies and Moby Dick -- content it shares nothing with. Short
#              embed texts sit near the centroid of embedding space
#              (corr(len, mean cosine) = -0.45 over the 48 benchmark queries,
#              -0.51 over the controls), so they win rank 1 for whatever is
#              asked. It took top-1 on 6/48 benchmark queries under
#              parent_child and 13/48 under structural. recall@5 hid this
#              completely; precision@1 in bench_retrieval.py now catches it.
EMPTY_STATE_RE = re.compile(
    r"\b(nothing\s+(pinned|posted|scheduled|here)\s+yet"
    r"|no\s+(notices|updates|announcements|entries|posts)\b"
    r"|coming\s+soon|stay\s+tuned|to\s+be\s+announced)\b",
    re.I,
)

# A doc shorter than this carries too little signal to be a trustworthy
# retrieval target, for the reason measured above. The shortest genuine doc on
# the site is a 95-char timeline milestone, so this floor cannot reach real
# content -- it only catches placeholders the pattern above missed.
MIN_DOC_CHARS = 60

BOILERPLATE = re.compile(
    r"^(check hype|apply|sound|←\s*hh goa|←\s*back to home|tap to reveal|"
    r"less noise\. more signal)$",
    re.I,
)


@dataclass
class Doc:
    """One retrievable unit of source content, before chunking."""

    doc_id: str
    url: str
    page: str
    section: str
    title: str
    text: str
    kind: str  # faq | task | timeline | notice | terms | prose
    volatile: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return len(self.text)


def clean(s: str) -> str:
    """Collapse whitespace and strip zero-width / non-breaking junk."""
    s = s.replace("​", "").replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def fetch(path: str, *, retries: int = 3) -> str:
    """GET a page with backoff. The corpus build is not allowed to half-succeed:
    a silently missing page would look like a retrieval bug later."""
    url = f"{BASE}{path}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = httpx.get(url, timeout=20.0, follow_redirects=True)
            r.raise_for_status()
            return r.text
        except Exception as e:  # noqa: BLE001 - retry any transport/status error
            last = e
            if attempt < retries - 1:
                sleep = 2**attempt
                print(f"  retry {attempt + 1}/{retries - 1} for {path} in {sleep}s ({e})")
                time.sleep(sleep)
    raise RuntimeError(f"could not fetch {url}: {last}")


def soupify(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")
    for bad in soup(["script", "style", "noscript", "svg"]):
        bad.decompose()
    return soup


# --------------------------------------------------------------------------
# home page
# --------------------------------------------------------------------------


def _looks_like_counter(text: str) -> bool:
    return any(m.lower() in text.lower() for m in COUNTER_MARKERS)


def extract_faqs(soup: BeautifulSoup, page: str) -> list[Doc]:
    """FAQ entries render as `question | + | answer`.

    Primary strategy is DOM-based via the stable `faq-panel-` id prefix. If that
    yields nothing (markup changed), fall back to a text-pattern parse so the
    build degrades instead of silently producing an empty FAQ set.
    """
    docs: list[Doc] = []

    panels = soup.select('[id^="faq-panel-"]')
    for i, panel in enumerate(panels):
        answer = clean(panel.get_text(" "))
        question = ""
        # The question is the nearest preceding heading/button text.
        node = panel
        while node is not None and not question:
            prev = node.find_previous(["button", "h3", "h4", "summary", "dt"])
            if prev is None:
                break
            cand = clean(prev.get_text(" ")).rstrip("+").strip()
            if cand and cand != "+" and len(cand) > 3:
                question = cand
            node = prev
        if question and answer:
            docs.append(
                Doc(
                    doc_id=f"faq-{i:02d}",
                    url=f"{BASE}{page}#faq",
                    page=page,
                    section="FAQs",
                    title=question,
                    # Keep Q and A together: an answer chunk without its
                    # question embeds poorly and reads as a fragment.
                    text=f"Q: {question}\nA: {answer}",
                    kind="faq",
                )
            )

    if docs:
        return docs

    # Fallback: flattened-text pattern parse.
    print("  ! faq-panel- selector matched nothing; using text-pattern fallback")
    flat = re.sub(r"\s*\|\s*", "|", re.sub(r"\s+", " ", soup.get_text("|")))
    for i, m in enumerate(re.finditer(r"\|([^|]{10,160}\?)\|\+\|([^|]{20,1200})\|", flat)):
        q, a = clean(m.group(1)), clean(m.group(2))
        docs.append(
            Doc(
                doc_id=f"faq-{i:02d}",
                url=f"{BASE}{page}#faq",
                page=page,
                section="FAQs",
                title=q,
                text=f"Q: {q}\nA: {a}",
                kind="faq",
                meta={"extraction": "text-fallback"},
            )
        )
    return docs


def extract_section(soup: BeautifulSoup, sec_id: str, kind: str, page: str) -> list[Doc]:
    """Pull one semantic section (#tasks, #timeline, #notice-board).

    Sections with h3 sub-items (Tasks) split per sub-item; otherwise the whole
    section becomes one doc.
    """
    node = soup.find(id=sec_id)
    if not isinstance(node, Tag):
        print(f"  ! section #{sec_id} not found")
        return []

    heading = node.find(["h1", "h2"])
    sec_title = clean(heading.get_text(" ")) if heading else sec_id

    subs = node.find_all("h3")
    docs: list[Doc] = []

    if subs:
        for i, sub in enumerate(subs):
            title = clean(sub.get_text(" "))
            parts: list[str] = []
            for sib in sub.find_all_next():
                if sib in subs and sib is not sub:
                    break
                if sib.name in ("p", "li", "h4") and sib.find_parent(id=sec_id):
                    t = clean(sib.get_text(" "))
                    if t and not BOILERPLATE.match(t) and t not in parts:
                        parts.append(t)
            body = "\n".join(parts)
            if not body:
                continue
            docs.append(
                Doc(
                    doc_id=f"{sec_id}-{i:02d}",
                    url=f"{BASE}{page}#{sec_id}",
                    page=page,
                    section=sec_title,
                    title=title,
                    text=f"{title}\n{body}",
                    kind=kind,
                )
            )
        if docs:
            return docs

    text = clean(node.get_text(" "))
    if _looks_like_counter(text):
        print(f"  - skipped #{sec_id}: matches excluded counter block")
        return []
    if not text:
        return []
    return [
        Doc(
            doc_id=f"{sec_id}-00",
            url=f"{BASE}{page}#{sec_id}",
            page=page,
            section=sec_title,
            title=sec_title,
            text=text,
            kind=kind,
        )
    ]


DATE_RE = re.compile(
    r"(20\d\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    re.I,
)


def extract_timeline(soup: BeautifulSoup, page: str = "/") -> list[Doc]:
    """The roadmap renders as a positional grid, not as paired rows:

        <div>                       <- row group
          <div>  5x <div><span>date</span></div>          <- dates column
          <div>  5x <div><p>title</p><p>desc</p></div>    <- events column

    A naive get_text() concatenates all dates, then all events, destroying the
    date->milestone mapping. Since "when does registration begin / when are
    results out" is the most likely question this corpus will ever be asked,
    that mapping is the whole value of the section. So we pair the two columns
    index-wise and emit one doc per milestone.
    """
    node = soup.find(id="timeline")
    if not isinstance(node, Tag):
        print("  ! section #timeline not found")
        return []

    docs: list[Doc] = []
    seen: set[str] = set()

    for row in node.find_all("div"):
        cols = row.find_all("div", recursive=False)
        if len(cols) != 2:
            continue
        dates = cols[0].find_all("div", recursive=False)
        events = cols[1].find_all("div", recursive=False)
        if len(dates) < 2 or len(dates) != len(events):
            continue

        date_texts = [clean(d.get_text(" ")) for d in dates]
        # Guard against matching an outer wrapper that happens to have 2 cols:
        # a real dates column is short strings that all look like dates.
        if not all(t and len(t) < 30 and DATE_RE.search(t) for t in date_texts):
            continue

        for date, ev in zip(date_texts, events):
            ps = ev.find_all("p")
            if not ps:
                continue
            title = clean(ps[0].get_text(" "))
            desc = clean(" ".join(p.get_text(" ") for p in ps[1:]))
            if not title or title in seen:
                continue
            seen.add(title)
            docs.append(
                Doc(
                    doc_id=f"timeline-{len(docs):02d}",
                    url=f"{BASE}{page}#timeline",
                    page=page,
                    section="The Timeline at a Glance",
                    title=title,
                    # Date first and restated: the date is the answer to most
                    # timeline questions, so it must survive any truncation.
                    text=f"{title} — {date}\n{desc}\nMilestone date: {date}.",
                    kind="timeline",
                    meta={"date": date, "milestone": title},
                )
            )

    if not docs:
        print("  ! timeline pairing failed; falling back to flat section text")
        return extract_section(soup, "timeline", "timeline", page)
    return docs


KNOWN_SECTION_IDS = ("notice-board", "timeline", "tasks", "check-hype")


def extract_home_prose(soup: BeautifulSoup, page: str = "/") -> list[Doc]:
    """Capture home-page prose that sits outside the semantic sections.

    The "Less Noise. More Signal" pitch block belongs to no section id, so
    section-based extraction silently dropped it. It is real site content and
    it states "500 elite builders" — which CONFLICTS with the timeline and
    terms, both of which say 247.

    We index it anyway. A corpus where two sources disagree is a much better
    test of whether the system cites its source instead of quietly picking a
    winner, and the disagreement is genuinely in the site rather than staged.
    """
    root = soup.find("main") or soup.body
    if root is None:
        return []

    parts: list[str] = []
    for p in root.find_all(["p", "li"]):
        if any(p.find_parent(id=sid) for sid in KNOWN_SECTION_IDS):
            continue
        if p.find_parent(attrs={"id": re.compile(r"^faq-panel-")}):
            continue
        t = clean(p.get_text(" "))
        # Long-form only: short strings here are nav labels and counter labels.
        if len(t) < 40 or _looks_like_counter(t) or BOILERPLATE.match(t):
            continue
        if t not in parts:
            parts.append(t)

    if not parts:
        return []

    body = "\n".join(parts)
    return [
        Doc(
            doc_id="about-00",
            url=f"{BASE}{page}",
            page=page,
            section="About HH Goa",
            title="About Hacker House Goa 2026",
            text=f"About Hacker House Goa 2026\n{body}",
            kind="prose",
            meta={"note": "may conflict with timeline/terms on builder count"},
        )
    ]


def extract_home(html: str) -> list[Doc]:
    soup = soupify(html)
    docs: list[Doc] = []
    docs += extract_section(soup, "notice-board", "notice", "/")
    docs += extract_timeline(soup, "/")
    docs += extract_section(soup, "tasks", "task", "/")
    docs += extract_faqs(soup, "/")
    docs += extract_home_prose(soup, "/")
    return docs


# --------------------------------------------------------------------------
# terms page — densest factual content on the site
# --------------------------------------------------------------------------


def extract_terms(html: str) -> list[Doc]:
    """Terms is long prose under headings. Split on heading boundaries so each
    clause stays with its own heading (a bare clause is unciteable)."""
    soup = soupify(html)
    root = soup.find("main") or soup.body
    if root is None:
        return []

    docs: list[Doc] = []
    headings = [h for h in root.find_all(["h2", "h3"]) if clean(h.get_text(" "))]

    for i, h in enumerate(headings):
        title = clean(h.get_text(" "))
        parts: list[str] = []
        for sib in h.find_all_next():
            if sib in headings and sib is not h:
                break
            if sib.name in ("p", "li"):
                t = clean(sib.get_text(" "))
                if t and not BOILERPLATE.match(t) and t not in parts:
                    parts.append(t)
        body = "\n".join(parts)
        if len(body) < 20:
            continue
        docs.append(
            Doc(
                doc_id=f"terms-{i:02d}",
                url=f"{BASE}/terms",
                page="/terms",
                section="Terms & Conditions",
                title=title,
                text=f"{title}\n{body}",
                kind="terms",
            )
        )

    if not docs:  # no headings — keep the page as one doc rather than lose it
        text = clean(root.get_text(" "))
        if text:
            docs.append(
                Doc(
                    doc_id="terms-00",
                    url=f"{BASE}/terms",
                    page="/terms",
                    section="Terms & Conditions",
                    title="Terms & Conditions",
                    text=text,
                    kind="terms",
                    meta={"extraction": "whole-page-fallback"},
                )
            )
    return docs


def extract_generic(html: str, page: str) -> list[Doc]:
    soup = soupify(html)
    root = soup.find("main") or soup.body
    text = clean(root.get_text(" ")) if root else ""
    if not text:
        return []
    return [
        Doc(
            doc_id=f"{page.strip('/') or 'home'}-00",
            url=f"{BASE}{page}",
            page=page,
            section=page,
            title=page,
            text=text,
            kind="prose",
            volatile=True,
        )
    ]


# --------------------------------------------------------------------------


def drop_placeholders(docs: list[Doc]) -> tuple[list[Doc], list[dict]]:
    """Remove empty-state and sub-threshold docs, loudly.

    Applied once over the whole corpus rather than inside each extractor, so a
    new extractor cannot forget the rule. Every drop is printed and recorded in
    the manifest: if the site ever pins a real notice, the doc stops matching
    EMPTY_STATE_RE and returns to the index on the next scrape without anyone
    editing this file.
    """
    kept: list[Doc] = []
    dropped: list[dict] = []
    for d in docs:
        m = EMPTY_STATE_RE.search(d.text)
        if m:
            reason = f"empty-state placeholder ({m.group(0)!r})"
        elif d.n_chars < MIN_DOC_CHARS:
            reason = f"below MIN_DOC_CHARS ({d.n_chars} < {MIN_DOC_CHARS})"
        else:
            kept.append(d)
            continue
        print(f"  - dropped {d.doc_id}: {reason}")
        dropped.append({"doc_id": d.doc_id, "reason": reason, "text": d.text})
    return kept, dropped


def build(include_volatile: bool = False) -> dict:
    docs: list[Doc] = []
    fetched: list[str] = []

    for path, volatile in PAGES:
        if volatile and not include_volatile:
            print(f"- {path}: skipped (volatile)")
            continue
        print(f"+ {path}: fetching")
        html = fetch(path)
        fetched.append(path)

        if path == "/":
            got = extract_home(html)
        elif path == "/terms":
            got = extract_terms(html)
        else:
            got = extract_generic(html, path)

        if volatile:
            for d in got:
                d.volatile = True
        print(f"  -> {len(got)} docs, {sum(d.n_chars for d in got)} chars")
        docs += got

    if not docs:
        raise RuntimeError("corpus build produced 0 docs — refusing to write an empty index")

    docs, dropped = drop_placeholders(docs)
    if not docs:
        raise RuntimeError("all docs were dropped as placeholders — refusing to write")

    by_kind: dict[str, int] = {}
    for d in docs:
        by_kind[d.kind] = by_kind.get(d.kind, 0) + 1

    manifest = {
        "source": BASE,
        "pages_fetched": fetched,
        "n_docs": len(docs),
        "n_chars": sum(d.n_chars for d in docs),
        "by_kind": by_kind,
        "exclusions": EXCLUSIONS,
        "dropped": dropped,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "docs.json").write_text(
        json.dumps([asdict(d) for d in docs], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the hhgoa.com RAG corpus.")
    ap.add_argument(
        "--include-volatile",
        action="store_true",
        help="also index /result (live leaderboard; answers go stale)",
    )
    args = ap.parse_args()

    m = build(include_volatile=args.include_volatile)
    print("\n=== corpus ===")
    print(f"docs   : {m['n_docs']}")
    print(f"chars  : {m['n_chars']}")
    print(f"by kind: {m['by_kind']}")
    print(f"written: {OUT_DIR / 'docs.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
