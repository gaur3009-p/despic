import re


# ── Regexes ──────────────────────────────────────────────────────────────────

_FILLERS = re.compile(
    r"\b(uh+|um+|hmm+|mhm+|huh|ugh|ahh?|ehh?|err?|"
    r"like,?\s*you\s*know|you\s*know,?|i\s*mean,?|"
    r"kind\s*of|sort\s*of|basically|literally|"
    r"actually,?\s*like|so\s*like|right\?|okay\?)\b[,\s]*",
    re.IGNORECASE,
)
_REPEATS      = re.compile(r"\b(\w+)(\s+\1)+\b", re.IGNORECASE)
_MULTI_SPACE  = re.compile(r" {2,}")
_SENT_END     = re.compile(r'([.!?]["\'»]?)\s+')
_ARTEFACTS    = re.compile(r"[\[\(][^\]\)]{0,40}[\]\)]|\*[^*]{0,40}\*")
_LEADING_JUNK = re.compile(r"^[\s,;.!?—–\-]+")


def _clean(raw: str) -> str:
    t = raw.strip()
    if not t:
        return ""
    t = _ARTEFACTS.sub(" ", t)
    t = _FILLERS.sub(" ", t)
    t = _REPEATS.sub(r"\1", t)
    t = re.sub(r"\.{2,}", "—", t)
    t = re.sub(r"\s*--\s*", "—", t)
    t = re.sub(r"\s+([,;:.!?])", r"\1", t)
    t = re.sub(r"([,;:])(?!\s)", r"\1 ", t)
    t = _MULTI_SPACE.sub(" ", t).strip()
    t = _LEADING_JUNK.sub("", t).strip()
    if t:
        t = t[0].upper() + t[1:]
    if t and t[-1] not in ".!?,;:—":
        t += "."
    return t


class TranscriptFormatter:
    """
    Stateful formatter for one session.

    push(raw, translation) — ingest one Whisper chunk + its translation
    display()              — returns (transcript_str, translation_str)
    reset()                — clear for a new session
    """

    MAX_LINE  = 85   # chars before hard-wrapping to next line
    MAX_LINES = 8    # rolling window kept on screen (YouTube uses ~2-3, we use more for context)

    def __init__(self):
        # completed display lines
        self._t_lines:  list[str] = []   # transcript
        self._tr_lines: list[str] = []   # translation (mirrors t_lines 1-for-1)
        # in-progress (not yet committed)
        self._t_cur:  str = ""
        self._tr_cur: str = ""

    def reset(self):
        self._t_lines  = []
        self._tr_lines = []
        self._t_cur    = ""
        self._tr_cur   = ""

    # ── Public ───────────────────────────────────────────────────────────────

    def push(self, raw_chunk: str, translation: str = "") -> None:
        """
        Ingest one raw Whisper chunk and its translation.
        Cleans the transcript text and flows it into display lines.
        Translation lines are committed in sync with transcript lines.
        """
        cleaned = _clean(raw_chunk)
        if not cleaned:
            return

        # Count committed lines before flowing so we know how many new ones appear
        before = len(self._t_lines)

        # Flow transcript text into lines
        joined = (self._t_cur + " " + cleaned).strip() if self._t_cur else cleaned
        self._t_cur = self._flow_transcript(joined)

        after = len(self._t_lines)
        new_lines = after - before

        # Mirror: one translation entry per newly committed transcript line
        for _ in range(new_lines):
            self._tr_lines.append(translation.strip() if translation else "")

        # Update the in-progress translation (shown alongside _t_cur)
        self._tr_cur = translation.strip() if translation else ""

    def display(self) -> tuple[str, str]:
        """
        Return (transcript_display, translation_display).
        Both are the same rolling MAX_LINES window so they stay in sync.
        """
        # All lines including in-progress
        all_t  = self._t_lines  + ([self._t_cur]  if self._t_cur  else [])
        all_tr = self._tr_lines + ([self._tr_cur] if self._t_cur  else [])

        # Rolling window
        all_t  = all_t [-self.MAX_LINES:]
        all_tr = all_tr[-self.MAX_LINES:]

        return "\n".join(all_t), "\n".join(all_tr)

    def full_text(self) -> str:
        """Full raw transcript text for the session (for logging etc.)."""
        all_t = self._t_lines + ([self._t_cur] if self._t_cur else [])
        return " ".join(all_t)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _flow_transcript(self, text: str) -> str:
        """
        Break text at sentence boundaries into committed lines.
        Returns the un-committed remainder (new _t_cur).
        """
        parts = _SENT_END.split(text)
        buf = ""
        i = 0
        while i < len(parts):
            seg = parts[i]
            # Attach the punctuation token that follows
            if i + 1 < len(parts) and re.match(r'^[.!?]["\'»]?$', parts[i + 1]):
                seg = seg + parts[i + 1]
                i += 2
            else:
                i += 1

            candidate = (buf + " " + seg).strip() if buf else seg.strip()

            if len(candidate) >= self.MAX_LINE:
                if buf.strip():
                    self._commit_t(buf.strip())
                # Hard-wrap if single segment is still too long
                while len(seg.strip()) > self.MAX_LINE:
                    wrap_at = seg.rfind(" ", 0, self.MAX_LINE)
                    if wrap_at == -1:
                        wrap_at = self.MAX_LINE
                    self._commit_t(seg[:wrap_at].strip())
                    seg = seg[wrap_at:].strip()
                buf = seg
            else:
                buf = candidate

            # Commit on natural sentence end (decent length)
            if buf and buf[-1] in ".!?" and len(buf) > 18:
                self._commit_t(buf)
                buf = ""

        return buf

    def _commit_t(self, line: str) -> None:
        if line.strip():
            self._t_lines.append(line.strip())


# ── Convenience: format a single complete text block ─────────────────────────

def format_full_text(raw: str) -> str:
    """
    Format a full Whisper output (sentence pipeline) into clean readable text.
    Returns a multi-line string with proper capitalisation and punctuation.
    No rolling window — returns everything.
    """
    fmt = TranscriptFormatter()
    # Split on existing sentence boundaries and push each
    sentences = re.split(r'(?<=[.!?])\s+', raw.strip())
    for s in sentences:
        if s.strip():
            fmt.push(s)
    # Return full text (ignore rolling window for one-shot use)
    all_lines = fmt._t_lines + ([fmt._t_cur] if fmt._t_cur else [])
    return "\n".join(all_lines)
