"""Everything that talks to YouTube: link parsing, video details and transcripts."""
import os
import re
from urllib.parse import parse_qs, urlsplit

from langdetect import DetectorFactory, detect, LangDetectException
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig


# langdetect is random unless seeded.
DetectorFactory.seed = 0


def youtube_proxy():
    """YouTube often blocks cloud servers; an HTTP(S) proxy such as a residential one gets around it.

    Read on each call, so a value from .env counts however early this module is imported.
    """
    return os.getenv("YOUTUBE_PROXY", "").strip()


# ============================================================
# 3. YOUTUBE UTILITIES
# ============================================================

def extract_video_id(url):
    invalid = "Invalid YouTube URL format. Please provide a standard YouTube video link, shorts URL, or youtu.be link."
    text = url.strip() if isinstance(url, str) else ""
    try:
        parsed = urlsplit(text if "://" in text else "https://" + text)
        if (parsed.scheme not in {"http", "https"} or parsed.username or parsed.password
                or parsed.port not in {None, 80, 443}):
            raise ValueError(invalid)
        parts = parsed.path.strip("/").split("/")
        video_id = None
        if parsed.hostname in {"youtu.be", "www.youtu.be"} and len(parts) == 1:
            video_id = parts[0]
        elif parsed.hostname in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
            if parts == ["watch"]:
                values = parse_qs(parsed.query).get("v", [])
                video_id = values[0] if len(values) == 1 else None
            elif len(parts) == 2 and parts[0] in {"embed", "shorts"}:
                video_id = parts[1]
        if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError(invalid)
        return video_id
    except ValueError:
        raise ValueError(invalid) from None


# ------------------------------------------------------------
# Metadata
# ------------------------------------------------------------

def get_video_metadata(url):
    options = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    proxy = youtube_proxy()
    if proxy:
        options["proxy"] = proxy
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title": info.get("title") or "Unknown title",
        "channel": info.get("channel") or info.get("uploader") or "Unknown channel",
        "channel_url": info.get("channel_url") or info.get("uploader_url") or "",
        "description": (info.get("description") or "")[:5000],
        "upload_date": info.get("upload_date") or "Unknown",
        "duration": info.get("duration"),
        "view_count": info.get("view_count"),
        "url": info.get("webpage_url") or url,
    }


def youtube_error_message(exc):
    """Explain why YouTube refused a video, instead of one generic error."""
    kinds = {cls.__name__ for cls in type(exc).__mro__}
    text = str(exc).lower()
    if kinds & {"RequestBlocked", "IpBlocked", "PoTokenRequired"} or re.search(
            r"not a bot|sign in to confirm|http error 429|too many requests", text):
        return ("🚫 **YouTube blocked this server:** YouTube is refusing requests from this "
                "server's network, which often happens on cloud hosting. Try again later, run the "
                "app on your own computer, or set `YOUTUBE_PROXY` in `.env`.")
    if kinds & {"TranscriptsDisabled", "NoTranscriptFound"} or (
            "ValueError" in kinds and "transcript" in text):
        return "📝 **No transcript:** This video has no captions or transcript, so it can't be loaded."
    if kinds & {"VideoUnavailable", "VideoUnplayable", "AgeRestricted", "InvalidVideoId"} or re.search(
            r"private video|video (?:is )?unavailable|age-restricted|confirm your age|not available in your country", text):
        return ("🔒 **Video unavailable:** This video is private, removed, age-restricted, "
                "or not available in the server's region.")
    return ("❌ **Could Not Load Video:** Please ensure the YouTube link is valid, "
            "publicly accessible, and contains a transcript.")


# ============================================================
# 4. TRANSCRIPT
# ============================================================

def get_transcript(video_id):
    """Return (text, detected language, caption track code, timed entries)."""
    proxy = youtube_proxy()
    api = YouTubeTranscriptApi(
        proxy_config=GenericProxyConfig(http_url=proxy, https_url=proxy)
    ) if proxy else YouTubeTranscriptApi()
    transcripts = list(api.list(video_id))
    if not transcripts:
        raise ValueError("No transcript is available for this video.")

    # Prefer English, then Hindi, then any other language, matching regional codes
    # such as en-US. Within a language, captions written by people beat auto-generated ones.
    preference = ["en", "hi"]

    def rank(position):
        item = transcripts[position]
        base = str(item.language_code).split("-")[0].lower()
        return (
            preference.index(base) if base in preference else len(preference),
            getattr(item, "is_generated", False) is True,
            position,
        )

    transcript = transcripts[min(range(len(transcripts)), key=rank)]
    entries = transcript.fetch().to_raw_data()
    transcript_text = " ".join(item["text"] for item in entries).strip()
    if not transcript_text:
        raise ValueError("Transcript is empty.")
    try:
        detected_language = detect(transcript_text[:12000])
    except LangDetectException:
        detected_language = transcript.language_code
    return transcript_text, detected_language, transcript.language_code, entries
