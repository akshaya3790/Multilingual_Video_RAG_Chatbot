"""Targeted regressions using original functions with isolated external I/O."""
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import shutil
import html
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit
import uuid

SOURCE = Path(__file__).resolve().parent.parent / 'app_yt.py'
import youtube_source
import logging
from app_logging import new_request_id


def functions():
    scope = dict(os=os, re=re, uuid=uuid, datetime=datetime, tempfile=tempfile, shutil=shutil, EXPORT_MAX_AGE=3600,
                 parse_qs=parse_qs, urlsplit=urlsplit, SESSION_STORE={},
                 time=time, threading=threading, SESSION_LOCK=threading.Lock(),
                 SESSION_IDLE_TIMEOUT=7200, MAX_SESSIONS=50, MAX_CHATS_PER_SESSION=10,
                 SESSION_EXPIRED_MESSAGE='Session expired', CHAT_STORE=None, html=html,
                 MAX_VIDEO_MINUTES=180, MAX_TRANSCRIPT_CHARS=250000, MAX_VIDEOS_PER_HOUR=20,
                 gr=SimpleNamespace(update=lambda **kw: dict(__type__='update', **kw), Warning=Mock()),
                 get_video_metadata=Mock(return_value={'title': 'Video', 'channel': 'Channel'}),
                 get_transcript=Mock(return_value=('Transcript', 'en', 'en', [])),
                 build_hybrid_retriever=Mock(),
                 # YouTube helpers now live in youtube_source.py and are imported by app_yt.
                 extract_video_id=youtube_source.extract_video_id,
                 youtube_error_message=youtube_source.youtube_error_message,
                 new_request_id=new_request_id, log=logging.getLogger('ytchat.app'))
    selected = {'create_session', 'get_chat_choices', 'load_video',
                'new_chat', 'get_viewer_chat', 'selected_video_url', 'export_conversation_notes',
                'get_session', 'evict_sessions', 'trim_chats', 'load_saved_chats',
                'save_chat', 'delete_saved_chat', 'allow_request', 'md_text'}
    for node in ast.parse(SOURCE.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.FunctionDef) and node.name in selected:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope


class TestDebugRegressions(unittest.TestCase):
    def setUp(self):
        self.app = functions()
        self.session = self.app['create_session']()

    def test_invalid_host_nested_url_and_extra_id_are_rejected(self):
        for url in ['https://notyoutube.com/watch?v=dQw4w9WgXcQ',
                    'https://youtube.com.evil.example/watch?v=dQw4w9WgXcQ',
                    'https://example.com/?next=https://youtu.be/dQw4w9WgXcQ',
                    'https://youtu.be/dQw4w9WgXcQEXTRA',
                    'ftp://youtube.com/watch?v=dQw4w9WgXcQ',
                    'https://youtube.com/watch?v=dQw4w9WgXcQ&v=jNQXAC9IVRw']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.app['extract_video_id'](url)

    def test_supported_url_forms_remain_available(self):
        for url in ['https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=shared',
                    'http://youtube.com/embed/dQw4w9WgXcQ',
                    'https://m.youtube.com/shorts/dQw4w9WgXcQ',
                    'https://youtu.be/dQw4w9WgXcQ?t=15',
                    'youtube.com/watch?v=dQw4w9WgXcQ']:
            self.assertEqual(self.app['extract_video_id'](url), 'dQw4w9WgXcQ')

    def test_loading_second_video_selects_that_video_in_sidebar(self):
        self.app['load_video'](self.session, 'https://youtu.be/dQw4w9WgXcQ')
        second = self.app['load_video'](self.session, 'https://youtu.be/jNQXAC9IVRw')
        self.assertEqual(second[1].get('value'), second[-1])
        self.assertEqual(len(second[1]['choices']), 2)

    def test_new_chat_clears_sidebar_selection_without_losing_chats(self):
        self.app['load_video'](self.session, 'https://youtu.be/dQw4w9WgXcQ')
        self.app['new_chat'](self.session)
        choices = self.app['get_chat_choices'](self.session)
        self.assertIn('value', choices)
        self.assertIsNone(choices['value'])
        self.assertEqual(len(choices['choices']), 1)

    def test_same_title_exports_and_reexports_are_independent(self):
        store = self.app['SESSION_STORE']
        with tempfile.TemporaryDirectory() as directory, patch.object(tempfile, 'gettempdir', return_value=directory):
            paths = []
            for number in range(3):
                session = self.app['create_session']() if number < 2 else session
                store[session]['chats']['chat'] = {'metadata': {'title': 'Same Title'},
                    'messages': [{'role': 'user', 'content': f'message-{number}'}]}
                paths.append(self.app['export_conversation_notes'](session, 'chat'))
            self.assertEqual(len(set(paths)), 3)
            for number, filename in enumerate(paths):
                self.assertIn(f'message-{number}', Path(filename).read_text(encoding='utf-8'))
                self.assertEqual(Path(filename).name, 'Same_Title_notes.md')

    def test_restored_url_uses_only_the_selected_session_chat(self):
        self.app['load_video'](self.session, 'https://youtu.be/dQw4w9WgXcQ')
        chat_id = self.app['SESSION_STORE'][self.session]['active_chat']
        url_for = self.app.get('selected_video_url')
        self.assertIsNotNone(url_for, 'Restoring a conversation must also restore its video URL')
        self.assertEqual(url_for(self.session, chat_id), 'https://www.youtube.com/watch?v=dQw4w9WgXcQ')
        other = self.app['create_session']()
        self.assertEqual(url_for(other, chat_id), '')

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for the isolated frontend callback check')
    def test_download_waits_for_generated_file_and_clears_stale_file_value(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        callback = next((kw.value.value for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == 'then'
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'export_event'
            for kw in node.keywords if kw.arg == 'js' and isinstance(kw.value, ast.Constant)), None)
        self.assertIsNotNone(callback, 'A newly generated file must download after the async callback finishes')
        javascript = '''const assert = require('node:assert/strict');
        const clicks = []; let attached = false;
        global.document = {createElement: () => ({click(){assert.ok(attached); clicks.push([this.href,this.download]);}}),
          body: {appendChild(){attached=true;}, removeChild(){attached=false;}}};
        const download = CALLBACK;
        assert.deepEqual(download(null), [null]);
        assert.equal(clicks.length, 0);
        assert.deepEqual(download({url:'http://127.0.0.1:7862/gradio_api/file=example',orig_name:'notes.md'}), [null]);
        assert.deepEqual(clicks, [['http://127.0.0.1:7862/gradio_api/file=example','notes.md']]);
        assert.equal(attached,false);
        '''.replace('CALLBACK', callback)
        result = subprocess.run([shutil.which('node'), '-e', javascript], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
