import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import internet_search as web


class DownloadTests(unittest.TestCase):
    def test_failed_first_results_do_not_hide_later_page(self):
        results = [{"url": f"https://example.com/{i}"} for i in range(6)]
        def fetch(url, **kwargs):
            if not url.endswith("/5"):
                raise web.InternetSearchError("HTTP 403")
            return {"url": url, "text": "A usable page."}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(web, "CACHE_PATH", Path(directory) / "cache.json"), \
                patch.object(web, "search", return_value=results), \
                patch.object(web, "fetch_page", side_effect=fetch):
            pages = web.search_pages("test")
        self.assertEqual(pages[0]["url"], results[5]["url"])

    def test_slow_page_does_not_discard_successful_page(self):
        def fetch(url, **kwargs):
            if url.endswith("slow"):
                time.sleep(0.6)
            return {"url": url, "text": "A usable page."}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(web, "CACHE_PATH", Path(directory) / "cache.json"), \
                patch.object(web, "search", return_value=[{"url": "https://example.com/slow"}, {"url": "https://example.com/fast"}]), \
                patch.object(web, "fetch_page", side_effect=fetch):
            pages = web.search_pages("test", timeout=0.3)
        self.assertEqual(pages[0]["url"], "https://example.com/fast")

    def test_unicode_path_is_encoded(self):
        class Response:
            status = 200
            def getheader(self, name, default=None):
                return "text/plain; charset=utf-8"
            def read(self, size):
                return b"Readable content."
        with patch.object(web, "_public_addresses", return_value=["93.184.216.34"]), \
                patch.object(web, "_PinnedHTTPSConnection") as connection:
            connection.return_value.getresponse.return_value = Response()
            web.fetch_page("https://example.com/sécurité?q=santé")
            path = connection.return_value.request.call_args.args[1]
        self.assertEqual(path, "/s%C3%A9curit%C3%A9?q=sant%C3%A9")


if __name__ == "__main__":
    unittest.main()
