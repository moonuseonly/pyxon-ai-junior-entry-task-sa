"""Regression test for a real bug caught during manual API testing: the
uploaded file's original name was being lost in favor of the temp file's
random on-disk name. Storage is stubbed out here entirely — this test is
about ingest_file's filename-selection logic, not the database or embedder.
"""

from contextlib import contextmanager

import pytest

from app.pipeline import ingest_file


class _FakeDocument:
    id = "fake-doc-id"
    chunks: list = []


class _FakeSession:
    def flush(self):
        pass


@pytest.fixture
def captured_filenames(monkeypatch):
    captured = []

    def fake_save_document_with_chunks(session, *, filename, **kwargs):
        captured.append(filename)
        return _FakeDocument()

    @contextmanager
    def fake_get_session():
        yield _FakeSession()

    monkeypatch.setattr("app.pipeline.save_document_with_chunks", fake_save_document_with_chunks)
    monkeypatch.setattr("app.pipeline.get_session", fake_get_session)
    monkeypatch.setattr("app.pipeline.vector_store.add_chunks", lambda **kwargs: None)
    return captured


def test_ingest_file_uses_display_filename_override(tmp_path, captured_filenames):
    doc_path = tmp_path / "a1b2c3_random_temp_name.txt"
    doc_path.write_text("Sample content for the filename regression test.")

    ingest_file(str(doc_path), display_filename="original_upload.txt")

    assert captured_filenames == ["original_upload.txt"]


def test_ingest_file_falls_back_to_parsed_filename_without_override(tmp_path, captured_filenames):
    doc_path = tmp_path / "my_report.txt"
    doc_path.write_text("Sample content for the filename regression test.")

    ingest_file(str(doc_path))

    assert captured_filenames == ["my_report.txt"]
