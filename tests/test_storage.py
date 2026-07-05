from frank_reader.storage import Storage


def test_create_get_list_job(tmp_path):
    storage = Storage(tmp_path)
    storage.create_job("job1", "pdf", "doc.pdf", "ru")
    job = storage.get_job("job1")
    assert job["status"] == "pending"
    assert job["source_name"] == "doc.pdf"

    storage.create_job("job2", "text", "text", "ru")
    jobs = storage.list_jobs()
    assert [j["id"] for j in jobs] == ["job2", "job1"]  # newest first


def test_get_missing_job_returns_none(tmp_path):
    storage = Storage(tmp_path)
    assert storage.get_job("missing") is None


def test_set_job_status(tmp_path):
    storage = Storage(tmp_path)
    storage.create_job("job1", "pdf", "doc.pdf", "ru")
    storage.set_job_status("job1", "running", total_pages=5)
    job = storage.get_job("job1")
    assert job["status"] == "running"
    assert job["total_pages"] == 5

    storage.set_job_status("job1", "failed", error="boom")
    job = storage.get_job("job1")
    assert job["status"] == "failed"
    assert job["error"] == "boom"


def test_upsert_and_get_pages(tmp_path):
    storage = Storage(tmp_path)
    storage.create_job("job1", "pdf", "doc.pdf", "ru")
    storage.upsert_page("job1", 1, "pending")
    storage.upsert_page("job1", 2, "pending")
    storage.upsert_page("job1", 1, "done", result_json='{"a": 1}', cache_key="k1")

    pages = storage.get_pages("job1")
    assert [p["page_number"] for p in pages] == [1, 2]
    assert pages[0]["status"] == "done"
    assert pages[0]["result_json"] == '{"a": 1}'
    assert pages[1]["status"] == "pending"


def test_cache_put_get(tmp_path):
    storage = Storage(tmp_path)
    assert storage.cache_get("missing") is None
    storage.cache_put("key1", '{"x": 1}')
    assert storage.cache_get("key1") == '{"x": 1}'


def test_running_marked_interrupted_on_restart(tmp_path):
    storage = Storage(tmp_path)
    storage.create_job("job1", "pdf", "doc.pdf", "ru")
    storage.set_job_status("job1", "running")
    storage.create_job("job2", "pdf", "doc2.pdf", "ru")
    storage.set_job_status("job2", "done")

    storage.mark_running_as_interrupted()

    assert storage.get_job("job1")["status"] == "interrupted"
    assert storage.get_job("job2")["status"] == "done"


def test_job_dir_created(tmp_path):
    storage = Storage(tmp_path)
    d = storage.job_dir("job1")
    assert d.exists()
    assert d == tmp_path / "jobs" / "job1"
