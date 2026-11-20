import datetime as dt
import io
import json
import re
import threading

from archive_app import create_app
from archive_app.catalog import (
    MAX_SPAN_DAYS,
    STATUS_BLOCKED,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_PENDING,
    Event,
    MockCatalogSource,
    Store,
)
from archive_app.collector import collect, contiguous_segments
from archive_app.csv_import import CsvValidationError, parse_csv, verify_includes_collected


def write_catalog(tmp_path, events):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"events": events}), encoding="utf-8")
    return str(path)


def ev(eid, day, hh=0, mm=0, mag=3.0):
    return {"id": eid, "time": f"{day}T{hh:02d}:{mm:02d}:00Z", "magnitude": mag}


def make_store(tmp_path):
    store = Store(str(tmp_path / "archive.db"))
    store.init()
    return store


def statuses(store, run_id):
    return {row["day"]: row["status"] for row in store.days(run_id)}


def test_contiguous_segments():
    d = dt.date(2024, 1, 1)
    segs = contiguous_segments([d, d + dt.timedelta(days=1), d + dt.timedelta(days=5)])
    assert len(segs) == 2
    assert len(segs[0]) == 2 and len(segs[1]) == 1


def test_under_full_range_marks_every_day_complete_including_empty(tmp_path):
    path = write_catalog(tmp_path, [ev("A", "2024-01-02"), ev("B", "2024-01-03")])
    store = make_store(tmp_path)
    run = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    summary = collect(store, MockCatalogSource(path), run)
    assert summary.completed == 5
    assert all(v == STATUS_COMPLETE for v in statuses(store, run).values())
    assert [r["event_id"] for r in store.events(run)] == ["A", "B"]


def test_full_page_splits_until_exhausted(tmp_path):
    # 25 events inside a 3-day range: the whole range is full (20), splitting
    # yields a 1-day + 2-day pair that are both under the limit.
    events = [ev(f"D1-{i}", "2024-01-01", i % 24, i) for i in range(8)]
    events += [ev(f"D2-{i}", "2024-01-02", i % 24, i) for i in range(9)]
    events += [ev(f"D3-{i}", "2024-01-03", i % 24, i) for i in range(8)]
    path = write_catalog(tmp_path, events)
    store = make_store(tmp_path)
    run = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 3))
    summary = collect(store, MockCatalogSource(path), run)
    assert summary.blocked == 0
    assert all(v == STATUS_COMPLETE for v in statuses(store, run).values())
    assert len(store.events(run)) == 25


def test_single_full_day_is_blocked_and_keeps_first_page(tmp_path):
    events = [ev(f"X{i:02d}", "2024-01-02", i % 24, i) for i in range(27)]
    path = write_catalog(tmp_path, events)
    store = make_store(tmp_path)
    run = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 3))
    collect(store, MockCatalogSource(path), run)
    st = statuses(store, run)
    assert st["2024-01-02"] == STATUS_BLOCKED
    assert st["2024-01-01"] == STATUS_COMPLETE
    assert st["2024-01-03"] == STATUS_COMPLETE
    assert len(store.events(run, "2024-01-02")) == 20


def test_failed_request_marks_days_and_retry_skips_complete_and_blocked(tmp_path):
    events = [ev("A", "2024-01-01")]
    events += [ev(f"X{i:02d}", "2024-01-02", i % 24, i) for i in range(25)]
    path = write_catalog(tmp_path, events)
    store = make_store(tmp_path)
    run = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 3))

    # The upstream rejects the whole window; none of its days can be considered
    # archived, so every non-complete day is marked for retry.
    summary = collect(store, MockCatalogSource(path, fail_dates=["2024-01-03"]), run)
    assert summary.failed == 3
    assert set(statuses(store, run).values()) == {STATUS_FAILED}

    # Retry: the full range is split, day 1/3 complete and day 2 is blocked.
    summary = collect(store, MockCatalogSource(path), run)
    assert summary.failed == 0
    st = statuses(store, run)
    assert st["2024-01-01"] == STATUS_COMPLETE
    assert st["2024-01-02"] == STATUS_BLOCKED
    assert st["2024-01-03"] == STATUS_COMPLETE

    # A further retry must not request anything: complete and blocked days are
    # skipped, and the blocked day keeps its first-page snapshot.
    before = [r["event_id"] for r in store.events(run, "2024-01-02")]
    calls = []

    class Recording(MockCatalogSource):
        def fetch_page(self, start, end):
            calls.append((start, end))
            return super().fetch_page(start, end)

    summary = collect(store, Recording(path), run)
    assert calls == []
    assert summary.completed == summary.blocked == summary.failed == 0
    assert before == [r["event_id"] for r in store.events(run, "2024-01-02")]


def test_retry_is_noop_when_everything_done(tmp_path):
    path = write_catalog(tmp_path, [ev("A", "2024-01-01")])
    store = make_store(tmp_path)
    run = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    collect(store, MockCatalogSource(path), run)
    calls = []

    class Recording(MockCatalogSource):
        def fetch_page(self, start, end):
            calls.append((start, end))
            return super().fetch_page(start, end)

    collect(store, Recording(path), run)
    assert calls == []


def test_run_rejects_more_than_max_span():
    assert MAX_SPAN_DAYS == 31
    # App-level enforcement is covered through the Flask test client below.


def test_csv_parse_accepts_header_and_canonicalises(tmp_path):
    data = (
        "event_id,utc_time,magnitude\n"
        "E1,2024-02-01T05:06:07Z,3.4\n"
        "E2,2024-02-01 23:59:59,1.2\n"
    ).encode()
    parsed = parse_csv(data, "2024-02-01")
    assert [e.event_id for e in parsed.events] == ["E1", "E2"]
    assert parsed.events[0].utc_time == "2024-02-01 05:06:07"


def test_csv_rejects_other_day_duplicates_and_bad_values():
    data = (
        "event_id,utc_time,magnitude\n"
        "E1,2024-02-01T05:06:07Z,3.4\n"
        "E1,2024-02-01T06:00:00Z,2.0\n"
        "E2,2024-02-02T00:00:00Z,1.0\n"
        "E3,2024-02-01T00:00:00Z,not-a-number\n"
    ).encode()
    try:
        parse_csv(data, "2024-02-01")
    except CsvValidationError as exc:
        text = " ".join(exc.errors)
        assert "重复" in text
        assert "不属于" in text
        assert "震级无效" in text
    else:  # pragma: no cover
        raise AssertionError("expected validation failure")


def test_csv_without_header_uses_positional_columns():
    parsed = parse_csv(b"E1,2024-02-01T01:02:03Z,4.1\n", "2024-02-01")
    assert parsed.events[0].event_id == "E1"


def test_verify_includes_collected_requires_verbatim_rows():
    parsed = parse_csv(b"E1,2024-02-01T01:00:00Z,3.0\n", "2024-02-01")
    verify_includes_collected(parsed, [Event("E1", "2024-02-01 01:00:00", 3.0)])
    try:
        verify_includes_collected(parsed, [Event("E1", "2024-02-01 01:00:00", 4.0)])
    except CsvValidationError:
        pass
    else:  # pragma: no cover
        raise AssertionError("magnitude change must be rejected")


# ---------------------------------------------------------------------------
# Flask flows
# ---------------------------------------------------------------------------


def make_client(tmp_path, events, fail_dates=()):
    path = write_catalog(tmp_path, events)
    app = create_app(
        database=str(tmp_path / "web.db"),
        source=MockCatalogSource(path, fail_dates=fail_dates),
        testing=True,
    )
    return app.test_client(), app


def upload(client, run_id, day, content, filename="day.csv", epoch=None):
    if epoch is None:
        page = client.get(f"/runs/{run_id}").get_data(as_text=True)
        match = re.search(
            rf'action="/runs/{run_id}/days/{re.escape(day)}/upload"[^>]*>.*?'
            r'name="epoch" value="(\d+)"',
            page,
            re.S,
        )
        assert match is not None, "upload form for the day must render an epoch"
        epoch = match.group(1)
    return client.post(
        f"/runs/{run_id}/days/{day}/upload",
        data={
            "epoch": str(epoch),
            "csv": (io.BytesIO(content.encode("utf-8")), filename),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )


def test_web_end_to_end_blocked_day_backfill_and_download(tmp_path):
    events = [ev("Q", "2024-01-01")]
    events += [ev(f"X{i:02d}", "2024-01-02", i % 24, i) for i in range(23)]
    client, _ = make_client(tmp_path, events)

    resp = client.post(
        "/runs", data={"start_date": "2024-01-01", "end_date": "2024-01-03"}
    )
    assert resp.status_code == 302
    run_id = 1

    client.post(f"/runs/{run_id}/collect")
    page = client.get(f"/runs/{run_id}").get_data(as_text=True)
    assert "待补档" in page
    assert "下载完整目录" in page

    # Download is blocked until the blocked day is backfilled.
    blocked = client.get(f"/runs/{run_id}/download")
    assert blocked.status_code == 302

    rows = "event_id,utc_time,magnitude\n"
    for i in range(23):
        rows += f"X{i:02d},2024-01-02T{i % 24:02d}:{i:02d}:00Z,3.0\n"

    resp = upload(client, run_id, "2024-01-02", rows)
    assert resp.status_code == 200
    page = resp.get_data(as_text=True)
    assert "已补档" in page
    assert "下载完整目录" in page

    out = client.get(f"/runs/{run_id}/download")
    assert out.status_code == 200
    text = out.get_data(as_text=True)
    lines = text.strip().splitlines()
    assert lines[0] == "event_id,utc_time,magnitude"
    assert len(lines) == 1 + 24  # 1 event on day 1 + 23 backfilled


def test_web_invalid_upload_preserves_progress(tmp_path):
    events = [ev(f"X{i:02d}", "2024-01-02", i % 24, i) for i in range(22)]
    client, app = make_client(tmp_path, events)
    client.post("/runs", data={"start_date": "2024-01-01", "end_date": "2024-01-02"})
    client.post("/runs/1/collect")
    store = app.extensions["store"]
    before = [dict(r) for r in store.events(1, "2024-01-02")]
    assert len(before) == 20

    # Missing an already-collected event -> rejected, progress untouched.
    bad = "event_id,utc_time,magnitude\nX00,2024-01-02T00:00:00Z,3.0\n"
    resp = upload(client, 1, "2024-01-02", bad)
    assert "校验失败" in resp.get_data(as_text=True)
    assert [dict(r) for r in store.events(1, "2024-01-02")] == before
    assert statuses(store, 1)["2024-01-02"] == STATUS_BLOCKED


def test_web_rejects_over_31_days_and_non_pending_upload(tmp_path):
    client, _ = make_client(tmp_path, [])
    resp = client.post(
        "/runs", data={"start_date": "2024-01-01", "end_date": "2024-02-01"}
    )
    assert resp.status_code == 302
    assert b"/runs/" not in resp.headers["Location"].encode()
    assert "最多 31" in client.get("/").get_data(as_text=True)

    client.post("/runs", data={"start_date": "2024-01-01", "end_date": "2024-01-01"})
    client.post("/runs/1/collect")
    resp = upload(client, 1, "2024-01-01", "E1,2024-01-01T00:00:00Z,1.0\n", epoch=0)
    assert "无需上传" in resp.get_data(as_text=True)


def test_web_failed_day_retry_flow(tmp_path):
    events = [ev("A", "2024-01-01")]
    path = write_catalog(tmp_path, events)
    app = create_app(
        database=str(tmp_path / "web.db"),
        source=MockCatalogSource(path, fail_dates=["2024-01-02"]),
        testing=True,
    )
    client = app.test_client()
    client.post("/runs", data={"start_date": "2024-01-01", "end_date": "2024-01-02"})
    client.post("/runs/1/collect")
    page = client.get("/runs/1").get_data(as_text=True)
    assert "待重试" in page

    app.extensions["source"] = MockCatalogSource(path)  # upstream recovered
    client.post("/runs/1/collect")
    assert statuses(app.extensions["store"], 1)["2024-01-02"] == STATUS_COMPLETE
    assert client.get("/runs/1/download").status_code == 200


def test_web_events_listing_shows_required_fields(tmp_path):
    client, _ = make_client(tmp_path, [ev("Z1", "2024-01-01", 3, 4, 2.5)])
    client.post("/runs", data={"start_date": "2024-01-01", "end_date": "2024-01-01"})
    client.post("/runs/1/collect")
    page = client.get("/runs/1").get_data(as_text=True)
    assert "Z1" in page
    assert "2024-01-01 03:04:00" in page
    assert "2.5" in page


# ---------------------------------------------------------------------------
# Rescheduling
# ---------------------------------------------------------------------------


def test_reschedule_keeps_retained_progress_and_resets_new_days(tmp_path):
    path = write_catalog(tmp_path, [ev("A", "2024-01-02"), ev("B", "2024-01-03")])
    store = make_store(tmp_path)
    run = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    collect(store, MockCatalogSource(path), run)
    assert all(v == STATUS_COMPLETE for v in statuses(store, run).values())
    snapshot = {d: dict(r) for d, r in ((r["day"], r) for r in store.days(run))}

    # Slide the window one day later: 01-01 leaves, 01-06 joins.
    result = store.reschedule_run(run, dt.date(2024, 1, 2), dt.date(2024, 1, 6))
    assert result.removed == ["2024-01-01"]
    assert result.added == ["2024-01-06"]
    assert result.retained == [f"2024-01-0{i}" for i in range(2, 6)]
    assert result.epoch == 1

    st = statuses(store, run)
    assert set(st) == {f"2024-01-0{i}" for i in range(2, 7)}
    assert st["2024-01-06"] == STATUS_PENDING
    # Retained days keep status, counts and the same epoch.
    for day in result.retained:
        row = next(r for r in store.days(run) if r["day"] == day)
        assert row["status"] == snapshot[day]["status"]
        assert row["event_count"] == snapshot[day]["event_count"]
        assert row["epoch"] == snapshot[day]["epoch"] == 0
    assert [r["event_id"] for r in store.events(run)] == ["A", "B"]


def test_reschedule_removes_events_and_frees_their_ids(tmp_path):
    # Two days share no IDs; after removing the first day its ID must become
    # reusable when the day is re-added later and collected again.
    events = [ev("SHARED", "2024-01-01"), ev("KEEP", "2024-01-02")]
    path = write_catalog(tmp_path, events)
    store = make_store(tmp_path)
    run = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 2))
    collect(store, MockCatalogSource(path), run)
    assert {r["event_id"] for r in store.events(run)} == {"SHARED", "KEEP"}

    store.reschedule_run(run, dt.date(2024, 1, 2), dt.date(2024, 1, 3))
    assert [r["event_id"] for r in store.events(run)] == ["KEEP"]
    assert "2024-01-01" not in statuses(store, run)

    # Re-add 01-01: it starts pending and a fresh collection re-inserts SHARED
    # without a primary-key clash, proving the ID was released.
    store.reschedule_run(run, dt.date(2024, 1, 1), dt.date(2024, 1, 3))
    assert statuses(store, run)["2024-01-01"] == STATUS_PENDING
    collect(store, MockCatalogSource(path), run)
    assert statuses(store, run)["2024-01-01"] == STATUS_COMPLETE
    assert {r["event_id"] for r in store.events(run)} == {"SHARED", "KEEP"}


def test_reschedule_rejects_over_31_days_and_bad_order_without_changes(tmp_path):
    store = make_store(tmp_path)
    run = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    before = statuses(store, run)
    for start, end in (
        (dt.date(2024, 1, 1), dt.date(2024, 2, 1)),  # 32 days
        (dt.date(2024, 1, 6), dt.date(2024, 1, 1)),  # end before start
    ):
        try:
            store.reschedule_run(run, start, end)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("reschedule must be rejected")
    # Saved range, progress and run epoch are untouched.
    assert statuses(store, run) == before
    row = store.get_run(run)
    assert row["start_date"] == "2024-01-01"
    assert row["end_date"] == "2024-01-05"
    assert row["epoch"] == 0


def test_reschedule_same_range_is_a_noop(tmp_path):
    store = make_store(tmp_path)
    run = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 3))
    result = store.reschedule_run(run, dt.date(2024, 1, 1), dt.date(2024, 1, 3))
    assert not result.changed
    assert result.epoch == 0
    assert all(r["epoch"] == 0 for r in store.days(run))


def test_readded_day_gets_fresh_epoch_even_with_same_date(tmp_path):
    store = make_store(tmp_path)
    run = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 2))
    store.reschedule_run(run, dt.date(2024, 1, 2), dt.date(2024, 1, 3))
    result = store.reschedule_run(run, dt.date(2024, 1, 1), dt.date(2024, 1, 3))
    epochs = {r["day"]: r["epoch"] for r in store.days(run)}
    assert epochs["2024-01-01"] == 2  # removed at epoch 1, re-added at epoch 2
    assert epochs["2024-01-02"] == 0  # never left: old requests stay valid
    assert epochs["2024-01-03"] == 1
    assert store.get_run(run)["epoch"] == 2


def test_in_flight_collection_skips_days_removed_during_request(tmp_path):
    events = [ev("A", "2024-01-01"), ev("B", "2024-01-02")]
    path = write_catalog(tmp_path, events)
    store = make_store(tmp_path)
    run = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 2))

    started = threading.Event()
    release = threading.Event()

    class SlowSource(MockCatalogSource):
        def fetch_page(self, start, end):
            started.set()
            assert release.wait(timeout=5)
            return super().fetch_page(start, end)

    worker = threading.Thread(
        target=collect, args=(store, SlowSource(path), run), daemon=True
    )
    worker.start()
    assert started.wait(timeout=5)
    # Reschedule while the single 2-day request is in flight: day 1 is removed.
    store.reschedule_run(run, dt.date(2024, 1, 2), dt.date(2024, 1, 3))
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()

    st = statuses(store, run)
    assert "2024-01-01" not in st                 # removed day gets no writes
    assert st["2024-01-02"] == STATUS_COMPLETE   # retained day still completes
    assert st["2024-01-03"] == STATUS_PENDING    # newly added, not retro-queried
    assert [r["event_id"] for r in store.events(run)] == ["B"]


def test_pre_reschedule_request_remains_valid_for_retained_day(tmp_path):
    # A blocked day retained across a reschedule keeps its epoch, so the first
    # page collected before the reschedule stays on file and its upload form
    # remains valid.
    events = [ev(f"X{i:02d}", "2024-01-02", i % 24, i) for i in range(23)]
    path = write_catalog(tmp_path, events)
    store = make_store(tmp_path)
    run = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 3))
    collect(store, MockCatalogSource(path), run)
    assert statuses(store, run)["2024-01-02"] == STATUS_BLOCKED
    blocked_epoch = next(
        r["epoch"] for r in store.days(run) if r["day"] == "2024-01-02"
    )
    store.reschedule_run(run, dt.date(2024, 1, 1), dt.date(2024, 1, 4))
    row = next(r for r in store.days(run) if r["day"] == "2024-01-02")
    assert row["status"] == STATUS_BLOCKED and row["epoch"] == blocked_epoch
    assert len(store.events(run, "2024-01-02")) == 20

    rows = "event_id,utc_time,magnitude\n"
    for i in range(23):
        rows += f"X{i:02d},2024-01-02T{i % 24:02d}:{i:02d}:00Z,3.0\n"
    parsed = parse_csv(rows.encode(), "2024-01-02")
    store.backfill_day(
        run,
        "2024-01-02",
        parsed.events,
        epoch=blocked_epoch,
        verify=lambda existing: verify_includes_collected(parsed, existing),
    )
    assert statuses(store, run)["2024-01-02"] == STATUS_COMPLETE


def test_backfill_with_old_epoch_rejected_even_when_day_readded(tmp_path):
    events = [ev(f"X{i:02d}", "2024-01-02", i % 24, i) for i in range(22)]
    path = write_catalog(tmp_path, events)
    store = make_store(tmp_path)
    run = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 3))
    collect(store, MockCatalogSource(path), run)
    old_epoch = next(
        r["epoch"] for r in store.days(run) if r["day"] == "2024-01-02"
    )
    # Remove the blocked day, then re-add it. It must restart as pending.
    store.reschedule_run(run, dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    store.reschedule_run(run, dt.date(2024, 1, 1), dt.date(2024, 1, 3))
    row = next(r for r in store.days(run) if r["day"] == "2024-01-02")
    assert row["status"] == STATUS_PENDING and row["epoch"] != old_epoch

    from archive_app.catalog import StaleDayError

    try:
        store.backfill_day(run, "2024-01-02", [], epoch=old_epoch)
    except StaleDayError:
        pass
    else:  # pragma: no cover
        raise AssertionError("old form epoch must be rejected")


def test_web_reschedule_flow_progress_events_and_flash(tmp_path):
    events = [ev("A", "2024-01-02"), ev("B", "2024-01-03")]
    client, app = make_client(tmp_path, events)
    client.post("/runs", data={"start_date": "2024-01-01", "end_date": "2024-01-04"})
    client.post("/runs/1/collect")
    store = app.extensions["store"]

    resp = client.post(
        "/runs/1/reschedule",
        data={"start_date": "2024-01-02", "end_date": "2024-01-05"},
        follow_redirects=True,
    )
    page = resp.get_data(as_text=True)
    assert "改期已保存" in page
    assert "移出 1 天" in page and "纳入 1 天" in page and "保留 3 天" in page

    st = statuses(store, 1)
    assert "2024-01-01" not in st
    assert st["2024-01-05"] == STATUS_PENDING
    assert st["2024-01-02"] == STATUS_COMPLETE
    assert [r["event_id"] for r in store.events(1)] == ["A", "B"]
    run = store.get_run(1)
    assert run["start_date"] == "2024-01-02" and run["end_date"] == "2024-01-05"

    # New day must be queried before download is offered.
    assert client.get("/runs/1/download").status_code == 302
    client.post("/runs/1/collect")
    out = client.get("/runs/1/download")
    assert out.status_code == 200
    assert (
        out.headers["Content-Disposition"]
        == 'attachment; filename="catalog_2024-01-02_2024-01-05.csv"'
    )
    lines = out.get_data(as_text=True).strip().splitlines()
    assert lines[0] == "event_id,utc_time,magnitude"
    assert len(lines) == 3  # header + A + B


def test_web_reschedule_rejects_32_days_and_keeps_saved_state(tmp_path):
    client, app = make_client(tmp_path, [ev("A", "2024-01-01")])
    client.post("/runs", data={"start_date": "2024-01-01", "end_date": "2024-01-02"})
    client.post("/runs/1/collect")
    resp = client.post(
        "/runs/1/reschedule",
        data={"start_date": "2024-01-01", "end_date": "2024-02-01"},
        follow_redirects=True,
    )
    assert "改期未保存" in resp.get_data(as_text=True)
    run = app.extensions["store"].get_run(1)
    assert run["start_date"] == "2024-01-01" and run["end_date"] == "2024-01-02"
    assert statuses(app.extensions["store"], 1)["2024-01-01"] == STATUS_COMPLETE


def test_web_stale_backfill_form_rejected_after_remove_and_readd(tmp_path):
    events = [ev(f"X{i:02d}", "2024-01-02", i % 24, i) for i in range(22)]
    client, app = make_client(tmp_path, events)
    client.post("/runs", data={"start_date": "2024-01-01", "end_date": "2024-01-03"})
    client.post("/runs/1/collect")
    page = client.get("/runs/1").get_data(as_text=True)
    match = re.search(
        r'action="/runs/1/days/2024-01-02/upload"[^>]*>.*?'
        r'name="epoch" value="(\d+)"',
        page,
        re.S,
    )
    assert match
    old_epoch = match.group(1)

    client.post(
        "/runs/1/reschedule",
        data={"start_date": "2024-01-01", "end_date": "2024-01-01"},
    )
    client.post(
        "/runs/1/reschedule",
        data={"start_date": "2024-01-01", "end_date": "2024-01-03"},
    )
    # The day exists again but starts pending; an old form's epoch is rejected.
    rows = "event_id,utc_time,magnitude\n"
    for i in range(22):
        rows += f"X{i:02d},2024-01-02T{i % 24:02d}:{i:02d}:00Z,3.0\n"
    resp = upload(client, 1, "2024-01-02", rows, epoch=old_epoch)
    assert "补档表单已因改期失效" in resp.get_data(as_text=True)
    store = app.extensions["store"]
    assert statuses(store, 1)["2024-01-02"] == STATUS_PENDING
    assert store.events(1, "2024-01-02") == []

    # A form missing the epoch token entirely is rejected as well.
    resp = client.post(
        "/runs/1/days/2024-01-02/upload",
        data={"csv": (io.BytesIO(rows.encode()), "day.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert "补档表单已因改期失效" in resp.get_data(as_text=True)


def test_web_run_page_renders_single_saved_state(tmp_path):
    client, _ = make_client(tmp_path, [ev("A", "2024-01-02")])
    client.post("/runs", data={"start_date": "2024-01-01", "end_date": "2024-01-03"})
    client.post("/runs/1/collect")
    client.post(
        "/runs/1/reschedule",
        data={"start_date": "2024-01-02", "end_date": "2024-01-04"},
    )
    page = client.get("/runs/1").get_data(as_text=True)
    # Range from the saved run row, progress rows and events all agree.
    assert "2024-01-02 ~ 2024-01-04" in page
    assert "改期" in page
    assert 'action="/runs/1/reschedule"' in page
    # The removed day must be absent from the saved progress table and the
    # events section (flash messages about earlier actions may still quote it).
    body = page.split("逐日进度", 1)[1]
    assert "2024-01-01" not in body
    assert "2024-01-04" in body      # newly added pending day present


def test_schema_migrates_databases_created_without_epoch(tmp_path):
    import sqlite3  # noqa: PLC0415
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, start_date TEXT NOT NULL,
            end_date TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE day_status (run_id INTEGER NOT NULL, day TEXT NOT NULL, status TEXT NOT NULL,
            event_count INTEGER NOT NULL DEFAULT 0, checked_at TEXT, last_error TEXT,
            PRIMARY KEY (run_id, day));
        CREATE TABLE events (run_id INTEGER NOT NULL, day TEXT NOT NULL, event_id TEXT NOT NULL,
            utc_time TEXT NOT NULL, magnitude REAL NOT NULL, source TEXT NOT NULL,
            PRIMARY KEY (run_id, event_id));
        INSERT INTO runs VALUES (1, '2024-01-01', '2024-01-02', 't');
        INSERT INTO day_status(run_id, day, status) VALUES (1, '2024-01-01', 'complete');
        INSERT INTO day_status(run_id, day, status) VALUES (1, '2024-01-02', 'pending');
        """
    )
    conn.commit()
    conn.close()

    store = Store(str(db_path))
    store.init()
    rows = {r["day"]: r["epoch"] for r in store.days(1)}
    assert rows == {"2024-01-01": 0, "2024-01-02": 0}
    result = store.reschedule_run(1, dt.date(2024, 1, 1), dt.date(2024, 1, 3))
    assert result.epoch == 1
    assert next(r for r in store.days(1) if r["day"] == "2024-01-03")["epoch"] == 1
