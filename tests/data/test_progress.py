"""Progress stays sparse and does not mix with JSON command results."""

from IT3160.data import progress


def test_progress_is_throttled_and_flushes_to_stderr(monkeypatch, capsys):
    now = [0.0]
    monkeypatch.setattr(progress, "monotonic", lambda: now[0])
    status = progress.Progress("Build", "records")
    for count in range(1, 301):
        now[0] = count / 10
        status.update(count)
    now[0] = 59.9
    status.update(599)
    now[0] = 60.0
    status.update(600)
    now[0] = 60.2
    status.finish("no errors")
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 4
    assert lines[0].endswith("Build: started")
    assert lines[1].endswith("Build: 300 records")
    assert lines[2].endswith("Build: 600 records")
    assert lines[3].endswith("Build: done, 600 records in 60.2s; no errors")


def test_empty_progress_still_reports_completion(monkeypatch, capsys):
    monkeypatch.setattr(progress, "monotonic", lambda: 0.0)
    status = progress.Progress("Validate", "records")
    status.finish("errors=0")
    assert "Validate: done, 0 records in 0.0s; errors=0" in capsys.readouterr().err
