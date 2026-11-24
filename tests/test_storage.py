import pytest

moto = pytest.importorskip("moto")
boto3 = pytest.importorskip("boto3")


def test_s3_roundtrip(tmp_path, monkeypatch):
    from passk_smooth import storage

    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with moto.mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="passk-test")
        d = tmp_path / "run"
        (d / "sub").mkdir(parents=True)
        (d / "counts.csv").write_text("a,b\n1,2\n")
        (d / "sub" / "x.txt").write_text("hi")
        keys = storage.sync_dir_to_s3(d, "s3://passk-test/exp1")
        assert sorted(keys) == ["exp1/counts.csv", "exp1/sub/x.txt"]
        local = storage.localize("s3://passk-test/exp1/counts.csv", cache_dir=tmp_path / "cache")
        assert local.read_text() == "a,b\n1,2\n"
