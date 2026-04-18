from napari_sam3_assistant.services.checkpoint_service import CheckpointService


def test_checkpoint_service_accepts_sam31_multiplex_weight(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "processor_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "sam3.1_multiplex.pt").write_bytes(b"")

    result = CheckpointService().validate(str(tmp_path), model_type="sam3.1")

    assert result.ok


def test_checkpoint_service_accepts_sam30_weight_for_sam30_type(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "processor_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "sam3.pt").write_bytes(b"")

    result = CheckpointService().validate(str(tmp_path), model_type="sam3")

    assert result.ok


def test_checkpoint_service_rejects_sam31_weight_for_sam30_type(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "processor_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "sam3.1_multiplex.pt").write_bytes(b"")

    result = CheckpointService().validate(str(tmp_path), model_type="sam3")

    assert not result.ok
    assert "sam3.pt" in result.message
