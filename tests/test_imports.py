def test_import_package():
    import napari_sam3_assistant
    assert napari_sam3_assistant.__version__ == "4.5.0"


def test_import_widget():
    from napari_sam3_assistant.widgets.main_widget import MainWidget  # noqa: F401
