import numpy as np
import pytest

from camera.capture import CameraManager, depth_to_colormap


def test_depth_to_colormap_keeps_invalid_pixels_black():
    depth = np.array([[0, 99, 100, 2500, 5000, 5001]], dtype=np.float32)
    image, valid = depth_to_colormap(depth, 100, 5000)

    assert image.shape == (1, 6, 3)
    assert valid.tolist() == [[False, False, True, True, True, False]]
    assert np.all(image[0, 0] == 0)
    assert np.all(image[0, 1] == 0)
    assert np.all(image[0, 5] == 0)
    assert np.any(image[0, 3] != 0)


def test_depth_to_colormap_validates_range():
    with pytest.raises(ValueError):
        depth_to_colormap(np.zeros((2, 2)), 5000, 1000)
    with pytest.raises(ValueError):
        depth_to_colormap(np.zeros((2, 2, 1)), 100, 5000)


def test_manager_settings_are_bounded():
    manager = CameraManager()
    assert manager.update_settings(depth_max_mm=6500, jpeg_quality=90) == {
        "depth_max_mm": 6500,
        "jpeg_quality": 90,
    }
    with pytest.raises(ValueError):
        manager.update_settings(depth_max_mm=999)
    with pytest.raises(ValueError):
        manager.update_settings(jpeg_quality=49)


def test_manager_targets_full_depth_resolution():
    manager = CameraManager()

    assert manager.depth_width == 1280
    assert manager.depth_height == 800
    assert manager.depth_fps == 30
    assert manager.status()["streams"]["depth"]["target_fps"] == 30
