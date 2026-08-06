from frigate.config import FrigateConfig


def test_runtime_deployment_config_is_accepted() -> None:
    config = FrigateConfig.parse_yaml(
        """
mqtt:
  enabled: false
runtime:
  cpu_limit: 3
  replay:
    loop: false
    sources:
      face_camera: D:/video/face.mp4
  integrations:
    enabled: true
go2rtc:
  streams: {}
cameras: {}
"""
    )

    assert config.runtime.cpu_limit == 3
    assert config.runtime.replay.loop is False
    assert config.runtime.replay.sources["face_camera"] == "D:/video/face.mp4"
    assert config.runtime.integrations.enabled is True


def test_runtime_cpu_limit_is_bounded() -> None:
    try:
        FrigateConfig.parse_yaml(
            """
mqtt:
  enabled: false
runtime:
  cpu_limit: 5
cameras: {}
"""
        )
    except ValueError as error:
        assert "cpu_limit" in str(error)
    else:
        raise AssertionError("runtime.cpu_limit above four must be rejected")
