from frigate.infrastructure.config import FrigateConfig


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


def test_runtime_cpu_limit_above_eight_is_accepted() -> None:
    config = FrigateConfig.parse_yaml(
        """
mqtt:
  enabled: false
runtime:
  cpu_limit: 10
cameras: {}
"""
    )

    assert config.runtime.cpu_limit == 10
