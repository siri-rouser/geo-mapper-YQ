from typing import List, Optional

from geojson_pydantic import Polygon
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated
from visionlib.pipeline.settings import LogLevel, YamlConfigSettingsSource


class RedisConfig(BaseModel):
    host: str = 'localhost'
    port: Annotated[int, Field(ge=1, le=65536)] = 6379
    input_stream_prefix: str = 'objectdetector'
    output_stream_prefix: str = 'geomapper'

class CameraConfig(BaseModel):
    stream_id: str
    passthrough: bool
    cam_config_path: str
    mapping_area: Optional[Polygon] = None
    remove_unmapped_detections: bool = False

class GeoMapperConfig(BaseSettings):
    log_level: LogLevel = LogLevel.WARNING
    redis: RedisConfig
    cameras: List[CameraConfig]
    object_center_elevation_m: float = 0

    model_config = SettingsConfigDict(env_nested_delimiter='__')

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls), file_secret_settings)