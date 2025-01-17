import logging
from typing import Any, Dict, List, NamedTuple
from numba import njit
import cameratransform as ct
from prometheus_client import Counter, Histogram, Summary
from shapely import Point as ShapelyPoint
from shapely import Polygon
from shapely.geometry import shape
from visionapi_yq.messages_pb2 import BoundingBox, Detection, SaeMessage, Tracklet, TrackletsByCamera, Trajectory

from .config import CameraConfig, GeoMapperConfig

logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s')
logger = logging.getLogger(__name__)

GET_DURATION = Histogram('geo_mapper_get_duration', 'The time it takes to deserialize the proto until returning the tranformed result as a serialized proto',
                         buckets=(0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25))
TRANSFORM_DURATION = Summary('geo_mapper_transform_duration', 'How long the coordinate transformation takes')
OBJECT_COUNTER = Counter('geo_mapper_object_counter', 'How many detections have been transformed')
PROTO_SERIALIZATION_DURATION = Summary('geo_mapper_proto_serialization_duration', 'The time it takes to create a serialized output proto')
PROTO_DESERIALIZATION_DURATION = Summary('geo_mapper_proto_deserialization_duration', 'The time it takes to deserialize an input proto')

class Point(NamedTuple):
    x: float
    y: float


class GeoMapper:
    def __init__(self, config: GeoMapperConfig) -> None:
        self._config = config
        logger.setLevel(self._config.log_level.value)

        self._cameras: Dict[str, ct.Camera] = dict()
        self._mapping_areas: Dict[str, Polygon] = dict()

        self._setup()

    def _setup(self):
        for cam_conf in self._config.cameras:
            if cam_conf.cam_config_path:
                self._cameras[cam_conf.stream_id] = ct.Camera.load(cam_conf.cam_config_path)

            if cam_conf.mapping_area is not None:
                self._mapping_areas[cam_conf.stream_id] = shape(cam_conf.mapping_area)

    def __call__(self, input_proto) -> Any:
        return self.get(input_proto)
    
    @GET_DURATION.time()
    def get(self, input_proto):
        sae_msg = self._unpack_proto(input_proto)
        cam_id = sae_msg.frame.source_id # NOTE: check if cam_id is stream1/starem2, should be right

        camera = self._cameras.get(cam_id)
        image_height_px = camera.parameters.parameters['image_height_px'].value # if this value is not contained, modify the autofitting part in Projectreclinar
        image_width_px = camera.parameters.parameters['image_width_px'].value

        if camera is None:
            return input_proto

        # retained_detections: List[Detection] = []

        with TRANSFORM_DURATION.time():
            for idx, track_id in enumerate(sae_msg.trajectory.cameras[cam_id].tracklets):
                detection = sae_msg.trajectory.cameras[cam_id].tracklets[track_id].detections_info[-1] # NOTE: Check if the detection is scaled format or normal format

                center = self._get_center(detection.bounding_box,image_height_px,image_width_px)
                gps = camera.gpsFromImage([center.x, center.y], Z=self._config.object_center_elevation_m)
                lat, lon = gps[0], gps[1]
                if self._is_filtered(cam_id, lat, lon):
                    logger.debug(f'SKIPPED: cls {detection.class_id}, oid {detection.object_id.hex()}, lat {lat}, lon {lon}')
                    continue
                detection.geo_coordinate.latitude = lat
                detection.geo_coordinate.longitude = lon

                # retained_detections.append(detection)
                # logger.debug(f'cls {detection.class_id}, oid {detection.object_id.hex()}, lat {lat}, lon {lon}')
        
                # if self._cam_configs[cam_id].remove_unmapped_detections:
                #     sae_msg.ClearField('detections')
                #     sae_msg.detections.extend(retained_detections)

        return self._pack_proto(sae_msg)
        
    @njit
    def _get_center(self, bbox: BoundingBox,image_width_px,image_height_px) -> Point:
        return Point(
            x=(bbox.min_x + bbox.max_x) * image_width_px / 2,
            y=(bbox.min_y + bbox.max_y) * image_height_px/ 2
        )
    
    def _is_filtered(self, cam_id: str, lat: float, lon: float):
        if cam_id in self._mapping_areas:
            point = ShapelyPoint(lon, lat)
            return not self._mapping_areas[cam_id].contains(point)

    @PROTO_DESERIALIZATION_DURATION.time()
    def _unpack_proto(self, sae_message_bytes):
        sae_msg = SaeMessage()
        sae_msg.ParseFromString(sae_message_bytes)

        return sae_msg
    
    @PROTO_SERIALIZATION_DURATION.time()
    def _pack_proto(self, sae_msg: SaeMessage):
        return sae_msg.SerializeToString()
    
