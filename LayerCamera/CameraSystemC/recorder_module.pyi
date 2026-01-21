import numpy
import numpy.typing
import typing

class Frame:
    is_eos: bool
    def __init__(self) -> None:
        """__init__(self: recorder_module.Frame) -> None"""
    @property
    def height(self) -> int:
        """(arg0: recorder_module.Frame) -> int"""
    @property
    def image(self) -> numpy.typing.NDArray[numpy.uint8]:
        """numpy.ndarray of BGR image
        (arg0: recorder_module.Frame) -> numpy.typing.NDArray[numpy.uint8]
        """
    @property
    def index(self) -> int:
        """(arg0: recorder_module.Frame) -> int"""
    @property
    def monotonic_timestamp(self) -> float:
        """(arg0: recorder_module.Frame) -> float"""
    @property
    def timestamp(self) -> float:
        """(arg0: recorder_module.Frame) -> float"""
    @property
    def width(self) -> int:
        """(arg0: recorder_module.Frame) -> int"""

class ImageBuffer:
    def __init__(self) -> None:
        """__init__(self: recorder_module.ImageBuffer) -> None"""
    def clear(self) -> None:
        """clear(self: recorder_module.ImageBuffer) -> None"""
    def pop(self, blocking: bool = ...) -> Frame:
        """pop(self: recorder_module.ImageBuffer, blocking: bool = True) -> recorder_module.Frame"""
    def push(self, arg0: Frame) -> None:
        """push(self: recorder_module.ImageBuffer, arg0: recorder_module.Frame) -> None"""

class MetricData:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def serialize(self) -> dict:
        """serialize(self: recorder_module.MetricData) -> dict"""
    @property
    def avg_fps(self) -> float:
        """(arg0: recorder_module.MetricData) -> float"""
    @property
    def fps(self) -> float:
        """(arg0: recorder_module.MetricData) -> float"""
    @property
    def frames_dropped(self) -> int:
        """(arg0: recorder_module.MetricData) -> int"""
    @property
    def frames_rendered(self) -> int:
        """(arg0: recorder_module.MetricData) -> int"""
    @property
    def kf(self) -> list[float]:
        """(arg0: recorder_module.MetricData) -> list[float]"""

class Recorder:
    def __init__(self, arg0: str) -> None:
        """__init__(self: recorder_module.Recorder, arg0: str) -> None"""
    def disableDisplay(self) -> None:
        """disableDisplay(self: recorder_module.Recorder) -> None"""
    def disableUdp(self) -> None:
        """disableUdp(self: recorder_module.Recorder) -> None"""
    def enableDisplay(self, win_id: typing.SupportsInt) -> None:
        """enableDisplay(self: recorder_module.Recorder, win_id: typing.SupportsInt) -> None"""
    def enableUdp(self, host: str, port: typing.SupportsInt) -> None:
        """enableUdp(self: recorder_module.Recorder, host: str, port: typing.SupportsInt) -> None"""
    def getCaptureFormats(self) -> list[dict[str, int | str | list[int]]]:
        """getCaptureFormats(self: recorder_module.Recorder) -> list[dict[str, int | str | list[int]]]"""
    def getDeviceInfo(self) -> dict[str, str]:
        """getDeviceInfo(self: recorder_module.Recorder) -> dict[str, str]"""
    def getImageBuffer(self) -> ImageBuffer:
        """getImageBuffer(self: recorder_module.Recorder) -> recorder_module.ImageBuffer"""
    def getMetricData(self) -> MetricData:
        """getMetricData(self: recorder_module.Recorder) -> recorder_module.MetricData"""
    def init(self, cam_serial: str, win_id: typing.SupportsInt, udp_host: str, udp_port: typing.SupportsInt, direction: typing.SupportsInt, clockoverlay: bool = ..., software_trigger: bool = ..., resync: bool = ...) -> None:
        """init(self: recorder_module.Recorder, cam_serial: str, win_id: typing.SupportsInt, udp_host: str, udp_port: typing.SupportsInt, direction: typing.SupportsInt, clockoverlay: bool = False, software_trigger: bool = False, resync: bool = False) -> None"""
    def listAvailableCamera(self) -> list[dict[str, str]]:
        """listAvailableCamera(self: recorder_module.Recorder) -> list[dict[str, str]]"""
    def release(self) -> None:
        """release(self: recorder_module.Recorder) -> None"""
    def resync(self, resync_timestamp_ns: typing.SupportsInt, resync_dt_ns: typing.SupportsInt) -> None:
        """resync(self: recorder_module.Recorder, resync_timestamp_ns: typing.SupportsInt, resync_dt_ns: typing.SupportsInt) -> None

        Synchronize camera with reference timestamp.

        Args:
          resync_timestamp_ns (int): Timestamp in nanoseconds. Must not be in the future.
        """
    def setCaptureFormat(self, width: typing.SupportsInt, height: typing.SupportsInt, target_fps: typing.SupportsInt, skipping: str = ...) -> None:
        """setCaptureFormat(self: recorder_module.Recorder, width: typing.SupportsInt, height: typing.SupportsInt, target_fps: typing.SupportsInt, skipping: str = '') -> None"""
    def setProperty(self, name: str, value: str) -> None:
        """setProperty(self: recorder_module.Recorder, name: str, value: str) -> None"""
    def start(self, trigger_start: typing.SupportsInt = ...) -> None:
        """start(self: recorder_module.Recorder, trigger_start: typing.SupportsInt = 0) -> None"""
    def startRecording(self, save_path: str, imgbuf: bool, imgbuf_width: typing.SupportsInt, imgbuf_height: typing.SupportsInt, mode: str) -> None:
        """startRecording(self: recorder_module.Recorder, save_path: str, imgbuf: bool, imgbuf_width: typing.SupportsInt, imgbuf_height: typing.SupportsInt, mode: str) -> None"""
    def startVideoFeeder(self, video_path: str, enable_imgbuf: bool) -> float:
        """startVideoFeeder(self: recorder_module.Recorder, video_path: str, enable_imgbuf: bool) -> float"""
    def stopRecording(self) -> None:
        """stopRecording(self: recorder_module.Recorder) -> None"""
    def stopVideoFeeder(self) -> None:
        """stopVideoFeeder(self: recorder_module.Recorder) -> None"""
    def takeSnapshot(self) -> Snapshot:
        """takeSnapshot(self: recorder_module.Recorder) -> recorder_module.Snapshot

        take single snapshot
        """
    @property
    def isStreaming(self) -> bool:
        """Is camera streaming.
        (arg0: recorder_module.Recorder) -> bool
        """

class Snapshot:
    def __init__(self) -> None:
        """__init__(self: recorder_module.Snapshot) -> None"""
    @property
    def height(self) -> int:
        """(arg0: recorder_module.Snapshot) -> int"""
    @property
    def image(self) -> numpy.typing.NDArray[numpy.uint8]:
        """numpy.ndarray of BGR image
        (arg0: recorder_module.Snapshot) -> numpy.typing.NDArray[numpy.uint8]
        """
    @property
    def width(self) -> int:
        """(arg0: recorder_module.Snapshot) -> int"""
