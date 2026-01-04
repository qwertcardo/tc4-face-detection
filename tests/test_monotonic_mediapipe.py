from unittest.mock import Mock

from src.monotonic_mediapipe import MonotonicVideoDetector


def test_increments_within_same_frame():
    detector = Mock()
    wrapper = MonotonicVideoDetector(detector=detector)

    wrapper.detect_for_video(Mock(), timestamp_ms=1_000, face_id=0)
    wrapper.detect_for_video(Mock(), timestamp_ms=1_000, face_id=1)

    first_ts = detector.detect_for_video.call_args_list[0].args[1]
    second_ts = detector.detect_for_video.call_args_list[1].args[1]
    assert first_ts == 0
    assert second_ts == 1
    assert second_ts > first_ts


def test_recovers_from_non_future_timestamp():
    detector = Mock()
    wrapper = MonotonicVideoDetector(detector=detector)

    wrapper.detect_for_video(Mock(), timestamp_ms=2_000, face_id=5)  # forwards 0
    wrapper.detect_for_video(Mock(), timestamp_ms=2_000, face_id=0)  # forwards 1

    last_call_ts = detector.detect_for_video.call_args_list[-1].args[1]
    assert last_call_ts == 1
    assert last_call_ts > detector.detect_for_video.call_args_list[-2].args[1]


def test_monotonic_over_many_calls():
    detector = Mock()
    wrapper = MonotonicVideoDetector(detector=detector)

    for i in range(5):
        wrapper.detect_for_video(Mock(), timestamp_ms=10_000, face_id=i)

    forwarded = [call.args[1] for call in detector.detect_for_video.call_args_list]
    assert forwarded == list(range(5))


def test_reset_allows_restarting_counter():
    detector = Mock()
    wrapper = MonotonicVideoDetector(detector=detector, start_at=10)

    wrapper.detect_for_video(Mock(), timestamp_ms=0, face_id=0)  # forwards 10
    wrapper.detect_for_video(Mock(), timestamp_ms=0, face_id=1)  # forwards 11

    wrapper.reset()

    wrapper.detect_for_video(Mock(), timestamp_ms=0, face_id=0)  # forwards 0
    wrapper.detect_for_video(Mock(), timestamp_ms=0, face_id=1)  # forwards 1

    forwarded = [call.args[1] for call in detector.detect_for_video.call_args_list]
    assert forwarded == [10, 11, 0, 1]
