"""Loader-routing checks without downloading or executing a neural checkpoint."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from m3_dl.predict import load_detector


class RoutingChecks(unittest.TestCase):
    def test_rtdetr_uses_its_predictor_for_named_checkpoint(self):
        sentinel=object()
        with patch.dict("sys.modules",{"ultralytics":SimpleNamespace(
            YOLO=lambda w: self.fail("Wrong predictor"),RTDETR=lambda w:sentinel)}):
            self.assertIs(load_detector("rtdetr-l.pt"),sentinel)

    def test_rtdetr_detected_inside_generic_best_checkpoint(self):
        underlying=type("RTDETRDetectionModel",(),{})()
        sentinel=object()
        with patch.dict("sys.modules",{"ultralytics":SimpleNamespace(
            YOLO=lambda w:SimpleNamespace(model=underlying),RTDETR=lambda w:sentinel)}):
            self.assertIs(load_detector("best.pt"),sentinel)

    def test_yolo_preserves_its_loader(self):
        sentinel=SimpleNamespace(model=type("DetectionModel",(),{})())
        with patch.dict("sys.modules",{"ultralytics":SimpleNamespace(
            YOLO=lambda w:sentinel,RTDETR=lambda w:self.fail("Wrong predictor"))}):
            self.assertIs(load_detector("best.pt"),sentinel)


if __name__=="__main__":unittest.main()
