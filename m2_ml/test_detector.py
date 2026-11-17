"""Small synthetic integration checks; these are not bird-dataset results.

Run: python -m unittest m2_ml.test_detector -v
"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from m2_ml.detector import (
    Config, coco_ground_truth, crop_rgb, evaluate_predictions, extract_features,
    fit_model, iou_one_to_many, is_background, iter_windows, learn_templates,
    load_records, make_training_features, mine_hard_negatives, nms,
    predict_records, predictions_from_raw, proposal_recall, save_bundle,
    scan_records,
    scan_image_many, read_rgb,
    hog_descriptor,
)


def make_synthetic_dataset(root):
    """Tiny labelled geometric objects for exercising code paths only."""
    rng = np.random.default_rng(12)
    for split, count in [("train", 8), ("val", 3), ("test", 2)]:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(count):
            pixels = rng.integers(185, 220, (40, 40, 3), dtype=np.uint8)
            image = Image.fromarray(pixels)
            label = ""
            if i % 3 != 2:
                x, y = 4 + i % 3 * 4, 8 + i % 2 * 4
                draw = ImageDraw.Draw(image)
                draw.polygon([(x,y+8),(x+4,y),(x+8,y+8)], fill=(25,35,45))
                label = f"0 {(x+4)/40} {(y+4)/40} 0.2 0.2\n"
            image.save(root / "images" / split / f"{split}_{i}.jpg", quality=100)
            (root / "labels" / split / f"{split}_{i}.txt").write_text(label)


class DetectorChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        make_synthetic_dataset(cls.root)
        cls.train = load_records(cls.root, "train")
        cls.val = load_records(cls.root, "val")
        cls.cfg = Config(patch_size=16, n_templates=2, negatives_per_image=4,
                         max_positives=100, batch_size=32, stride_fraction=0.5,
                         min_stride=2, max_hard_negatives=8)
        cls.templates = learn_templates(cls.train, cls.cfg)
        cls.X, cls.y, cls.audit = make_training_features(
            cls.train, cls.templates, cls.cfg, progress=False)
        cls.bundle = fit_model(cls.X["hog"], cls.y, cls.templates, cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_geometry_and_background_containment(self):
        np.testing.assert_allclose(iou_one_to_many([0,0,10,10],
                                                  [[0,0,10,10],[20,20,2,2]]), [1,0])
        self.assertFalse(is_background([0,0,40,40], np.array([[4,4,2,2]]), self.cfg))
        self.assertTrue(is_background([20,20,4,4], np.array([[4,4,2,2]]), self.cfg))
        np.testing.assert_array_equal(nms([[0,0,10,10],[1,1,10,10],[20,20,4,4]],
                                          [3,2,1]), [0,2])
        self.assertEqual(len(nms([], [])), 0)
        windows = list(iter_windows(41, 39, [(8,8)], self.cfg))
        self.assertIn((33,31,8,8), windows)
        self.assertEqual(list(iter_windows(4,4,[(8,8)],self.cfg)), [])

    def test_feature_pairing_training_and_serialisation(self):
        self.assertEqual(self.X["hog_color"].shape[1], self.X["hog"].shape[1] + 24)
        np.testing.assert_array_equal(self.X["hog"], self.X["hog_color"][:,:-24])
        np.testing.assert_array_equal(np.unique(self.y), [0,1])
        color_bundle = fit_model(self.X["hog_color"], self.y, self.templates,
                                 self.cfg, use_color=True)
        pred, times = predict_records(self.val[:1], color_bundle, progress=False)
        self.assertEqual(len(times), 1)
        self.assertTrue(all(np.isfinite(p["score"]) for p in pred))
        path = self.root / "model.joblib"
        save_bundle(self.bundle, path)
        import joblib
        restored = joblib.load(path)
        np.testing.assert_allclose(restored["pipeline"].decision_function(self.X["hog"]),
                                   self.bundle["pipeline"].decision_function(self.X["hog"]))

    def test_scanning_and_postprocessing_match(self):
        raw, _ = scan_records(self.val, self.bundle, progress=False)
        cached = predictions_from_raw(raw, self.cfg)
        direct, _ = predict_records(self.val, self.bundle, progress=False)
        self.assertEqual(cached, direct)
        metrics = evaluate_predictions(direct, self.val, self.root / "eval", "model")
        self.assertTrue(0 <= metrics["AP50"] <= 1)
        coverage = proposal_recall(self.train, self.templates, self.cfg)
        self.assertGreater(coverage["recall"], 0.5)

    def test_evaluator_perfect_and_negative_image_false_positive(self):
        perfect = [{"image_id": r.path.stem, "bbox": b.tolist(), "score": 1.0}
                   for r in self.val for b in r.boxes]
        metrics = evaluate_predictions(perfect, self.val, self.root / "eval", "perfect")
        self.assertAlmostEqual(metrics["AP50"], 1.0)
        self.assertTrue(np.isnan(metrics["AP50_medium"]))
        false = [{"image_id": self.val[-1].path.stem, "bbox": [0,0,8,8], "score": 2.0}]
        metrics = evaluate_predictions(false + perfect, self.val, self.root / "eval", "false")
        self.assertLess(metrics["AP50"], 1.0)
        self.assertEqual(len(coco_ground_truth(self.val)["images"]), 3)

    def test_mining_rejects_bird_windows_and_supports_retraining(self):
        class AlwaysBird:
            n_features_in_ = self.X["hog"].shape[1]
            def decision_function(self, X):
                return np.ones(len(X))
        mining_bundle = dict(self.bundle, pipeline=AlwaysBird())
        hard, audit = mine_hard_negatives(self.train, mining_bundle, progress=False)
        self.assertGreater(len(hard), 0)
        self.assertLessEqual(len(hard), self.cfg.max_hard_negatives)
        by_name = {r.path.stem: r for r in self.train}
        for entry in audit:
            self.assertTrue(is_background(entry["bbox"], by_name[entry["image_id"]].boxes,
                                           self.cfg))
            self.assertTrue(entry["image_id"].startswith("train_"))
        model = fit_model(np.vstack([self.X["hog"], hard]),
                          np.concatenate([self.y, np.zeros(len(hard), dtype=np.int8)]),
                          self.templates, self.cfg)
        self.assertEqual(model["training_samples"], len(self.y) + len(hard))

    def test_missing_labels_fail_instead_of_silent_negatives(self):
        path = self.root / "labels" / "test" / "test_0.txt"
        contents = path.read_text()
        path.unlink()
        try:
            with self.assertRaises(FileNotFoundError):
                load_records(self.root, "test")
        finally:
            path.write_text(contents)

    def test_shared_scanning_preserves_individual_model_results(self):
        color = fit_model(self.X["hog_color"], self.y, self.templates, self.cfg, True)
        bundles = {"hog": self.bundle, "color": color}
        result = scan_image_many(read_rgb(self.val[0]), bundles, self.cfg.score_floor)
        for name, bundle in bundles.items():
            raw, _ = scan_records(self.val[:1], bundle, progress=False)
            expected = raw[self.val[0].path.stem]
            np.testing.assert_array_equal(result[name][0], expected[0])
            np.testing.assert_allclose(result[name][1], expected[1], rtol=1e-6, atol=1e-6)

    def test_vectorized_hog_matches_public_reference(self):
        rng = np.random.default_rng(7)
        for size, cell, block in [(32,4,2),(32,8,2),(16,4,2),(48,6,3)]:
            cfg = replace(self.cfg, patch_size=size, cell_size=cell, block_cells=block)
            images = [np.zeros((size,size)), np.ones((size,size)),
                      rng.random((size,size)), np.eye(size)]
            for image in images:
                np.testing.assert_allclose(hog_descriptor(image,cfg),
                    hog_descriptor(image,replace(cfg,hog_backend="reference")),atol=1e-12,rtol=1e-12)

    def test_shared_ground_truth_preserves_precision_and_matches_m2(self):
        from evaluation.make_coco_gt import build_coco_gt
        label=self.root/"labels"/"val"/"val_0.txt"
        original=label.read_text()
        label.write_text("0 0.333333 0.444444 0.123456 0.234567\n")
        try:
            records=load_records(self.root,"val")
            with patch("utils.data.get_data_root",return_value=self.root):
                shared=build_coco_gt("val")
            self.assertEqual(shared,coco_ground_truth(records))
            width=shared["annotations"][0]["bbox"][2]
            self.assertNotEqual(width,round(width,2))
        finally:
            label.write_text(original)


if __name__ == "__main__":
    unittest.main()
