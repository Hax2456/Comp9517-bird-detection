"""Checks specific to the revised feature-map detector."""
from dataclasses import replace
import unittest
import numpy as np
from skimage.color import rgb2gray,rgb2hsv
from skimage.feature import hog
from m2_ml.dense_detector import (
    DenseConfig,prepare_level,window_features,window_boxes,sample_image,
    fit_dense,scan_dense,background_mask,
    box_targets,refine_boxes,
)


class DenseChecks(unittest.TestCase):
    def setUp(self):
        self.cfg=DenseConfig(templates=2,negatives_per_image=8,candidate_cap=30,batch_size=32)
        self.image=np.random.default_rng(1).integers(0,256,(48,48,3),dtype=np.uint8)

    def test_dense_hog_matches_public_full_image_map(self):
        level=prepare_level(self.image,(16,16),self.cfg)
        reference=hog(rgb2gray(level["padded"]),orientations=9,pixels_per_cell=(6,6),
                      cells_per_block=(2,2),block_norm="L2-Hys",feature_vector=False)
        features=window_features(level,[1],[2],self.cfg,False)[0]
        np.testing.assert_allclose(features,reference[1:8,2:9].ravel(),rtol=1e-6,atol=1e-7)

    def test_integral_color_matches_direct_histogram(self):
        level=prepare_level(self.image,(16,16),self.cfg)
        features=window_features(level,[1],[2],self.cfg,True)[0][-24:]
        crop=level["padded"][6:54,12:60]
        hsv=rgb2hsv(crop.astype(np.float32)/255)
        expected=np.concatenate([np.histogram(hsv[...,k],bins=8,range=(0,1))[0]/(48*48) for k in range(3)])
        np.testing.assert_allclose(features,expected,atol=1e-7)
        np.testing.assert_allclose(features.reshape(3,8).sum(1),1)

    def test_geometry_and_training_alignment(self):
        level=prepare_level(self.image,(16,16),self.cfg)
        np.testing.assert_allclose(window_boxes(level,[0],[0],self.cfg)[0],[0,0,16,16])
        gt=np.array([[16,16,16,16]],dtype=float)
        X,y,audit=sample_image(self.image,gt,[(16,16),(24,24)],self.cfg,42)
        self.assertEqual(audit["birds_with_positive_windows"],1)
        self.assertEqual(X.shape[1],1788)
        self.assertGreater(int(y.sum()),1)
        self.assertFalse(background_mask(gt,gt,self.cfg)[0])

    def test_linear_export_and_bounded_scanning(self):
        X,y,_=sample_image(self.image,np.array([[16,16,16,16]]),[(16,16)],self.cfg,42)
        bundle=fit_dense(X,y,[(16,16)],self.cfg,.1,True)
        np.testing.assert_allclose(X@bundle["weights"]+bundle["bias"],
                                   bundle["svm"].decision_function(X),rtol=1e-4,atol=1e-5)
        raw=scan_dense(self.image,{"color":bundle},collect_features=True)["color"]
        self.assertLessEqual(len(raw["scores"]),self.cfg.candidate_cap)
        self.assertEqual(raw["features"].shape[1],1788)
        self.assertTrue(np.all(np.diff(raw["scores"])<=0))

    def test_box_regression_roundtrip_and_training_targets(self):
        boxes=np.array([[5,6,20,18],[10,12,14,16]],dtype=np.float32)
        target=np.array([[6,7,21,19],[9,11,16,16]],dtype=np.float32)
        np.testing.assert_allclose(refine_boxes(boxes,box_targets(boxes,target),48,48),target,atol=1e-5)
        X,y,audit,targets=sample_image(self.image,np.array([[16,16,16,16]]),[(16,16)],self.cfg,42,True)
        self.assertTrue(np.isfinite(targets[y==1]).all())
        self.assertTrue(np.isnan(targets[y==0]).all())

    def test_reranking_preserves_descriptor_order_and_capacity(self):
        class FirstFeature:
            def decision_function(self,X):return X[:,0]-.3
        X,y,_=sample_image(self.image,np.array([[16,16,16,16]]),[(16,16)],self.cfg,42)
        base=fit_dense(X,y,[(16,16)],self.cfg,.1,False)
        model=dict(base,reranker=FirstFeature())
        raw=scan_dense(self.image,{"kernel":model},collect_features=True,
                       collect_floor=-np.inf)["kernel"]
        self.assertEqual(raw["features"].shape[1],1788)
        self.assertLessEqual(len(raw["scores"]),self.cfg.candidate_cap)
        np.testing.assert_allclose(raw["scores"],raw["features"][:,0]-.3)
        self.assertTrue(np.all(np.diff(raw["scores"])<=0))


if __name__=="__main__":unittest.main()
