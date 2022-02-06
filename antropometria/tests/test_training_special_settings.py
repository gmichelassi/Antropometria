from context import set_tests_context
set_tests_context()

from antropometria.utils.parameter_calibration.special_settings import skip_current_test


class TestTrainingSpecialSettings:
    def test_skips_current_test(self):
        assert not skip_current_test(classifier='RandomForest', reduction=None)
        assert not skip_current_test(classifier='RandomForest', reduction='RFSelect')
        assert skip_current_test(classifier='RandomForest', reduction='PCA')
        assert skip_current_test(classifier='SVM', reduction=None)
        assert not skip_current_test(classifier='SVM', reduction='PCA')
