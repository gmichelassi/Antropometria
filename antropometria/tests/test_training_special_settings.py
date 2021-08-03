from context import set_tests_context
set_tests_context()

from antropometria.utils.training.special_settings import stop_running_rf, skip_current_test
from antropometria.classifiers.RandomForests import RandomForests as Rf
from antropometria.classifiers.NaiveBayes import NaiveBayes as OtherClassifier


RANDOM_FOREST = Rf
OTHER_CLASSIFIER = OtherClassifier


class TestTrainingSpecialSettings:
    def test_rf_stops_running(self):
        assert not stop_running_rf(is_random_forest_done=False)
        assert not stop_running_rf(is_random_forest_done=False, classifier_name='RandomForests', reduction=None)
        assert stop_running_rf(is_random_forest_done=False, classifier_name='RandomForests', reduction='RFSelect')
        assert stop_running_rf(is_random_forest_done=True)

    def test_skips_current_test(self):
        assert not skip_current_test(is_random_forest_done=False, classifier='RandomForests')
        assert skip_current_test(is_random_forest_done=True, classifier='RandomForests')
        assert skip_current_test(reduction=None)
