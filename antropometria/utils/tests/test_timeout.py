import time
from unittest.mock import MagicMock

import pytest
from pytest import approx

from antropometria.utils.timeout import timeout

WAIT_TIME = 5
MOCK = MagicMock()


@timeout(WAIT_TIME, True)
def this_function_will_take_time(time_to_wait):
    time.sleep(time_to_wait)
    MOCK()


@timeout(WAIT_TIME, False)
def this_function_will_be_called_instantly():
    MOCK()


class TestTimeout:
    def test_function_is_not_called_when_time_expire(self):
        MOCK.reset_mock()
        start_time = time.time()
        with pytest.raises(TimeoutError):
            this_function_will_take_time(WAIT_TIME + 1)

        ellapsed_time = time.time() - start_time

        assert not MOCK.called
        assert approx(ellapsed_time, abs=0.11) == WAIT_TIME

    def test_function_is_called_when_time_does_not_expire(self):
        MOCK.reset_mock()
        start_time = time.time()
        this_function_will_take_time(WAIT_TIME - 1)

        ellapsed_time = time.time() - start_time

        assert approx(ellapsed_time, abs=0.11) == WAIT_TIME - 1
        assert MOCK.called

    def test_timeout_is_not_applied_if_condition_is_not_matched(self):
        MOCK.reset_mock()
        start_time = time.time()
        this_function_will_be_called_instantly()

        ellapsed_time = time.time() - start_time

        assert approx(ellapsed_time, abs=0.11) == 0
        assert MOCK.called
