import pytest
from blinker.ear import safe_EAR_calc

@pytest.mark.parametrize("bad_data", [
    [(226, 146), (258, 146), (235, 151), (247, 150), (245, 140)],  # too short
    [(226, 146), (258, 146), (235, 151), (247, 150), (245, 140), (100, 500), (494, 747)],  # too long
    [[258, 220], [299, 216], [272, 221], [287, 219], [285, 220], [270, 222]],  # wrong type (lists not tuples)
    [(226, 146), (258, 146), (235, 151), (226, 146), (245, 140), (100, 500)],  # div by zero
])
def test_safe_ear_calc_returns_none_on_bad_input(bad_data):
    assert safe_EAR_calc(bad_data) is None
