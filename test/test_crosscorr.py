import pytest
from numpy import arange, allclose, asarray, expand_dims
from scipy.ndimage.interpolation import shift

from registration import CrossCorr

pytestmark = pytest.mark.usefixtures("eng")

def test_fit(eng):
    reference = arange(25).reshape(5, 5)
	algorithm = CrossCorr()
	deltas = [[1, 2], [-2, 1]]
	shifted = [shift(reference, delta, mode='wrap', order=0) for delta in deltas]
	model = algorithm.fit(shifted, reference=reference)
	assert allclose(model.toarray(), deltas)


def test_fit_3d(eng):
	reference = arange(125).reshape(5, 5, 5)
	algorithm = CrossCorr()
	deltas = [[1, 0, 2], [0, 1, 2]]
	shifted = [shift(reference, delta, mode='wrap', order=0) for delta in deltas]
	model = algorithm.fit(shifted, reference=reference)
	assert allclose(model.toarray(), deltas)


def test_fit_axis(eng):
	reference = arange(60).reshape(2, 5, 6)
	algorithm = CrossCorr(axis=0)
	a = shift(reference[0], [1, 2], mode='wrap', order=0)
	b = shift(reference[1], [-2, 1], mode='wrap', order=0)
	c = shift(reference[0], [2, 1], mode='wrap', order=0)
	d = shift(reference[1], [1, -2], mode='wrap', order=0)
	shifted = [asarray([a, b]), asarray([c, d]),]
	model = algorithm.fit(shifted, reference=reference)
	assert allclose(model.toarray(), [[[1, 2], [-2, 1]], [[2, 1], [1, -2]]])
