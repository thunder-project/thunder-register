import pytest
from numpy import linspace, exp, allclose, meshgrid
from scipy.ndimage.interpolation import shift
from SimpleITK import ImageRegistrationMethod, TranslationTransform
from registration import SimpleITK_Registration

pytestmark = pytest.mark.usefixtures("eng")

def test_fit(eng):
    x, y = meshgrid(linspace(-4,4,100), linspace(-4,4,100))
    reference = exp(-x**2 + -y**2)
    r = ImageRegistrationMethod()
    r.SetMetricAsCorrelation()
    r.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1,
                                               minStep=1e-5,
                                               numberOfIterations=10000,
                                               gradientMagnitudeTolerance=1e-8)
    r.SetInitialTransform(TranslationTransform(len(reference.shape)), inPlace=False)
    algorithm = SimpleITK_Registration(r)
    deltas = [[1.5, -10], [-1.5, 10]]
    shifted = [shift(reference, delta) for delta in deltas]
    model = algorithm.fit(shifted, reference=reference)
    # flip the dimensions of model.toarray() before comparing to deltas because SimpleITK uses xy ordering.
    model_deltas = map(lambda v: v[::-1], model.toarray())
    assert allclose(model_deltas, deltas)