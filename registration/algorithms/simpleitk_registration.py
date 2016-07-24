from numpy import ndarray, asarray

from ..utils import check_images, check_reference
from ..model import RegistrationModel
from ..transforms import SimpleITKTransformation


class SimpleITK_Registration(object):
    """
    Registration using a SimpleITK ImageRegistrationMethod object

    Attributes
    __________

    transform : SimpleITK ImageRegistrationMethod

    """

    def __init__(self, transform_estimator=None):
        from SimpleITK import ImageRegistrationMethod
        if not isinstance(transform_estimator, ImageRegistrationMethod):
            raise ValueError('transform_estimator must be an instance of SimpleITK.ImageRegistrationMethod')
        self.transform_estimator = transform_estimator

    def _get(self, image, reference):
        return SimpleITKTransformation.compute(image, reference, self.transform_estimator)

    def fit(self, images, reference=None):
        """
        Estimate registration model using SimpleITK

        Use the user-supplied SimpleITK.ImageRegistrationMethod to estimate a transformation between each image
        or volume and a reference.

        Parameters
        ----------
        images : array-like or thunder images
            The sequence of images / volumes to register.

        reference : array-like
            A reference image / volume to align to.
        """
        images = check_images(images)
        reference = check_reference(images, reference)

        def func(item):
            key, image = item
            return asarray([key, self._get(image, reference)])

        transformations = images.map(func, with_keys=True).toarray()
        if images.shape[0] == 1:
            transformations = [transformations]

        algorithm = self.__class__.__name__
        return RegistrationModel(dict(transformations), algorithm=algorithm)