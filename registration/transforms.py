from numpy import asarray, rollaxis

class Transformation(object):
    """ 
    Base class for transformations 
    """
    def apply(self, im):
        raise NotImplementedError

    def toarray(self):
        raise NotImplementedError        


class Displacement(Transformation):
    """
    Class for transformations based on spatial displacements.

    Can be applied to either images or volumes.

    Attributes
    ----------
    delta : list
        A list of spatial displacements for each dimension,
        e.g. [10, 5, 2] for a displacement of 10 in x, 5 in y, 2 in z
    """

    def __init__(self, delta=None):
        self.delta = delta

    def toarray(self):
        """
        Return transformation as an array
        """
        return asarray(self.delta)

    def apply(self, im):
        """
        Apply an n-dimensional displacement by shifting an image or volume.

        Parameters
        ----------
        im : ndarray
            The image or volume to shift
        """
        from scipy.ndimage.interpolation import shift
        return shift(im, map(lambda x: -x, self.delta), mode='nearest')

    @staticmethod
    def compute(a, b):
        """
        Compute an optimal displacement between two ndarrays.

        Finds the displacement between two ndimensional arrays. Arrays must be
        of the same size. Algorithm uses a cross correlation, computed efficiently
        through an n-dimensional fft.

        Parameters
        ----------
        a : ndarray
            The first array

        b : ndarray
            The second array
        """
        from numpy.fft import rfftn, irfftn
        from numpy import unravel_index, argmax

        # compute real-valued cross-correlation in fourier domain
        s = a.shape
        f = rfftn(a)
        f *= rfftn(b).conjugate()
        c = abs(irfftn(f, s))

        # find location of maximum
        inds = unravel_index(argmax(c), s)

        # fix displacements that are greater than half the total size
        pairs = zip(inds, a.shape)
        # cast to basic python int for serialization
        adjusted = [int(d - n) if d > n // 2 else int(d) for (d, n) in pairs]

        return Displacement(adjusted)

    def __repr__(self):
        return "Displacement(delta=%s)" % repr(self.delta)

class LocalDisplacement(Transformation):
    """
    Class for transformations based on axis-localized spatial displacements.

    Applied separately along an axis.

    Attributes
    ----------
    delta : list
        A nested list, where the first list is over planes, and
        for each plane a list of [x,y] displacements

    axis : int
        Which axis to localize displacements to
    """

    def __init__(self, delta=None, axis=None):
        self.delta = delta
        self.axis = axis

    def toarray(self):
        """
        Return transformation as an array
        """
        return asarray(self.delta)

    @staticmethod
    def compute(a, b, axis):
        """
        Finds optimal displacements localized along an axis
        """
        delta = []
        for aa, bb in zip(rollaxis(a, axis, 0), rollaxis(b, axis, 0)):
            delta.append(Displacement.compute(aa, bb).delta)
        return LocalDisplacement(delta, axis=axis)

    def apply(self, im):
        """
        Apply axis-localized displacements.

        Parameters
        ----------
        im : ndarray
            The image or volume to shift
        """
        from scipy.ndimage.interpolation import shift

        im = rollaxis(im, self.axis)
        im.setflags(write=True)
        for ind in range(0, im.shape[0]):
            im[ind] = shift(im[ind],  map(lambda x: -x, self.delta[ind]), mode='nearest')
        im = rollaxis(im, 0, self.axis+1)
        return im

    def __repr__(self):
        return "LocalDisplacement(delta=%s)" % repr(self.delta)


class SimpleITKTransformation(Transformation):
    """

    Attributes
    ----------
    transform_estimator

    transform

    """
    def __init__(self, transform_estimator=None, transform=None):
        self.transform_estimator = transform_estimator
        self.transform = transform

    def toarray(self):
        pass

    def apply(self, im):
        from SimpleITK import GetImageFromArray, GetArrayFromImage, Resample, sitkBSpline
        im_ = GetImageFromArray(im)
        #todo: add default pixel value kwarg
        im_txed = Resample(im_, self.transform, sitkBSpline)
        return GetArrayFromImage(im_txed)

    @staticmethod
    def compute(a, b, transform_estimator):
        """
        Estimate SimpleITK transform

        Parameters
        ----------
        a : ndarray
            The first array, treated as the moving image

        b : ndarray
            The second array, treated as the reference image

        transform_estimator : SimpleITK.Transform object
            The transformation to be estimated using a and b
        """
        from SimpleITK import GetImageFromArray
        moving, fixed = GetImageFromArray(a), GetImageFromArray(b)
        tx = transform_estimator.Execute(fixed, moving)
        return SimpleITKTransformation(transform_estimator=transform_estimator, transform=tx)

    def __repr__(self):
        print(self.transform.GetParameters())
