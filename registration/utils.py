from thunder.images import fromarray, Images
from numpy import asarray, ndarray


def check_images(data):
    """
    Check and reformat input images if needed
    """
    if isinstance(data, ndarray):
        data = fromarray(data)
    
    if not isinstance(data, Images):
        data = fromarray(asarray(data))

    if len(data.shape) not in set([3, 4]):
        raise Exception('Number of image dimensions %s must be 2 or 3' % (len(data.shape)))

    return data


def check_reference(images, reference):
    """
    Ensure the reference matches image dimensions
    """
    if not images.shape[1:] == reference.shape:
        raise Exception('Image shape %s and reference shape %s must match'
                        % (images.shape[1:], reference.shape))
    return reference
