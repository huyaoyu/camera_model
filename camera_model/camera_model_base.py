
from typing import List, Tuple, Iterable

import cv2
import numpy as np

class CameraModel(object):
    def __init__(self, 
        fx: float, 
        fy: float, 
        cx: float,
        cy: float,
        shape: Iterable[int] # ( H, W )
        ) -> None:
        super().__init__()

        self.fx    = fx
        self.fy    = fy
        self.cx    = cx
        self.cy    = cy
        self.shape = shape

        self.interpolation_type_dict = {
            'linear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST
        }

    def project_3d_2_image_plane(self, points: np.ndarray) -> np.ndarray:
        '''
        points: 3 x N array, 3D points.

        return:
        2 x N array, projected locations in the image plane, (-1, 1).
        N-length array, valid mask.
        '''

        raise NotImplementedError()

    def sample_points(self, 
        img: np.ndarray, 
        points: np.ndarray, 
        interpolation: str = 'linear') -> np.ndarray:
        '''
        img: OpenCV image.
        points: 3 X N array, 3D points.
        interpolation: One of the strings ( 'linear', 'nearest' )

        return:
        N-length array.
        N-length array, valid mask.
        '''

        assert ( points.shape[1] < 32767 ), \
            f'Too many points. points.shape = {points.shape}. '

        projected, mask = self.project_3d_2_image_plane( points )

        x = ( projected[0, :] * ( img.shape[1] - 1 ) ).reshape( (1, -1) )
        y = ( projected[1, :] * ( img.shape[0] - 1 ) ).reshape( (1, -1) )

        x = x.astype(np.float32)
        y = y.astype(np.float32)

        sampled = cv2.remap( 
            img, 
            x, y, 
            interpolation=self.interpolation_type_dict[ interpolation ] )
        
        sampled = sampled.reshape((-1))

        return sampled, mask

    def sample_image(self, 
        img: np.ndarray, 
        points: np.ndarray, 
        interpolation: str = 'linear') -> np.ndarray:
        '''
        img: OpenCV image.
        points: 3 X H x W array, 3D points.
        interpolation: One of the strings ( 'linear', 'nearest' )

        return:
        Image ( H, W, 3 ).
        Mask (H, W), valid mask.
        '''

        # Reshape.
        H, W = points.shape[1:3]
        points = points.reshape((3, -1))

        projected, mask = self.project_3d_2_image_plane( points )

        x = ( ( projected[0, :] + 1 ) / 2 * ( img.shape[1] - 1 ) ).reshape( (H, W) )
        y = ( ( projected[1, :] + 1 ) / 2 * ( img.shape[0] - 1 ) ).reshape( (H, W) )

        x = x.astype(np.float32)
        y = y.astype(np.float32)

        sampled = cv2.remap( 
            img, 
            x, y, 
            interpolation=self.interpolation_type_dict[ interpolation ] )
        
        sampled = sampled.reshape((-1))

        return sampled, mask