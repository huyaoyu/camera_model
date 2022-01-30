
from typing import List, Tuple, Iterable

import numpy as np

from .camera_model_base import CameraModel

class DoubleSphere(CameraModel):
    def __init__(self, 
        fx: float, 
        fy: float, 
        cx: float, 
        cy: float, 
        shape: Iterable[int],
        xi: float,
        alpha: float) -> None:
        super().__init__(fx, fy, cx, cy, shape)

        self.xi = xi
        self.alpha = alpha

        if ( self.alpha <= 0.5 ):
            self.w1 = self.alpha / ( 1 - self.alpha )
        else:
            self.w1 = ( 1 - self.alpha ) / self.alpha

        self.w2 = ( self.w1 + self.xi ) / \
            np.sqrt( 2 * self.w1 * self.xi + self.xi**2 + 1 )

    # Override parrent.
    def project_3d_2_image_plane(self, points: np.ndarray) -> np.ndarray:
        '''
        points: 3 x N array, 3D points.

        return:
        2 x N array, projected locations in the image plane.
        '''

        x = points[0, :]
        y = points[1, :]
        z = points[2, :]

        x2 = x**2
        y2 = y**2
        z2 = z**2

        d1 = np.sqrt( x2 + y2 + z2 )
        d2 = np.sqrt( x2 + y2 + ( self.xi * d1 + z )**2 )

        t = self.alpha * d2 + ( 1 - self.alpha ) * ( self.xi * d1 + z )

        ux = ( self.fx / t * x + self.cx ) / ( self.shape[1] - 1 ) * 2 - 1
        uy = ( self.fy / t * y + self.cy ) / ( self.shape[0] - 1 ) * 2 - 1

        mask = z > -self.w2 * d1

        return np.stack( (ux, uy), axis=0 ), mask
