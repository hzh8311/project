import visdom
import numpy as np
vis = visdom.Visdom()
vis.text('hello')
vis.image(np.ones((3,10,10)))
