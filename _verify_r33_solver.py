import numpy as np
if not hasattr(np,"asfarray"): np.asfarray = lambda a,dtype=np.float64: np.asarray(a,dtype=dtype)
if not hasattr(np,"float_"): np.float_ = np.float64
import motmetrics as mm
mm.lap.default_solver = 'scipy'
