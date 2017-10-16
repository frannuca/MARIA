namespace MARIA.RBM
open MathNet.Numerics.LinearAlgebra

module Data=
    type OptParams={
            ftol:double;
            niter:int;
            xtol:double;
            activationSigma:double option;
            lrW:float;
            lrAh:float;
            momentum:double option; 
            l2regularization:double option;
            nBatches:int
            }
    

