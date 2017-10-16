namespace MARIA.RBM
open MathNet.Numerics.LinearAlgebra

type MatrixSet={W:Matrix<float>;dW:Matrix<float>;Av:Vector<float>;Ah:Vector<float>}

module MatrixGenerator=
    let CreateInitialMatrixSet (Nv:int) (Nh:int) (Wmin:float) (Wmax:float) (eAv:float)(eAh:float)=        
        let mutable W = Matrix<float>.Build.Random(Nv+1,Nh+1,new MathNet.Numerics.Distributions.ContinuousUniform(Wmin,Wmax))
        let mutable dW = Matrix<float>.Build.Random(Nv+1,Nh+1,new MathNet.Numerics.Distributions.ContinuousUniform(Wmin/100.0,Wmax/100.0))  
        let mutable Av = Vector<float>.Build.DenseOfArray  [| for i in 0 .. Nv do yield eAv|] //Exponential factor for sigmoids in visible layer
        let mutable Ah = Vector<float>.Build.DenseOfArray  [| for i in 0 .. Nh do yield eAh|] //Exponential factor for sigmoids in hidden layer    
        {MatrixSet.W=W;MatrixSet.Ah=Ah;MatrixSet.Av=Av;MatrixSet.dW=dW}
    
    let ExtendVector (v:Vector<float>) (x:double) =
        let s = Vector<float>.Build.Dense(v.Count+1,x)
        s.SetSubVector(0,v.Count,v)
        s
