namespace MARIA.RBM
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Random
open MARIA.RBM.Data

(**
Continuous Restricted Boltzmann Machine
*)
type CRBM(Nv:int,Nh:int,opsparam:OptParams)=
               
    let d = MatrixGenerator.CreateInitialMatrixSet Nv Nh -0.01 0.01 0.05 0.25
    let niter = opsparam.niter
    let mutable W = d.W
    let mutable dW = d.dW
    let mutable Ah = d.Ah
    let mutable Av = d.Av
  
    let mutable bestW = W.Clone()
    let mutable bestAh = Ah.Clone()
    
    let dAh = Ah*0.0
    let nBatches = opsparam.nBatches

    let sigmoid = (new activation.Sigmoid(-10.0,10.0)) :> activation.IActivation<float>

    //Optimization parameters:
    let sigma = opsparam.activationSigma |> defaultArg <| 0.001
    let mutable epsW = opsparam.lrW
    let mutable epsA = opsparam.lrAh
    let mutable cost = opsparam.l2regularization |> defaultArg <| 0.0
    let mutable momentum = opsparam.momentum |> defaultArg <| 0.0

    let normd = new MathNet.Numerics.Distributions.Normal(0.0,sigma,new System.Random(42))
       
    let updateHidden(sv:Vector<float>)=      
        let e = Vector<float>.Build.Random(Nh+1,normd)
        let auxs = W.Transpose()*sv + e
        let sr = sigmoid.vec_eval(Ah.ToArray()) auxs 
        let n = sr.Count
        sr.[ n - 1] <- 1.0
        sr

    let updateVisible(sh:Vector<float>)=      
        let e = Vector<float>.Build.Random(Nv+1,normd)
        let auxs = W*sh + e
        let sr = sigmoid.vec_eval(Av.ToArray()) auxs 
        let n = sr.Count
        sr.[ n - 1] <- 1.0
        sr
        
    
    member self.learn(data:Matrix<float>)(k:int)=
        
        let mutable err = Array.zeroCreate<float>(niter)
        let mutable E = Array.zeroCreate<float>(niter)
        let mutable noimprovement=0
        let mutable totalnoimprovement=0
        let nsamples = float(data.ColumnCount)
        
        let nSamplesPerBatch = nsamples/float(nBatches)
        let mutable minErrever = 1e9

        let optimizeforbatch nb =
            epsW <- opsparam.lrW
            epsA <- opsparam.lrAh
            cost <- opsparam.l2regularization |> defaultArg <| 0.0
            momentum <- opsparam.momentum |> defaultArg <| 0.0
            err <- Array.zeroCreate<float>(niter)
            [0 .. niter-1]
            |> Seq.iter(fun i ->                         
                        
                        let nbatch = nb*(int)nSamplesPerBatch

                        let mutable wpos = Matrix<float>.Build.Dense(Nv+1,Nh+1)
                        let mutable wneg = Matrix<float>.Build.Dense(Nv+1,Nh+1)
                        let mutable apos = Vector<float>.Build.Dense(Nh+1)
                        let mutable aneg = Vector<float>.Build.Dense(Nh+1)
                        let mutable sh= Vector<float>.Build.Dense(Nh+1)
                        
                        for b in 0 .. int(nSamplesPerBatch)-1 do
                            let n = (nbatch + b) % (int)nsamples
                            let xin = MatrixGenerator.ExtendVector(data.Column(n))(1.0)
                
                            sh <- updateHidden(xin)
                            wpos <- wpos + xin.OuterProduct(sh)
                            apos <- apos + sh.PointwiseMultiply(sh)

                            let mutable sv = Vector<float>.Build.DenseOfArray (Array.zeroCreate(Nv+1))

                            for i in 0 .. k-1 do
                                sv <- updateVisible(sh)
                                sh <- updateHidden(sv)
                
                            let tmp = sv.OuterProduct(sh)
                            wneg <- wneg + tmp
                            aneg <- aneg + sh.PointwiseMultiply(sh)

                            let delta = sv - xin
                            err.[i] <- err.[i] + delta.L2Norm()

                            ()
                    
                        if minErrever>err.[i] then
                            minErrever <- err.[i]

                        dW <- dW * momentum + epsW* (wpos-wneg)/nSamplesPerBatch - cost*W
                        W <- W + dW
                        Ah <- Ah + epsA* (apos-aneg)/(nSamplesPerBatch* sh.PointwiseMultiply(sh))
                
                          
                        if i>0 then
                            let minerr = minErrever
                            printfn " iter=%i, ERROR=%f" i err.[i]
                            if i > 0 && err.[i] < minerr then
                                bestW <- W
                                bestAh <- Ah

                            if err.[i]> (minerr*10.0) then
                                printfn "ACTIVATING RESETTING ------------------------------------"
                                W <- bestW
                                Ah <- bestAh
                                epsW <- opsparam.lrW *0.5
                                epsA <- opsparam.lrAh*0.5
                                cost <- (opsparam.l2regularization |> defaultArg <| 0.0) *0.5
                                momentum <- (opsparam.momentum |> defaultArg <| 0.0) *0.5
                        else
                                bestW <- W
                                bestAh <- Ah


                        )
        [0 .. nBatches-1]
        |> Seq.iter(fun n ->
                            printfn " ------------> BATCH=%i" n
                            optimizeforbatch n
                            )

        W <- bestW
        Ah <- bestAh
     
    member self.reconstruct(data:Matrix<float>)(nPoints:int,nSteps:int)=
        let Z = Matrix<float>.Build.Dense(Nv,nPoints)

        let rng = new MathNet.Numerics.Random.MersenneTwister(42)
        let vo = Array.concat [Array.zeroCreate(Nv);[|1.0|]]
        let mutable sv = Vector<float>.Build.DenseOfArray(vo)
        let mutable sh = Vector<float>.Build.Dense(Nh+1)
        use writer = new System.IO.StreamWriter("E:/Sampling.csv")
        for i in 0 .. nPoints-1 do
            let xr = Vector<float>.Build.Random(Nv)
            let x = Vector<float>.Build.DenseOfArray(Array.concat [xr.ToArray();[|1.0|]])
            
            if i%nSteps = 0 then
               let vo = Array.concat [data.Column(rng.Next(0,data.ColumnCount-1)).ToArray();[|1.0|]]
               sv <- Vector<float>.Build.DenseOfArray(vo)
               writer.Write("Sample->")
               writer.WriteLine(sv.ToRowMatrix().ToString())
            
            sh <- updateHidden(sv)
            sv <- updateVisible(sh)
            writer.WriteLine((i%nSteps).ToString()+"-->"+sv.ToRowMatrix().ToString())

            Z.SetColumn(i,sv.SubVector(0,sv.Count-1))    
        Z
    
