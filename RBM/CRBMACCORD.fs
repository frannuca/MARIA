namespace MARIA.RBM
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Random
open Accord.Neuro
open Accord.Neuro.ActivationFunctions
open Accord.Neuro.Networks
open Accord.Neuro.Learning

type CRBMACCORD(Nv:int,Nh:int,inputs:float[][])=

    let activation = new ActivationFunctions.GaussianFunction(0.1,new Accord.DoubleRange(-1.0,1.0))
    let rbm = new RestrictedBoltzmannMachine(activation,  Nv, Nh)

    // Create the learning algorithm for RBMs
    let teacher =   let o = new Learning.ContrastiveDivergenceLearning(rbm)
                    o.Momentum <- 0.9
                    o.LearningRate <- 0.1
                    o.Decay <- 0.01
                    o
                   
    
    member self.Run(niter:int)=
    // learn 5000 iterations
        for i  in 0 .. niter-1 do
            teacher.RunEpoch(inputs) |> ignore
            
    
    member self.sample(n:int,k:int)=
        let rng = new MathNet.Numerics.Random.MersenneTwister(42)

        let xv = ref (Array.zeroCreate(Nv)) //(inputs.[rng.Next(0,inputs.[0].Length)] )
        let xh = ref (Array.zeroCreate(Nh))
        let r=
            [0 .. n]
            |>Seq.map(fun i ->     
                    xv := inputs.[rng.Next(0,inputs.Length-1)]
                    [0 .. k-1 ]
                    |> Seq.iter(fun _ ->
                                         xh := rbm.GenerateOutput(!xv)
                                         xv := rbm.GenerateInput(!xh)
                                         )
                    !xv
                )
            |> Array.ofSeq
        r