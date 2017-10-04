// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.

open MARIA.RBM
open MathNet.Numerics.LinearAlgebra
open Accord.Neuro
open Accord.Neuro.ActivationFunctions
open Accord.Neuro.Networks
open Accord.Neuro.Learning

[<EntryPoint>]
let main argv = 
    printfn "%A" argv
    let rbm = new CRBM(2,11)
    let rhoM = 
        let rho = Matrix<float>.Build.Dense(2,2)

        rho.[0,0] <- 1.0
        rho.[0,1] <- 0.8
        //rho.[0,2] <- 0.3
        //rho.[0,3] <- 0.4

        rho.[1,0] <- rho.[0,1]
        rho.[1,1] <- 1.0
        //rho.[1,2] <- 0.5
        //rho.[1,3] <- 0.15

        //rho.[2,0] <- rho.[0,2]
        //rho.[2,1] <- rho.[1,2]
        //rho.[2,2] <- 1.0
        //rho.[2,3] <- 0.2    

        //rho.[3,0] <- rho.[0,3]
        //rho.[3,1] <- rho.[1,3]
        //rho.[3,2] <- rho.[2,3]
        //rho.[3,3] <- 1.0    

        rho

    let genData (N:int) =        
        let mu =  [|3.0;0.0|]
        let mu2 = [|-3.0;0.0|]

        let s = System.Math.Sqrt(0.5)
        let sigma = Matrix<float>.Build.DiagonalOfDiagonalArray([|s;s|])
        
        let C= sigma*rhoM*sigma
        
        let phi = new Accord.Statistics.Distributions.Multivariate.MultivariateNormalDistribution(mu,C.ToArray())
        let phi2 = new Accord.Statistics.Distributions.Multivariate.MultivariateNormalDistribution(mu2,C.ToArray())
                
        let a = phi.Generate(N)
        let b = phi2.Generate(N)
        
        let c = [0..N-1] |> Seq.map(fun i -> if i%2 = 0 then a.[i] else b.[i])   |> Array.ofSeq
        (Matrix<float>.Build.DenseOfColumnArrays(c))

    let data = genData 2000
    //let rmb2 = new CRBMACCORD(2,13,data.Transpose().ToRowArrays())
    //rmb2.Run(2000)
    rbm.learn(data)(1000,1)
    let Z=data.Transpose()
    use writer=new System.IO.StreamWriter(@"E:\crbm_input.csv")
    for i in 0 .. Z.RowCount-1 do
        for j in 0 .. Z.ColumnCount-1 do   
            writer.Write(Z.[i,j].ToString()+",")
        writer.WriteLine()


    //rbm.learn(data)(1500,10) |> ignore
    let Zn = rbm.reconstruct(data)(10000,1)
    let Z=Zn.Transpose()
    //let Z = Matrix<float>.Build.DenseOfRowArrays( rmb2.sample(500,20))

    //let mutable mu = Vector<float>.Build.Dense(2)
    
    use writer=new System.IO.StreamWriter(@"E:\crbm.csv")
    for i in 0 .. Z.RowCount-1 do
    //    mu <- mu + Z.Row(i)
        for j in 0 .. Z.ColumnCount-1 do   
            writer.Write(Z.[i,j].ToString()+",")
        writer.WriteLine()

    //mu <- mu / float(Z.RowCount)
    //printf "u = (%f,%f)" mu.[0] mu.[1]

    let C = MathNet.Numerics.Statistics.Correlation.PearsonMatrix(Z.ToColumnArrays())
    let C0 = MathNet.Numerics.Statistics.Correlation.PearsonMatrix(data.ToRowArrays())
    
    
    printfn "Sampled Matrix= %A" (C.ToString())
    printfn "Actual Matrix= %A" (C0.ToString())
    
    0 // return an integer exit code
