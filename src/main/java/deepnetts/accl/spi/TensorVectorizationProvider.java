package deepnetts.accl.spi;

import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;

public interface TensorVectorizationProvider {   
    public Tensor1D addVectorized(Tensor1D addTo, Tensor1D toAdd);
    public Tensor2D addVectorized(Tensor2D addTo, Tensor1D toAdd);
    
    public void outerProductVectorized(Tensor1D firstTensor, Tensor1D otherTensor, Tensor2D result);
//    public Tensor1D matMul(Tensor1D vectorA,Tensor1D vectorB, Tensor1D result);
//    public Tensor2D matMul(Tensor2D matrixA, Tensor2D matrixB, Tensor2D result);
    
    public Tensor1D matMulFma(Tensor2D matrixA, Tensor1D vector, Tensor1D result);
    public Tensor2D matMulFma(Tensor2D matrixA, Tensor2D matrixB, Tensor2D result);
    
    public Tensor1D matMulFmaParallel(Tensor2D matrixA, Tensor1D vector, Tensor1D result);
    public Tensor2D matMulFmaParallel(Tensor2D matrixA, Tensor2D matrixB, Tensor2D result);
         
    public Tensor1D matMulWithAddVector(Tensor2D matrixA, Tensor1D vector, Tensor1D result);
    public Tensor2D matMulWithAddVector(Tensor2D matrixA, Tensor2D matrixB, Tensor2D result);
    
    public Tensor1D matMulWithAddVectorParallel(Tensor2D matrixA, Tensor1D vector, Tensor1D result);
    public Tensor2D matMulWithAddVectorParallel(Tensor2D matrixA, Tensor2D matrixB, Tensor2D result);
        
}
