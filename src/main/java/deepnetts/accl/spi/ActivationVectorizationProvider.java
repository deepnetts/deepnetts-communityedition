
package deepnetts.accl.spi;

import deepnetts.tensor.Tensor;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;


public interface ActivationVectorizationProvider {    
    public void relu(Tensor tensor);
    public void tanh(Tensor tensor);
    public void sigmoid(Tensor tensor);
    public void leakyRelu(Tensor tensor, float a);
    public void applySoftmaxVectorized(Tensor1D outputs1D, float maxWs);
    public void applySoftmaxBatchVectorized(Tensor2D logits);
}
