package deepnetts.accl.spi;

import deepnetts.accl.AcceleratorHandle;
import deepnetts.accl.AcceleratorTensorBridge;
import deepnetts.net.layers.Backward;
import deepnetts.net.layers.ConvolutionalLayer;
import deepnetts.net.layers.FlattenLayer;
import deepnetts.net.layers.Forward;
import deepnetts.net.layers.FullyConnectedLayer;
import deepnetts.net.layers.MaxPoolingLayer;
import deepnetts.net.layers.OutputLayer;
import deepnetts.net.layers.SoftmaxOutputLayer;
import deepnetts.tensor.TensorBase;
import java.lang.foreign.Arena;

public interface AcceleratorProvider {
    
    public abstract AcceleratorHandle createAcceleratorHandle(Arena arena);
    public abstract AcceleratorTensorBridge createAcceleratorTensorBridge(TensorBase tensor);
    public abstract Forward createFullyConnectedForwardAcc(AcceleratorHandle handle, FullyConnectedLayer layer);
    public abstract Backward createFullyConnectedBackwardAcc(AcceleratorHandle handle, FullyConnectedLayer layer);
    public abstract Forward createFlattenForwardAcc(AcceleratorHandle handle, FlattenLayer layer);
    public abstract Backward createFlattenBackwardAcc(AcceleratorHandle handle, FlattenLayer layer);
    public abstract Forward createOutputForwardAcc(AcceleratorHandle handle, OutputLayer layer);
    public abstract Backward createOutputBackwardAcc(AcceleratorHandle handle, OutputLayer layer);    
    public abstract Forward createSoftmaxOutputForwardAcc(AcceleratorHandle handle, SoftmaxOutputLayer layer);
    public abstract Backward createSoftmaxOutputBackwardAcc(AcceleratorHandle handle, SoftmaxOutputLayer layer);  
    public abstract Forward createMaxpoolingForwardAcc(AcceleratorHandle handle, MaxPoolingLayer layer);
    public abstract Backward createMaxpoolingBackwardAcc(AcceleratorHandle handle, MaxPoolingLayer layer);  
    public abstract Forward createConvolutionalForwardAcc(AcceleratorHandle handle, ConvolutionalLayer layer);
    public abstract Backward createConvolutionalBackwardAcc(AcceleratorHandle handle, ConvolutionalLayer layer);      
      
}
