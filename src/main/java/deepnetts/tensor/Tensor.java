package deepnetts.tensor;

/**
 * A wrapper for multidimensional array/tensor.
 * Provides commonly used generic methods for accessing number of dimensions,
 * number of elements, shape and values/elements.
 */
public interface Tensor {
       
    /**
     * Returns a number of dimensions of this tensor
     * 
     * @return 
     */
    public int numDimensions();
    
    
    /**
     * Returns a total number of elements in this tensor. 
     * @return 
     */
    public int numElements();
    
    /**
     * Returns a shape of this tensor.
     * @return 
     */
    public Shape shape();    
    
    /**
     * Returns an array of values/elements stored in this tensor.
     * @return 
     */
    public float[] getValues();
    
    //TBD
    // public Layout getLayout(); row major, col major, nchw ...
}
