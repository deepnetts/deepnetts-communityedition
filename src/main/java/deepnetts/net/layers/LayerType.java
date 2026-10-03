package deepnetts.net.layers;

/**
 * Supported types of layers.
 * Layers are main processing and building block in a neural network.
 * 
 * @see Layer
 * @see AbstractLayer
 *

 */
public enum LayerType {
    /**
     * Accepts external input to the network.
     */
    INPUT("INPUT"),
    
    /**
     * Performs detection of pixel patterns.
     */
    CONVOLUTIONAL("CONVOLUTIONAL"),
    
    /**
     * Downsizes the input.
     */
    MAXPOOLING("MAXPOOLING"),
    
    /**
     * Looks for patterns in inputs.
     */
    FULLY_CONNECTED("FULLY_CONNECTED"),
    
    /**
     * Provides final output of the network.
     */
    OUTPUT("OUTPUT");

    private final String name;

    private LayerType(String s) {
        name = s;
    }

    public boolean equalsName(String otherName) {
        return name.equals(otherName);
    }

    @Override
    public String toString() {
        return this.name;
    }

}
