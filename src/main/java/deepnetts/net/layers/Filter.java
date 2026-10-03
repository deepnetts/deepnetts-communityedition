package deepnetts.net.layers;

/**
 * Settings of a convolutional filter which is used to learn to detect pixel patterns.
 * Used in ConvolutionalLayer.
 * size
 * width
 * height
 * stride
 * padding
 * groups
 *
 * @see deepnetts.net.layers.ConvolutionalLayer
 * @see deepnetts.net.ConvolutionalNetwork
 */
public class Filter {
    private final int width, height;
    private int stride=1, padding, groups=1;

    public Filter(int size) {
        this.width = size;
        this.height = size;
    }    
    
    public Filter(int width, int height) {
        this.width = width;
        this.height = height;
    }
 
    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public int getStride() {
        return stride;
    }

    public int getPadding() {
        return padding;
    }

    public int getGroups() {
        return groups;
    }        
               
    public Filter stride(int stride) {
        this.stride = stride;
        return this;
    }
    
    public Filter padding(int padding) {
        this.padding = padding;
        return this;
    }    
    
    public Filter groups(int groups) {
        this.groups = groups;
        return this;
    }      
          
        
}
