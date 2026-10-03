package deepnetts.net.layers;


public class Filters {
    
    
    /**
     * Factory method that creates filter settings with  specified size (using same size for filter width and height).
     * @param size size of the filter (same width and height)
     * 
     * @return filter with specified size
     */
    public static Filter ofSize(int size) {
        Filter filter = new Filter(size);
        return filter;
    }    
    
    /**
     * Factory method that creates a filter settings with specified width and height
     * @param width width of the filter (number of filter columns)
     * @param height height of the filter (number of filter rows)
     * 
     * @return filter with specified size
     */
    public static Filter ofSize(int width, int height) {
        Filter filter = new Filter(width, height);
        return filter;
    } 
    
    
       
}
