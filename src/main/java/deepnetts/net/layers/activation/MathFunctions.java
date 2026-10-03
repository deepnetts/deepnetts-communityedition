package deepnetts.net.layers.activation;

/**
 * Misc math utility functions.
 */
public class MathFunctions {
   
    // prevent instantiation
    private MathFunctions() { }
                
    /**
     * First derivative of the abs function.
     * 
     * @param w
     * @return 
     */
    public static  final float absPrime(float w) {
        if (w==0) return 0;
        
        return (w>0?1:-1);
    }
}
