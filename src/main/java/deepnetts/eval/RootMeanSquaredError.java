package deepnetts.eval;

/**
 * A measure of error for regression tasks.
 * Represents average error for all predictions.
 * 
 * @author Zoran Sevarac
 */
public class RootMeanSquaredError {
    
    private float totalSum;
    private float patternCount=0;
    
    public void add(float[] predicted, float[] target) {
        for(int i=0; i<predicted.length; i++)
            totalSum += Math.pow((predicted[i] - target[i]), 2);
        
        patternCount++;
    }
    
    
    public float getTotal() {
        return (float)Math.sqrt(totalSum / patternCount);
    }
    
}