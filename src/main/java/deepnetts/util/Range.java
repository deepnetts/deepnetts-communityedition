package deepnetts.util;

public final class Range {
    private final float min, max;

    public Range(float min, float max) {
        this.min = min;
        this.max = max;
    }   
    
    public float min() {
        return min;
    }

    public float max() {
        return max;
    }
    
    public static Range of(float min, float max) {
        return new Range(min, max);
    }
    
}
